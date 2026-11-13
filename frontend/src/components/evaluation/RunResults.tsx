import { useEffect, useRef, useState } from 'react'
import { useApi } from '../../hooks/useApi'
import { api } from '../../services/api'
import type { EvaluationRunDetail, EvaluationRunSummary } from '../../types/api'
import { formatDate } from '../../utils/format'
import { Badge } from '../ui/Badge'
import { Card, CardHeader } from '../ui/Card'
import { ErrorState, Skeleton, Spinner } from '../ui/Feedback'
import { AnswerQuality } from './AnswerQuality'
import { MetricsTable } from './MetricsTable'
import { QuestionResults } from './QuestionResults'
import { CategoryChart, MethodLegend, MetricsAtKChart, RecallCurveChart } from './RetrievalCharts'

const POLL_MS = 1500

function isActive(run: EvaluationRunSummary): boolean {
  return run.status === 'pending' || run.status === 'running'
}

function Progress({ run }: { run: EvaluationRunSummary }) {
  const progress = run.metrics.progress
  const phase =
    progress?.phase === 'answers'
      ? 'Answering and judging questions'
      : progress?.phase === 'retrieval'
        ? 'Searching with each method'
        : 'Starting'
  const share = progress && progress.total > 0 ? progress.done / progress.total : 0
  return (
    <Card className="p-5">
      <div className="flex items-center gap-3 text-sm text-fg">
        <Spinner className="size-5" />
        <span className="font-medium">{phase}…</span>
        {progress && progress.total > 0 && (
          <span className="text-fg-muted tabular-nums">
            {progress.done} of {progress.total} questions
          </span>
        )}
      </div>
      <div
        className="mt-3 h-1.5 overflow-hidden rounded-full bg-surface-muted"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(share * 100)}
      >
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${share * 100}%` }} />
      </div>
    </Card>
  )
}

function Results({ run }: { run: EvaluationRunDetail }) {
  const [k, setK] = useState(5)
  const retrieval = run.metrics.retrieval
  const questions = retrieval?.[run.config.methods[0]]?.questions ?? 0
  const settings = run.config.settings

  return (
    <div className="space-y-6">
      {retrieval && (
        <Card>
          <CardHeader
            title="Retrieval metrics"
            description={`${questions} answerable questions · each method retrieves the top ${run.config.top_k} passages · a passage is relevant if it contains the question's evidence`}
            action={<Badge tone="accent">No LLM involved</Badge>}
          />
          <div className="space-y-6 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <MethodLegend methods={run.config.methods} mark="bar" />
              <label className="flex items-center gap-2 text-xs text-fg-muted">
                Cut-off K
                <select
                  value={k}
                  onChange={(event) => setK(Number(event.target.value))}
                  className="h-8 rounded-lg border border-border bg-surface px-2 text-xs text-fg"
                >
                  {run.config.cutoffs.map((cutoff) => (
                    <option key={cutoff} value={cutoff}>
                      {cutoff}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="grid gap-8 lg:grid-cols-2">
              <MetricsAtKChart run={run} k={k} />
              <RecallCurveChart run={run} />
            </div>
            <CategoryChart run={run} />
            <div>
              <h3 className="text-sm font-semibold text-fg">All numbers</h3>
              <div className="mt-2 rounded-lg border border-border">
                <MetricsTable run={run} />
              </div>
            </div>
            <p className="text-xs text-fg-subtle">
              Measured with {String(settings.embedding_model)} (dense), {String(settings.reranker_model)}{' '}
              (rerank), {String(settings.chunk_size_tokens)}-token chunks with{' '}
              {String(settings.chunk_overlap_tokens)} overlap, RRF k = {String(settings.rrf_k)}. Results
              describe this small collection, not retrieval in general.
            </p>
          </div>
        </Card>
      )}

      <AnswerQuality run={run} />

      <Card className="overflow-hidden">
        <CardHeader title="Questions" description="Select a question for its evidence and, if judged, its answer." />
        <QuestionResults run={run} />
      </Card>
    </div>
  )
}

/** One run: progress while it runs, then its results. */
export function RunResults({ runId, onFinished }: { runId: string; onFinished: () => void }) {
  const run = useApi((signal) => api.evaluationRun(runId, signal), {
    pollMs: POLL_MS,
    pollWhile: isActive,
  })
  // Refresh the run list once this run finishes.
  const wasActive = useRef(false)
  const active = run.data ? isActive(run.data) : false
  useEffect(() => {
    if (wasActive.current && !active) onFinished()
    wasActive.current = active
  }, [active, onFinished])

  if (run.error) {
    return <ErrorState title="Couldn't load this run" message={run.error.message} onRetry={run.reload} />
  }
  if (!run.data) return <Skeleton className="h-64 w-full rounded-xl" />

  const data = run.data
  return (
    <section aria-label="Evaluation results" className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold text-fg">
          Run of {formatDate(data.created_at)}
          <span className="ml-2 font-normal text-fg-subtle">
            {new Date(data.created_at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
          </span>
        </h2>
      </div>
      {isActive(data) && <Progress run={data} />}
      {data.status === 'failed' && (
        <ErrorState title="The run did not finish" message={data.error_message ?? 'Unknown error.'} />
      )}
      {!isActive(data) && <Results run={data} />}
    </section>
  )
}
