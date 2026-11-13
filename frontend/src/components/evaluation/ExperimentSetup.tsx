import { CircleCheck, CircleDashed, FlaskConical, KeyRound, LoaderCircle, PackagePlus } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'
import { ApiError, api } from '../../services/api'
import type { EvaluationDataset, EvaluationRunSummary, QuestionCategory, SearchMethod } from '../../types/api'
import {
  CATEGORY_DESCRIPTIONS,
  CATEGORY_LABELS,
  METHOD_LABELS,
  METHOD_ORDER,
  methodColor,
} from '../../utils/evaluation'
import { Button } from '../ui/Button'
import { Card, CardHeader } from '../ui/Card'
import { ErrorState, Spinner } from '../ui/Feedback'

interface ExperimentSetupProps {
  dataset: EvaluationDataset
  llmConfigured: boolean
  /** A run is pending or running, so another can't start. */
  busy: boolean
  onInstalled: () => void
  onStarted: (run: EvaluationRunSummary) => void
}

function PaperStatus({ status }: { status: EvaluationDataset['papers'][number]['indexing_status'] }) {
  if (status === 'indexed') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-fg-muted">
        <CircleCheck className="size-3.5 text-success" aria-hidden /> Searchable
      </span>
    )
  }
  if (status === 'indexing' || status === 'not_indexed') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-fg-muted">
        <LoaderCircle className="size-3.5 animate-spin text-fg-subtle" aria-hidden /> Indexing…
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs text-fg-subtle">
      <CircleDashed className="size-3.5" aria-hidden /> {status === 'failed' ? 'Indexing failed' : 'Not in library'}
    </span>
  )
}

export function ExperimentSetup({ dataset, llmConfigured, busy, onInstalled, onStarted }: ExperimentSetupProps) {
  const [methods, setMethods] = useState<SearchMethod[]>(METHOD_ORDER)
  const [judgeAnswers, setJudgeAnswers] = useState(false)
  const [installing, setInstalling] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const missing = dataset.papers.filter((paper) => paper.document_id === null)

  const toggle = (method: SearchMethod) =>
    setMethods((current) =>
      current.includes(method)
        ? current.filter((item) => item !== method)
        : METHOD_ORDER.filter((item) => item === method || current.includes(item)),
    )

  const install = async () => {
    setInstalling(true)
    setError(null)
    try {
      await api.installSamplePapers()
      onInstalled()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause : new ApiError(0, 'unknown', 'Something went wrong.'))
    } finally {
      setInstalling(false)
    }
  }

  const start = async () => {
    setStarting(true)
    setError(null)
    try {
      onStarted(
        await api.startEvaluation({
          dataset: dataset.slug,
          methods,
          answer_quality: judgeAnswers && llmConfigured,
        }),
      )
    } catch (cause) {
      setError(cause instanceof ApiError ? cause : new ApiError(0, 'unknown', 'Something went wrong.'))
    } finally {
      setStarting(false)
    }
  }

  return (
    <Card>
      <CardHeader
        title="Run an evaluation"
        description="Measure how well each retrieval method finds the evidence for a curated set of questions, on a fixed collection of papers."
      />
      <div className="grid gap-6 p-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section aria-labelledby="dataset-heading" className="space-y-3">
          <div>
            <h3 id="dataset-heading" className="text-sm font-semibold text-fg">
              Dataset: {dataset.name}
            </h3>
            <p className="mt-1 text-xs text-fg-muted">{dataset.description}</p>
          </div>
          <ul className="flex flex-wrap gap-1.5">
            {(Object.entries(dataset.categories) as [QuestionCategory, number][]).map(([category, count]) => (
              <li
                key={category}
                title={CATEGORY_DESCRIPTIONS[category]}
                className="rounded-md bg-surface-muted px-2 py-0.5 text-xs text-fg-muted"
              >
                {CATEGORY_LABELS[category]} <span className="font-semibold text-fg">{count}</span>
              </li>
            ))}
          </ul>
          <div>
            <p className="text-xs font-medium text-fg-muted">Papers</p>
            <ul className="mt-1.5 space-y-1.5">
              {dataset.papers.map((paper) => (
                <li key={paper.key} className="flex items-start justify-between gap-3 text-sm">
                  <span className="min-w-0 text-fg">{paper.title}</span>
                  <span className="shrink-0">
                    <PaperStatus status={paper.indexing_status} />
                  </span>
                </li>
              ))}
            </ul>
          </div>
          {missing.length > 0 && dataset.sample_collection && (
            <div className="rounded-lg border border-dashed border-border px-3 py-3">
              <p className="text-xs text-fg-muted">
                These are short synthetic papers bundled with ResearchMate (fictional methods and
                results), so every run measures the same collection. Adding them puts them in your
                library like any upload; you can delete them afterwards.
              </p>
              <Button size="sm" className="mt-2" onClick={() => void install()} disabled={installing}>
                {installing ? <Spinner className="text-accent-fg" /> : <PackagePlus className="size-3.5" aria-hidden />}
                Add the {missing.length === dataset.papers.length ? '' : `${missing.length} missing `}
                sample papers
              </Button>
            </div>
          )}
        </section>

        <section aria-labelledby="methods-heading" className="space-y-4">
          <fieldset>
            <legend id="methods-heading" className="text-sm font-semibold text-fg">
              Retrieval methods
            </legend>
            <div className="mt-2 grid gap-1.5 sm:grid-cols-2">
              {METHOD_ORDER.map((method) => (
                <label key={method} className="flex items-center gap-2 text-sm text-fg">
                  <input
                    type="checkbox"
                    checked={methods.includes(method)}
                    onChange={() => toggle(method)}
                    className="size-4 accent-accent"
                  />
                  <span className="size-2.5 rounded-sm" style={{ background: methodColor(method) }} aria-hidden />
                  {METHOD_LABELS[method]}
                </label>
              ))}
            </div>
            <p className="mt-1.5 text-xs text-fg-subtle">
              Retrieval metrics use only the search pipeline: no language model is involved.
            </p>
          </fieldset>

          <div>
            <label className="flex items-start gap-2 text-sm text-fg">
              <input
                type="checkbox"
                checked={judgeAnswers && llmConfigured}
                disabled={!llmConfigured}
                onChange={(event) => setJudgeAnswers(event.target.checked)}
                className="mt-0.5 size-4 accent-accent"
              />
              <span>
                Judge answers too
                <span className="block text-xs text-fg-muted">
                  Answers every question with the full pipeline, checks refusals, and has the
                  model score faithfulness, relevance and correctness. About two model calls per
                  question.
                </span>
              </span>
            </label>
            {!llmConfigured && (
              <p className="mt-1.5 flex items-start gap-1.5 pl-6 text-xs text-fg-muted">
                <KeyRound className="mt-0.5 size-3.5 shrink-0 text-warning" aria-hidden />
                <span>
                  Needs a Gemini API key (see{' '}
                  <Link to="/settings" className="text-accent-text hover:underline">
                    Settings
                  </Link>
                  ).
                </span>
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={() => void start()}
              disabled={!dataset.ready || busy || starting || methods.length === 0}
            >
              {starting ? <Spinner className="text-accent-fg" /> : <FlaskConical className="size-4" aria-hidden />}
              Run evaluation
            </Button>
            {!dataset.ready && (
              <span className="text-xs text-fg-subtle">All papers must be searchable first.</span>
            )}
            {busy && dataset.ready && <span className="text-xs text-fg-subtle">A run is in progress.</span>}
          </div>
          {error && <ErrorState title="Couldn't start the evaluation" message={error.message} />}
        </section>
      </div>
    </Card>
  )
}
