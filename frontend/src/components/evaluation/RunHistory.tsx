import { Trash2 } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../services/api'
import type { EvaluationRunSummary, SearchMethod } from '../../types/api'
import { METHOD_LABELS, formatScore } from '../../utils/evaluation'
import { cn, formatRelativeTime } from '../../utils/format'
import { Badge, type BadgeTone } from '../ui/Badge'
import { Card, CardHeader } from '../ui/Card'
import { ConfirmDialog } from '../ui/ConfirmDialog'

const STATUS: Record<EvaluationRunSummary['status'], { label: string; tone: BadgeTone }> = {
  pending: { label: 'Starting', tone: 'neutral' },
  running: { label: 'Running', tone: 'accent' },
  completed: { label: 'Completed', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
}

function bestMethod(run: EvaluationRunSummary): [SearchMethod, number] | null {
  const retrieval = run.metrics.retrieval
  if (!retrieval) return null
  const scored = run.config.methods
    .map((method) => [method, retrieval[method]?.mrr] as const)
    .filter((entry): entry is readonly [SearchMethod, number] => typeof entry[1] === 'number')
  if (scored.length === 0) return null
  const [method, mrr] = scored.reduce((best, entry) => (entry[1] > best[1] ? entry : best))
  return [method, mrr]
}

interface RunHistoryProps {
  runs: EvaluationRunSummary[]
  selectedId: string | null
  onSelect: (runId: string) => void
  onDeleted: (runId: string) => void
}

/** Earlier runs, so changes to settings or models can be compared over time. */
export function RunHistory({ runs, selectedId, onSelect, onDeleted }: RunHistoryProps) {
  const [pendingDelete, setPendingDelete] = useState<EvaluationRunSummary | null>(null)
  const [deleting, setDeleting] = useState(false)

  const confirmDelete = async () => {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await api.deleteEvaluationRun(pendingDelete.id)
      onDeleted(pendingDelete.id)
    } finally {
      setDeleting(false)
      setPendingDelete(null)
    }
  }

  return (
    <Card>
      <CardHeader title="Run history" description="Each run records the models and settings it used." />
      <ul className="divide-y divide-border">
        {runs.map((run) => {
          const best = bestMethod(run)
          const settings = run.config.settings
          return (
            <li
              key={run.id}
              className={cn('flex items-center gap-3 px-5 py-3', run.id === selectedId && 'bg-accent-soft')}
            >
              <button
                type="button"
                onClick={() => onSelect(run.id)}
                aria-current={run.id === selectedId}
                className="min-w-0 flex-1 text-left"
              >
                <span className="flex flex-wrap items-center gap-2 text-sm text-fg">
                  {formatRelativeTime(run.created_at)}
                  <Badge tone={STATUS[run.status].tone}>{STATUS[run.status].label}</Badge>
                  {run.config.answer_quality && <Badge tone="warning">Answers judged</Badge>}
                </span>
                <span className="mt-0.5 block truncate text-xs text-fg-subtle">
                  {run.dataset_name} · {run.config.methods.length} methods ·{' '}
                  {String(settings.chunk_size_tokens)}-token chunks
                  {best && ` · best MRR ${formatScore(best[1])} (${METHOD_LABELS[best[0]]})`}
                </span>
              </button>
              <button
                type="button"
                onClick={() => setPendingDelete(run)}
                disabled={run.status === 'pending' || run.status === 'running'}
                className="rounded-md p-1.5 text-fg-subtle hover:bg-danger-soft hover:text-danger disabled:opacity-40"
                aria-label="Delete run"
              >
                <Trash2 className="size-4" />
              </button>
            </li>
          )
        })}
      </ul>
      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete this run?"
        description="Its stored metrics and per-question results will be removed. This can't be undone."
        confirmLabel="Delete run"
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </Card>
  )
}
