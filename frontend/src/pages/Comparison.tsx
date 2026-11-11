import { GitCompareArrows, KeyRound, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { CitationDialog, type CitedPassage } from '../components/research/CitationDialog'
import { ComparisonTable } from '../components/research/ComparisonTable'
import { FollowUpChat } from '../components/research/FollowUpChat'
import { PaperPicker } from '../components/research/PaperPicker'
import { Button, ButtonLink } from '../components/ui/Button'
import { Card, CardHeader } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { ErrorState, Skeleton, Spinner } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { ApiError, api } from '../services/api'
import type { ComparisonResponse } from '../types/api'
import { pluralize } from '../utils/format'

// Used until /api/system/info has loaded; the server enforces the real limit.
const DEFAULT_MAX_PAPERS = 5

function toApiError(cause: unknown): ApiError {
  return cause instanceof ApiError ? cause : new ApiError(0, 'unknown', 'Something went wrong.')
}

function sameIds(first: string[], second: string[]): boolean {
  return first.length === second.length && first.every((id, index) => id === second[index])
}

export default function Comparison() {
  const [searchParams, setSearchParams] = useSearchParams()
  const documents = useApi(api.listDocuments)
  const { data: info } = useApi(api.systemInfo)
  const { data: health } = useApi(api.health)
  const maxPapers = info?.analysis.max_compare_papers ?? DEFAULT_MAX_PAPERS
  const searchable = (documents.data?.items ?? []).filter((doc) => doc.indexing_status === 'indexed')

  // Papers in the URL are compared again on load (e.g. after a reload). Their
  // profiles are saved, so this makes no new model calls.
  const [initialIds] = useState(() => (searchParams.get('papers') ?? '').split(',').filter(Boolean))
  const [selected, setSelected] = useState<string[]>(initialIds)
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null)
  const [running, setRunning] = useState(initialIds.length >= 2)
  const [error, setError] = useState<ApiError | null>(null)
  const [viewing, setViewing] = useState<CitedPassage | null>(null)

  const run = async (ids: string[], refresh = false) => {
    setRunning(true)
    setError(null)
    try {
      const result = await api.compare(ids, refresh)
      setComparison(result)
      setSearchParams({ papers: ids.join(',') }, { replace: true })
    } catch (cause) {
      setError(toApiError(cause))
    } finally {
      setRunning(false)
    }
  }

  useEffect(() => {
    if (initialIds.length < 2) return
    const controller = new AbortController()
    api
      .compare(initialIds, false, controller.signal)
      .then(setComparison)
      .catch((cause) => {
        if (!controller.signal.aborted) setError(toApiError(cause))
      })
      .finally(() => {
        if (!controller.signal.aborted) setRunning(false)
      })
    return () => controller.abort()
  }, [initialIds])

  const compared = comparison?.papers.map((paper) => paper.document_id) ?? []
  const reused = comparison?.papers.filter((paper) => paper.cached).length ?? 0
  const canCompare = selected.length >= 2 && !running && !sameIds(selected, compared)

  if (documents.data && searchable.length < 2) {
    return (
      <Card>
        <EmptyState
          icon={GitCompareArrows}
          title="Two searchable papers needed"
          description="Upload at least two papers and wait until they show as Searchable, then compare their objectives, methods, datasets and results side by side."
          action={<ButtonLink to="/papers">Go to papers</ButtonLink>}
        />
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Choose papers to compare"
          description="Each paper is read once: its evidence is retrieved aspect by aspect and summarised with citations, then saved, so later comparisons are instant."
        />
        <div className="space-y-4 p-5">
          {documents.error ? (
            <ErrorState
              title="Couldn't load your papers"
              message={documents.error.message}
              onRetry={documents.reload}
            />
          ) : documents.loading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <PaperPicker
              papers={searchable}
              selected={selected}
              max={maxPapers}
              onChange={setSelected}
            />
          )}
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => void run(selected)} disabled={!canCompare}>
              {running ? (
                <Spinner className="text-accent-fg" />
              ) : (
                <GitCompareArrows className="size-4" aria-hidden />
              )}
              Compare {selected.length >= 2 ? pluralize(selected.length, 'paper') : 'papers'}
            </Button>
            {selected.length < 2 && (
              <span className="text-xs text-fg-subtle">Select at least two papers.</span>
            )}
          </div>
          {health && !health.llm_configured && (
            <p className="flex items-start gap-2 text-xs text-fg-muted">
              <KeyRound className="mt-0.5 size-3.5 shrink-0 text-warning" aria-hidden />
              <span>
                Comparing papers that haven't been summarised needs a Gemini API key in{' '}
                <code className="font-mono">backend/.env</code> (see{' '}
                <Link to="/settings" className="text-accent-text hover:underline">
                  Settings
                </Link>
                ).
              </span>
            </p>
          )}
        </div>
      </Card>

      {error && (
        <ErrorState
          title={
            error.code === 'llm_not_configured' ? 'Gemini API key needed' : "Couldn't compare the papers"
          }
          message={error.message}
        />
      )}

      {running && (
        <Card className="flex items-start gap-3 p-5 text-sm text-fg-muted">
          <Spinner className="mt-0.5 size-5" />
          <div>
            <p className="font-medium text-fg">Comparing {pluralize(selected.length, 'paper')}…</p>
            <p className="mt-1">
              Papers read before are reused instantly. Each new paper takes 10–30 seconds; they are
              read in parallel.
            </p>
          </div>
        </Card>
      )}

      {comparison && !running && (
        <>
          <Card className="overflow-hidden">
            <CardHeader
              title="Comparison"
              description={`${pluralize(comparison.papers.length, 'paper')} · ${comparison.aspects.length} aspects · select a page to read the passage it cites${
                reused > 0 ? ` · ${reused} of ${comparison.papers.length} reused from saved summaries` : ''
              }`}
              action={
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => void run(compared, true)}
                  title="Read the papers again and rebuild every column"
                >
                  <RefreshCw className="size-3.5" aria-hidden />
                  Regenerate
                </Button>
              }
            />
            <ComparisonTable comparison={comparison} onCite={setViewing} />
          </Card>

          <Card>
            <CardHeader
              title="Ask about these papers"
              description="Answers search only the compared papers, with each paper's best passages included, and cite them."
            />
            <div className="p-5">
              <FollowUpChat key={compared.join(',')} documentIds={compared} />
            </div>
          </Card>
        </>
      )}

      <CitationDialog passage={viewing} onClose={() => setViewing(null)} />
    </div>
  )
}
