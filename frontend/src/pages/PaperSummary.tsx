import { ArrowLeft, BookOpenText, ExternalLink, KeyRound, RefreshCw, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router'
import { AspectContent } from '../components/research/AspectContent'
import { CitationDialog, type CitedPassage } from '../components/research/CitationDialog'
import { Button, ButtonLink } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { ErrorState, Skeleton, Spinner } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { ApiError, api } from '../services/api'
import type { PaperProfile } from '../types/api'
import { formatRelativeTime, pluralize } from '../utils/format'

const NOT_FOUND_TEXT = 'Not found in the passages retrieved from this paper.'

function SummarySections({
  summary,
  onCite,
}: {
  summary: PaperProfile
  onCite: (passage: CitedPassage) => void
}) {
  return (
    <Card>
      <ol className="divide-y divide-border">
        {summary.aspects.map((aspect, index) => (
          <li key={aspect.key} className="px-5 py-4">
            <h3 className="flex items-baseline gap-2 text-sm font-semibold text-fg">
              <span className="w-4 text-xs font-medium text-fg-subtle tabular-nums">{index + 1}</span>
              {aspect.label}
            </h3>
            <div className="mt-2 pl-6">
              <AspectContent
                aspect={aspect}
                profile={summary}
                notFoundText={NOT_FOUND_TEXT}
                noun="section"
                onCite={onCite}
              />
            </div>
          </li>
        ))}
      </ol>
    </Card>
  )
}

function SummaryView({ documentId }: { documentId: string }) {
  const document = useApi((signal) => api.getDocument(documentId, signal))
  const saved = useApi((signal) => api.getSummary(documentId, signal))
  const { data: health } = useApi(api.health)
  const [generated, setGenerated] = useState<PaperProfile | null>(null)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [viewing, setViewing] = useState<CitedPassage | null>(null)

  const summary = generated ?? saved.data
  const noSummary = saved.error?.code === 'summary_not_found'
  const title = document.data?.title ?? document.data?.filename ?? summary?.document_title
  const searchable = document.data?.indexing_status === 'indexed'

  const generate = async (refresh: boolean) => {
    setGenerating(true)
    setError(null)
    try {
      setGenerated(await api.summarize(documentId, refresh))
    } catch (cause) {
      setError(cause instanceof ApiError ? cause : new ApiError(0, 'unknown', 'Something went wrong.'))
    } finally {
      setGenerating(false)
    }
  }

  let body
  if (document.error) {
    body = (
      <ErrorState
        title="Couldn't open this paper"
        message={document.error.message}
        onRetry={document.reload}
      />
    )
  } else if (generating) {
    body = (
      <Card className="flex items-start gap-3 p-5 text-sm text-fg-muted">
        <Spinner className="mt-0.5 size-5" />
        <div>
          <p className="font-medium text-fg">Reading the paper…</p>
          <p className="mt-1">
            Retrieving the passages for each section, then writing a summary that cites them. This
            usually takes 10–30 seconds.
          </p>
        </div>
      </Card>
    )
  } else if (summary) {
    body = <SummarySections summary={summary} onCite={setViewing} />
  } else if (saved.loading || document.loading) {
    body = (
      <div className="space-y-3">
        {[0, 1, 2].map((row) => (
          <Skeleton key={row} className="h-24 w-full rounded-xl" />
        ))}
      </div>
    )
  } else if (saved.error && !noSummary) {
    body = (
      <ErrorState
        title="Couldn't load the summary"
        message={saved.error.message}
        onRetry={saved.reload}
      />
    )
  } else if (!searchable) {
    body = (
      <Card>
        <EmptyState
          icon={BookOpenText}
          title="This paper isn't searchable yet"
          description="Summaries are built from the paper's indexed passages. Wait for indexing to finish, or index it from My Research Papers."
          action={<ButtonLink to="/papers">Go to papers</ButtonLink>}
        />
      </Card>
    )
  } else {
    body = (
      <Card>
        <EmptyState
          icon={BookOpenText}
          title="No summary yet"
          description="Nine sections, from the research objective to future scope. Each one is retrieved from the paper separately, every statement cites its page, and anything the paper doesn't state is reported as not found."
          action={
            <Button onClick={() => void generate(false)}>
              <BookOpenText className="size-4" aria-hidden />
              Generate summary
            </Button>
          }
        />
      </Card>
    )
  }

  const found = summary?.aspects.filter((aspect) => aspect.status === 'found').length ?? 0

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <Link to="/papers" className="inline-flex items-center gap-1.5 text-sm text-fg-muted hover:text-fg">
        <ArrowLeft className="size-4" aria-hidden />
        My Research Papers
      </Link>

      <Card className="p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-accent-text">Paper summary</p>
            <h2 className="mt-1 text-lg leading-snug font-semibold text-fg">
              {title ?? <Skeleton className="h-6 w-2/3" />}
            </h2>
            {summary && (
              <p className="mt-1.5 text-xs text-fg-subtle">
                {found} of {summary.aspects.length} sections found · built from{' '}
                {pluralize(summary.sources.length, 'passage')} · generated{' '}
                {formatRelativeTime(summary.generated_at)} with {summary.model}
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <a
              href={api.documentFileUrl(documentId)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border bg-surface px-3 text-xs font-medium text-fg shadow-card hover:bg-surface-muted"
            >
              <ExternalLink className="size-3.5" aria-hidden />
              Open PDF
            </a>
            {summary && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => void generate(true)}
                disabled={generating || !searchable}
                title="Retrieve the passages again and rewrite the summary"
              >
                <RefreshCw className="size-3.5" aria-hidden />
                Regenerate
              </Button>
            )}
          </div>
        </div>
        {summary?.truncated && (
          <p className="mt-3 flex gap-1.5 text-xs text-warning">
            <TriangleAlert className="size-3.5" aria-hidden />
            The model hit its output limit, so some sections may be incomplete.
          </p>
        )}
      </Card>

      {error && (
        <ErrorState
          title={
            error.code === 'llm_not_configured'
              ? 'Gemini API key needed'
              : "Couldn't summarise this paper"
          }
          message={error.message}
        />
      )}
      {health && !health.llm_configured && !summary && !error && (
        <p className="flex items-start gap-2 text-xs text-fg-muted">
          <KeyRound className="mt-0.5 size-3.5 shrink-0 text-warning" aria-hidden />
          <span>
            Summaries need a Gemini API key in <code className="font-mono">backend/.env</code> (see{' '}
            <Link to="/settings" className="text-accent-text hover:underline">
              Settings
            </Link>
            ).
          </span>
        </p>
      )}

      {body}

      {summary && (
        <p className="text-xs text-fg-subtle">
          Each section was retrieved from the paper on its own, and every statement cites the page of
          its supporting sentence (select a page to read it). The summary is saved and reused until
          the paper is re-indexed.
        </p>
      )}

      <CitationDialog passage={viewing} onClose={() => setViewing(null)} />
    </div>
  )
}

export default function PaperSummary() {
  const { documentId = '' } = useParams()
  // A new key per paper resets the page's state when moving between papers.
  return <SummaryView key={documentId} documentId={documentId} />
}
