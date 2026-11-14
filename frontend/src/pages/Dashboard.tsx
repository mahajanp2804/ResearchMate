import {
  Activity,
  ArrowRight,
  BookOpenText,
  Database,
  FilePlus2,
  FileText,
  FlaskConical,
  GitCompareArrows,
  Layers,
  Lightbulb,
  MessageSquareText,
  RefreshCw,
  Trash2,
  type LucideIcon,
} from 'lucide-react'
import { Link } from 'react-router'
import { MetricCard } from '../components/MetricCard'
import { DocumentStatusBadge } from '../components/StatusBadge'
import { ButtonLink } from '../components/ui/Button'
import { Card, CardHeader } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { ErrorState, Skeleton } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'
import type { ActivityItem, ActivityType, RecentDocument } from '../types/api'
import { BUSY_POLL_MS, isDocumentBusy } from '../utils/documents'
import { formatNumber, formatRelativeTime, pluralize } from '../utils/format'
import { askAboutUrl, summaryUrl } from '../utils/links'

const QUICK_ACTIONS: { to: string; title: string; description: string; icon: LucideIcon }[] = [
  {
    to: '/papers',
    title: 'Upload papers',
    description: 'Add PDFs to your library and index them for search.',
    icon: FilePlus2,
  },
  {
    to: '/assistant',
    title: 'Ask a question',
    description: 'Get answers grounded in your papers, with page citations.',
    icon: MessageSquareText,
  },
  {
    to: '/papers',
    title: 'Summarise a paper',
    description: 'Open a paper’s Summary: nine sections, every statement cited.',
    icon: BookOpenText,
  },
  {
    to: '/comparison',
    title: 'Compare papers',
    description: 'Line up objectives, methods, datasets and results.',
    icon: GitCompareArrows,
  },
  {
    to: '/insights',
    title: 'Find research gaps',
    description: 'Stated limitations, shared patterns and possible directions.',
    icon: Lightbulb,
  },
  {
    to: '/evaluation',
    title: 'Evaluate retrieval',
    description: 'Measure how well each search method finds the evidence.',
    icon: FlaskConical,
  },
]

// Where each kind of event happened, and its icon.
const ACTIVITY: Record<ActivityType, { icon: LucideIcon; link: (event: ActivityItem) => string | null }> = {
  document_uploaded: { icon: FilePlus2, link: (e) => (e.document_id ? '/papers' : null) },
  document_reindexed: { icon: RefreshCw, link: (e) => (e.document_id ? '/papers' : null) },
  document_deleted: { icon: Trash2, link: () => null },
  question_asked: { icon: MessageSquareText, link: () => '/assistant' },
  summary_generated: {
    icon: BookOpenText,
    link: (e) => (e.document_id ? summaryUrl(e.document_id) : null),
  },
  comparison_run: { icon: GitCompareArrows, link: () => '/comparison' },
  gap_analysis_run: { icon: Lightbulb, link: () => '/insights' },
  evaluation_run: { icon: FlaskConical, link: () => '/evaluation' },
}

function ListSkeleton() {
  return (
    <div className="space-y-3 p-5">
      {[0, 1, 2].map((row) => (
        <Skeleton key={row} className="h-10 w-full" />
      ))}
    </div>
  )
}

function RecentDocuments({ documents }: { documents: RecentDocument[] }) {
  if (documents.length === 0) {
    return (
      <EmptyState
        icon={FileText}
        title="No papers yet"
        description="Upload research papers to start asking questions and comparing findings."
        action={
          <ButtonLink to="/papers" size="sm">
            <FilePlus2 className="size-3.5" aria-hidden />
            Upload papers
          </ButtonLink>
        }
      />
    )
  }
  return (
    <ul className="divide-y divide-border">
      {documents.map((document) => (
        <li key={document.id} className="flex items-center gap-3 px-5 py-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-surface-muted">
            <FileText className="size-4 text-fg-muted" aria-hidden />
          </div>
          <div className="min-w-0 flex-1">
            {document.indexing_status === 'indexed' ? (
              <Link
                to={summaryUrl(document.id)}
                className="block truncate text-sm font-medium text-fg hover:text-accent-text hover:underline"
                title="Open the summary"
              >
                {document.title ?? document.filename}
              </Link>
            ) : (
              <p className="truncate text-sm font-medium text-fg">
                {document.title ?? document.filename}
              </p>
            )}
            <p className="truncate text-xs text-fg-subtle">
              {document.page_count !== null && `${pluralize(document.page_count, 'page')} · `}
              {formatRelativeTime(document.created_at)}
            </p>
          </div>
          <DocumentStatusBadge
            processingStatus={document.processing_status}
            indexingStatus={document.indexing_status}
          />
          {document.indexing_status === 'indexed' && (
            <Link
              to={askAboutUrl([document.id])}
              className="shrink-0 rounded-md p-1.5 text-fg-subtle hover:bg-surface-muted hover:text-fg"
              aria-label={`Ask about ${document.title ?? document.filename}`}
              title="Ask about this paper"
            >
              <MessageSquareText className="size-4" aria-hidden />
            </Link>
          )}
        </li>
      ))}
    </ul>
  )
}

function RecentActivity({ events }: { events: ActivityItem[] }) {
  if (events.length === 0) {
    return (
      <EmptyState
        icon={Activity}
        title="No activity yet"
        description="Uploads, questions, comparisons and evaluations will appear here."
      />
    )
  }
  return (
    <ol className="space-y-1 px-3 py-3">
      {events.map((event) => {
        const { icon: Icon, link } = ACTIVITY[event.event_type]
        const to = link(event)
        const body = (
          <>
            <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-md bg-surface-muted text-fg-muted">
              <Icon className="size-3.5" aria-hidden />
            </span>
            <span className="min-w-0">
              <span className="block text-sm text-fg">{event.description}</span>
              <span className="block text-xs text-fg-subtle">
                {formatRelativeTime(event.created_at)}
              </span>
            </span>
          </>
        )
        return (
          <li key={event.id}>
            {to ? (
              <Link to={to} className="flex gap-3 rounded-lg px-2 py-2 hover:bg-surface-muted/60">
                {body}
              </Link>
            ) : (
              <div className="flex gap-3 px-2 py-2">{body}</div>
            )}
          </li>
        )
      })}
    </ol>
  )
}

export default function Dashboard() {
  const { data: stats, error, loading, reload } = useApi(api.dashboardStats, {
    pollMs: BUSY_POLL_MS,
    pollWhile: (current) => current.recent_documents.some(isDocumentBusy),
  })

  return (
    <div className="space-y-6">
      <section className="flex flex-col gap-4 rounded-xl border border-border bg-surface px-6 py-6 shadow-card sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-fg">
            Welcome to your research workspace
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-fg-muted">
            Upload papers, ask questions across them, and get answers you can verify. Every
            answer links back to the paper and page it came from.
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          <ButtonLink to="/assistant" variant="secondary">
            Ask the assistant
          </ButtonLink>
          <ButtonLink to="/papers">
            <FilePlus2 className="size-4" aria-hidden />
            Upload papers
          </ButtonLink>
        </div>
      </section>

      {error && (
        <ErrorState title="Couldn't load dashboard" message={error.message} onRetry={reload} />
      )}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Workspace totals">
        <MetricCard
          label="Research papers"
          loading={loading}
          value={stats?.total_papers ?? null}
          icon={FileText}
          hint={stats ? `${formatNumber(stats.total_chunks)} searchable passages` : undefined}
        />
        <MetricCard
          label="Indexed papers"
          loading={loading}
          value={stats?.indexed_papers ?? null}
          icon={Database}
          hint="Ready for semantic and keyword search"
        />
        <MetricCard
          label="Questions asked"
          loading={loading}
          value={stats?.total_questions ?? null}
          icon={MessageSquareText}
        />
        <MetricCard
          label="Conversations"
          loading={loading}
          value={stats?.total_conversations ?? null}
          icon={Layers}
        />
      </section>

      <section className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader
            title="Recently uploaded papers"
            description="The latest additions to your library"
            action={
              stats && stats.recent_documents.length > 0 ? (
                <Link
                  to="/papers"
                  className="inline-flex items-center gap-1 text-xs font-medium text-accent-text hover:underline"
                >
                  View all <ArrowRight className="size-3" aria-hidden />
                </Link>
              ) : undefined
            }
          />
          {loading && !stats ? <ListSkeleton /> : stats && <RecentDocuments documents={stats.recent_documents} />}
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader title="Recent activity" description="What happened in your workspace" />
          {loading && !stats ? <ListSkeleton /> : stats && <RecentActivity events={stats.recent_activity} />}
        </Card>
      </section>

      <section aria-labelledby="quick-actions">
        <h2 id="quick-actions" className="mb-3 text-sm font-semibold text-fg">
          Quick actions
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {QUICK_ACTIONS.map(({ to, title, description, icon: Icon }) => (
            <Link
              key={title}
              to={to}
              className="group rounded-xl border border-border bg-surface p-5 shadow-card transition-colors hover:border-border-strong"
            >
              <div className="flex size-9 items-center justify-center rounded-lg bg-accent-soft text-accent-text">
                <Icon className="size-4" aria-hidden />
              </div>
              <p className="mt-4 flex items-center gap-1 text-sm font-semibold text-fg">
                {title}
                <ArrowRight
                  className="size-3.5 text-fg-subtle transition-transform group-hover:translate-x-0.5"
                  aria-hidden
                />
              </p>
              <p className="mt-1 text-xs text-fg-muted">{description}</p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
