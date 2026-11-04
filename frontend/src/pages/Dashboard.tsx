import {
  Activity,
  ArrowRight,
  Database,
  FilePlus2,
  FileText,
  GitCompareArrows,
  Layers,
  Lightbulb,
  MessageSquareText,
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
import type { ActivityItem, RecentDocument } from '../types/api'
import { formatNumber, formatRelativeTime, pluralize } from '../utils/format'

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
    to: '/comparison',
    title: 'Compare papers',
    description: 'Line up objectives, methods, datasets and results.',
    icon: GitCompareArrows,
  },
  {
    to: '/insights',
    title: 'Find research gaps',
    description: 'Surface stated limitations and open directions.',
    icon: Lightbulb,
  },
]

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
            <p className="truncate text-sm font-medium text-fg">
              {document.title ?? document.filename}
            </p>
            <p className="truncate text-xs text-fg-subtle">
              {document.page_count !== null && `${pluralize(document.page_count, 'page')} · `}
              {formatRelativeTime(document.created_at)}
            </p>
          </div>
          <DocumentStatusBadge
            processingStatus={document.processing_status}
            indexingStatus={document.indexing_status}
          />
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
    <ol className="space-y-4 px-5 py-4">
      {events.map((event) => (
        <li key={event.id} className="flex gap-3">
          <span className="mt-1.5 size-2 shrink-0 rounded-full bg-accent" aria-hidden />
          <div className="min-w-0">
            <p className="text-sm text-fg">{event.description}</p>
            <p className="text-xs text-fg-subtle">{formatRelativeTime(event.created_at)}</p>
          </div>
        </li>
      ))}
    </ol>
  )
}

export default function Dashboard() {
  const { data: stats, error, loading, reload } = useApi(api.dashboardStats)

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
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {QUICK_ACTIONS.map(({ to, title, description, icon: Icon }) => (
            <Link
              key={to}
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
