import {
  ChevronDown,
  CircleAlert,
  DatabaseZap,
  ExternalLink,
  FileText,
  Layers,
  RefreshCw,
  Trash2,
  TriangleAlert,
} from 'lucide-react'
import { useState } from 'react'
import { api } from '../services/api'
import type { DocumentSummary } from '../types/api'
import { needsIndexing } from '../utils/documents'
import { cn, formatBytes, formatDate, pluralize } from '../utils/format'
import { DocumentStatusBadge } from './StatusBadge'
import { Button } from './ui/Button'
import { Card } from './ui/Card'
import { Spinner } from './ui/Feedback'

const TITLE_SOURCE_LABEL = {
  metadata: 'PDF metadata',
  first_page: 'first page heading',
  filename: 'file name',
} as const

interface DocumentCardProps {
  document: DocumentSummary
  reindexing: boolean
  onDelete: (document: DocumentSummary) => void
  onReindex: (document: DocumentSummary) => void
  onViewPassages: (document: DocumentSummary) => void
}

export function DocumentCard({
  document,
  reindexing,
  onDelete,
  onReindex,
  onViewPassages,
}: DocumentCardProps) {
  const [expanded, setExpanded] = useState(false)
  const metadata = document.extracted_metadata
  const title = document.title ?? document.filename
  const processingFailed = document.processing_status === 'failed'
  const failed = processingFailed || document.indexing_status === 'failed'
  const canIndex = needsIndexing(document)

  const details: [string, string | undefined][] = [
    ['File name', document.filename],
    ['Authors', metadata.author],
    ['Subject', metadata.subject],
    ['Keywords', metadata.keywords],
    ['Created', metadata.created],
    ['Producer', metadata.producer],
    ['Title detected from', metadata.title_source && TITLE_SOURCE_LABEL[metadata.title_source]],
  ]

  return (
    <Card className="overflow-hidden">
      <div className="flex gap-4 p-5">
        <div className="hidden size-10 shrink-0 items-center justify-center rounded-lg bg-surface-muted sm:flex">
          <FileText className="size-5 text-fg-muted" aria-hidden />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
            <h3 className="min-w-0 text-sm leading-snug font-semibold text-fg">{title}</h3>
            <DocumentStatusBadge
              processingStatus={document.processing_status}
              indexingStatus={document.indexing_status}
            />
          </div>
          {metadata.author && (
            <p className="mt-1 truncate text-xs text-fg-muted">{metadata.author}</p>
          )}
          <p className="mt-1 text-xs text-fg-subtle">
            {document.page_count !== null && `${pluralize(document.page_count, 'page')} · `}
            {formatBytes(document.file_size_bytes)} · Uploaded {formatDate(document.upload_date)}
          </p>
          {document.status_message && (
            <p
              className={cn(
                'mt-3 flex gap-2 rounded-lg px-3 py-2 text-xs text-fg-muted',
                failed ? 'bg-danger-soft' : 'bg-warning-soft',
              )}
            >
              {failed ? (
                <CircleAlert className="mt-px size-3.5 shrink-0 text-danger" aria-hidden />
              ) : (
                <TriangleAlert className="mt-px size-3.5 shrink-0 text-warning" aria-hidden />
              )}
              {document.status_message}
            </p>
          )}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            {(processingFailed || canIndex) && (
              <Button size="sm" onClick={() => onReindex(document)} disabled={reindexing}>
                {reindexing ? (
                  <Spinner className="text-accent-fg" />
                ) : processingFailed ? (
                  <RefreshCw className="size-3.5" aria-hidden />
                ) : (
                  <DatabaseZap className="size-3.5" aria-hidden />
                )}
                {processingFailed ? 'Retry processing' : 'Index now'}
              </Button>
            )}
            {document.chunk_count > 0 && (
              <Button variant="secondary" size="sm" onClick={() => onViewPassages(document)}>
                <Layers className="size-3.5" aria-hidden />
                {pluralize(document.chunk_count, 'passage')}
              </Button>
            )}
            <a
              href={api.documentFileUrl(document.id)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border bg-surface px-3 text-xs font-medium text-fg shadow-card hover:bg-surface-muted"
            >
              <ExternalLink className="size-3.5" aria-hidden />
              Open PDF
            </a>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded((value) => !value)}
              aria-expanded={expanded}
            >
              <ChevronDown
                className={cn('size-3.5 transition-transform', expanded && 'rotate-180')}
                aria-hidden
              />
              Details
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDelete(document)}
              className="ml-auto text-danger hover:bg-danger-soft hover:text-danger"
            >
              <Trash2 className="size-3.5" aria-hidden />
              Delete
            </Button>
          </div>
        </div>
      </div>
      {expanded && (
        <dl className="grid gap-x-6 gap-y-3 border-t border-border bg-surface-muted/40 px-5 py-4 sm:grid-cols-2">
          {details
            .filter((entry): entry is [string, string] => Boolean(entry[1]))
            .map(([label, value]) => (
              <div key={label} className="min-w-0">
                <dt className="text-xs text-fg-subtle">{label}</dt>
                <dd className="mt-0.5 text-sm break-words text-fg">{value}</dd>
              </div>
            ))}
        </dl>
      )}
    </Card>
  )
}
