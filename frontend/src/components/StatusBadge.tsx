import type { IndexingStatus, ProcessingStatus } from '../types/api'
import { Badge, type BadgeTone } from './ui/Badge'
import { Spinner } from './ui/Feedback'

const STATUS: Record<
  ProcessingStatus | IndexingStatus,
  { label: string; tone: BadgeTone; busy?: boolean }
> = {
  pending: { label: 'Queued', tone: 'neutral' },
  processing: { label: 'Processing', tone: 'accent', busy: true },
  completed: { label: 'Processed', tone: 'success' },
  not_indexed: { label: 'Not indexed', tone: 'neutral' },
  indexing: { label: 'Indexing', tone: 'accent', busy: true },
  indexed: { label: 'Searchable', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
}

interface DocumentStatusBadgeProps {
  processingStatus: ProcessingStatus
  indexingStatus: IndexingStatus
}

/** One badge summarising where a document is in the ingestion pipeline. */
export function DocumentStatusBadge({ processingStatus, indexingStatus }: DocumentStatusBadgeProps) {
  const status = processingStatus === 'completed' ? indexingStatus : processingStatus
  const { label, tone, busy } = STATUS[status]
  return (
    <Badge tone={tone}>
      {busy && <Spinner className="size-3 text-current" />}
      {label}
    </Badge>
  )
}
