import type { IndexingStatus, ProcessingStatus } from '../types/api'
import { Badge, type BadgeTone } from './ui/Badge'

const STATUS: Record<ProcessingStatus | IndexingStatus, { label: string; tone: BadgeTone }> = {
  pending: { label: 'Queued', tone: 'neutral' },
  processing: { label: 'Processing', tone: 'accent' },
  completed: { label: 'Processed', tone: 'success' },
  not_indexed: { label: 'Not indexed', tone: 'neutral' },
  indexing: { label: 'Indexing', tone: 'accent' },
  indexed: { label: 'Indexed', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
}

interface DocumentStatusBadgeProps {
  processingStatus: ProcessingStatus
  indexingStatus: IndexingStatus
}

/** One badge summarising where a document is in the ingestion pipeline. */
export function DocumentStatusBadge({ processingStatus, indexingStatus }: DocumentStatusBadgeProps) {
  const status = processingStatus === 'completed' ? indexingStatus : processingStatus
  const { label, tone } = STATUS[status]
  return <Badge tone={tone}>{label}</Badge>
}
