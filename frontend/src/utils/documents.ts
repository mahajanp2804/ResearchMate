import type { IndexingStatus, ProcessingStatus } from '../types/api'

interface HasStatus {
  processing_status: ProcessingStatus
  indexing_status: IndexingStatus
}

/** The backend is still working on this paper; its status will change soon. */
export function isDocumentBusy(document: HasStatus): boolean {
  return document.processing_status === 'processing' || document.indexing_status === 'indexing'
}

/** Text is processed but the paper is not (or no longer) searchable. */
export function needsIndexing(document: HasStatus): boolean {
  return (
    document.processing_status === 'completed' &&
    (document.indexing_status === 'not_indexed' || document.indexing_status === 'failed')
  )
}

// How often lists refresh while a paper is being processed or indexed.
export const BUSY_POLL_MS = 2000
