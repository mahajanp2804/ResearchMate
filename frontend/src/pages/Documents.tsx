import { DatabaseZap, FileText, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { ChunksDialog } from '../components/ChunksDialog'
import { DocumentCard } from '../components/DocumentCard'
import { UploadQueue } from '../components/UploadQueue'
import { UploadZone } from '../components/UploadZone'
import { Button } from '../components/ui/Button'
import { Card, CardHeader } from '../components/ui/Card'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { EmptyState } from '../components/ui/EmptyState'
import { ErrorState, Skeleton, Spinner } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { isActiveUpload, useUploadQueue } from '../hooks/useUploadQueue'
import { ApiError, api } from '../services/api'
import type { DocumentSummary } from '../types/api'
import { BUSY_POLL_MS, isDocumentBusy, needsIndexing } from '../utils/documents'
import { pluralize } from '../utils/format'

// Used until /api/system/info has loaded; the server enforces the real limits.
const DEFAULT_MAX_SIZE_MB = 50
const DEFAULT_MAX_PAGES = 500

function matchesQuery(document: DocumentSummary, query: string): boolean {
  const haystack = [document.title, document.filename, document.extracted_metadata.author]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
  return haystack.includes(query)
}

export default function Documents() {
  const { data: info } = useApi(api.systemInfo)
  // Refresh while any paper is still being processed or indexed in the background.
  const documents = useApi(api.listDocuments, {
    pollMs: BUSY_POLL_MS,
    pollWhile: (list) => list.items.some(isDocumentBusy),
  })
  const maxSizeMb = info?.max_upload_size_mb ?? DEFAULT_MAX_SIZE_MB
  const uploads = useUploadQueue({ maxSizeMb, onUploaded: documents.reload })

  const [query, setQuery] = useState('')
  const [pendingDelete, setPendingDelete] = useState<DocumentSummary | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [reindexingId, setReindexingId] = useState<string | null>(null)
  const [indexingAll, setIndexingAll] = useState(false)
  const [viewing, setViewing] = useState<DocumentSummary | null>(null)
  const [actionError, setActionError] = useState<{ title: string; message: string } | null>(null)

  const items = documents.data?.items
  const visible = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return normalized ? (items ?? []).filter((doc) => matchesQuery(doc, normalized)) : (items ?? [])
  }, [items, query])

  const confirmDelete = async () => {
    if (!pendingDelete) return
    setDeleting(true)
    setActionError(null)
    try {
      await api.deleteDocument(pendingDelete.id)
      documents.reload()
    } catch (error) {
      setActionError({
        title: 'Delete failed',
        message: error instanceof ApiError ? error.message : 'The paper could not be deleted.',
      })
    } finally {
      setDeleting(false)
      setPendingDelete(null)
    }
  }

  const reindex = async (document: DocumentSummary) => {
    setReindexingId(document.id)
    setActionError(null)
    try {
      // A processing failure comes back in the document's status, not as an error.
      await api.reindexDocument(document.id)
    } catch (error) {
      setActionError({
        title: 'Re-indexing failed',
        message: error instanceof ApiError ? error.message : 'The paper could not be re-processed.',
      })
    } finally {
      setReindexingId(null)
      documents.reload()
    }
  }

  // Papers whose text is ready but which are not searchable, e.g. after the
  // embedding model was changed or an indexing run failed.
  const unindexed = (items ?? []).filter(needsIndexing)
  const indexAll = async () => {
    setIndexingAll(true)
    setActionError(null)
    try {
      for (const document of unindexed) await api.reindexDocument(document.id)
    } catch (error) {
      setActionError({
        title: 'Indexing failed',
        message: error instanceof ApiError ? error.message : 'Some papers could not be indexed.',
      })
    } finally {
      setIndexingAll(false)
      documents.reload()
    }
  }

  const finishedUploads = uploads.items.filter((item) => !isActiveUpload(item)).length

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Add research papers"
          description="Text is extracted page by page so every answer can cite its page."
          action={
            finishedUploads > 0 && (
              <Button variant="ghost" size="sm" onClick={uploads.clearFinished}>
                Clear finished
              </Button>
            )
          }
        />
        <div className="p-5">
          <UploadZone
            maxSizeMb={maxSizeMb}
            maxPages={info?.max_pages_per_document ?? DEFAULT_MAX_PAGES}
            onFiles={uploads.addFiles}
          />
        </div>
        {uploads.items.length > 0 && (
          <div className="border-t border-border">
            <UploadQueue items={uploads.items} />
          </div>
        )}
      </Card>

      <section aria-labelledby="library-heading" className="space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h2 id="library-heading" className="text-sm font-semibold text-fg">
            Library
            {documents.data && (
              <span className="ml-2 font-normal text-fg-subtle">
                {pluralize(documents.data.total, 'paper')}
              </span>
            )}
          </h2>
          {items && items.length > 0 && (
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              {unindexed.length >= 2 && (
                <Button variant="secondary" onClick={indexAll} disabled={indexingAll}>
                  {indexingAll ? <Spinner /> : <DatabaseZap className="size-4" aria-hidden />}
                  Index all ({unindexed.length})
                </Button>
              )}
              <label className="relative block sm:w-72">
                <span className="sr-only">Search papers</span>
                <Search
                  className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-fg-subtle"
                  aria-hidden
                />
                <input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search by title, file or author"
                  className="h-9 w-full rounded-lg border border-border bg-surface pr-3 pl-9 text-sm text-fg shadow-card placeholder:text-fg-subtle focus:border-accent focus:outline-none"
                />
              </label>
            </div>
          )}
        </div>

        {documents.error && (
          <ErrorState
            title="Couldn't load your papers"
            message={documents.error.message}
            onRetry={documents.reload}
          />
        )}
        {actionError && <ErrorState title={actionError.title} message={actionError.message} />}

        {documents.loading && !documents.data ? (
          <div className="space-y-3">
            {[0, 1, 2].map((row) => (
              <Skeleton key={row} className="h-32 w-full rounded-xl" />
            ))}
          </div>
        ) : items && items.length === 0 ? (
          <Card>
            <EmptyState
              icon={FileText}
              title="Your library is empty"
              description="Upload your first research paper above to start building your library."
            />
          </Card>
        ) : items && visible.length === 0 ? (
          <Card>
            <EmptyState
              icon={Search}
              title="No matching papers"
              description={`Nothing in your library matches “${query.trim()}”.`}
            />
          </Card>
        ) : (
          <div className="space-y-3">
            {visible.map((document) => (
              <DocumentCard
                key={document.id}
                document={document}
                reindexing={reindexingId === document.id}
                onDelete={setPendingDelete}
                onReindex={reindex}
                onViewPassages={setViewing}
              />
            ))}
          </div>
        )}
      </section>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete this paper?"
        description={`“${pendingDelete?.title ?? pendingDelete?.filename ?? ''}” and its extracted text will be removed from your library. This can't be undone.`}
        confirmLabel="Delete paper"
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
      <ChunksDialog document={viewing} onClose={() => setViewing(null)} />
    </div>
  )
}
