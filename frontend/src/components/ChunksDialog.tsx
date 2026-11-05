import { X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api } from '../services/api'
import type { Chunk, DocumentSummary } from '../types/api'
import { formatPageRange, pluralize } from '../utils/format'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'
import { ErrorState, Spinner } from './ui/Feedback'

const PAGE_SIZE = 20

interface ChunksDialogProps {
  document: DocumentSummary | null
  onClose: () => void
}

/** Shows how a paper was split into passages: the units retrieval and citations use. */
export function ChunksDialog({ document, onClose }: ChunksDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (documentId: string, offset: number) => {
    setLoading(true)
    setError(null)
    try {
      const page = await api.listChunks(documentId, offset, PAGE_SIZE)
      setChunks((current) => (offset === 0 ? page.items : [...current, ...page.items]))
      setTotal(page.total)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not load passages.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (document && !dialog.open) {
      dialog.showModal()
      void load(document.id, 0)
    }
    if (!document && dialog.open) dialog.close()
  }, [document, load])

  return (
    <dialog
      ref={dialogRef}
      onCancel={(event) => {
        event.preventDefault()
        onClose()
      }}
      onClose={() => setChunks([])}
      // `open:flex`, not `flex`: a display class would override the browser
      // rule that hides a closed <dialog>.
      className="m-auto max-h-[85vh] w-[calc(100%-2rem)] max-w-3xl flex-col rounded-xl border border-border bg-surface p-0 text-fg shadow-xl backdrop:bg-black/50 open:flex"
    >
      <header className="flex items-start justify-between gap-4 border-b border-border px-6 py-4">
        <div className="min-w-0">
          <h2 className="truncate text-base font-semibold">{document?.title ?? document?.filename}</h2>
          <p className="mt-0.5 text-xs text-fg-muted">
            {pluralize(total, 'passage')}. Each one is embedded and searched on its own, and
            answers cite its pages.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md p-1 text-fg-muted hover:bg-surface-muted"
          aria-label="Close"
        >
          <X className="size-5" />
        </button>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto px-6 py-5">
        {error && <ErrorState message={error} onRetry={() => document && load(document.id, 0)} />}
        {chunks.map((chunk) => (
          <article key={chunk.id} className="rounded-lg border border-border">
            <div className="flex flex-wrap items-center gap-2 border-b border-border bg-surface-muted/50 px-4 py-2 text-xs">
              <span className="font-semibold text-fg">#{chunk.chunk_index + 1}</span>
              <Badge tone="accent">{formatPageRange(chunk.page_start, chunk.page_end)}</Badge>
              {chunk.section && <span className="truncate text-fg-muted">{chunk.section}</span>}
              <span className="ml-auto text-fg-subtle tabular-nums">{chunk.token_count} tokens</span>
            </div>
            <p className="px-4 py-3 text-sm leading-relaxed whitespace-pre-line text-fg-muted">
              {chunk.text}
            </p>
          </article>
        ))}
        {loading && (
          <div className="flex justify-center py-4">
            <Spinner className="size-5" />
          </div>
        )}
        {!loading && document && chunks.length < total && (
          <div className="flex justify-center">
            <Button variant="secondary" size="sm" onClick={() => load(document.id, chunks.length)}>
              Load more ({chunks.length} of {total})
            </Button>
          </div>
        )}
      </div>
    </dialog>
  )
}
