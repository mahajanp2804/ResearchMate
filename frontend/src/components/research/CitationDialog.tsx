import { ExternalLink, X } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { api } from '../../services/api'
import type { ProfileCitation, ProfileSource } from '../../types/api'
import { formatPageRange } from '../../utils/format'

/** A passage cited by a summary section or comparison cell. */
export interface CitedPassage {
  documentId: string
  documentTitle: string
  /** The section or row that cites it, e.g. "Datasets". */
  aspectLabel: string
  source: ProfileSource
  citation: ProfileCitation
}

interface CitationDialogProps {
  passage: CitedPassage | null
  onClose: () => void
}

/** Shows the sentence a statement relies on, inside its full passage, with a link to the page. */
export function CitationDialog({ passage, onClose }: CitationDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (passage && !dialog.open) dialog.showModal()
    if (!passage && dialog.open) dialog.close()
  }, [passage])

  const source = passage?.source
  const citation = passage?.citation

  return (
    <dialog
      ref={dialogRef}
      onCancel={(event) => {
        event.preventDefault()
        onClose()
      }}
      onClick={(event) => {
        // A click on the backdrop lands on the <dialog> element itself.
        if (event.target === event.currentTarget) onClose()
      }}
      // `open:flex`, not `flex`: a display class would override the browser
      // rule that hides a closed <dialog>.
      className="m-auto max-h-[85vh] w-[calc(100%-2rem)] max-w-2xl flex-col rounded-xl border border-border bg-surface p-0 text-fg shadow-xl backdrop:bg-black/50 open:flex"
    >
      {passage && source && citation && (
        <>
          <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
            <div className="min-w-0">
              <p className="text-xs font-medium text-accent-text">{passage.aspectLabel}</p>
              <h2 className="mt-0.5 text-sm font-semibold">{passage.documentTitle}</h2>
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
          <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
            <section>
              <h3 className="text-xs font-medium text-fg-muted">
                Supporting sentence · p. {citation.page}
              </h3>
              <blockquote className="mt-1.5 border-l-2 border-accent pl-3 text-sm leading-relaxed">
                “{citation.excerpt}”
              </blockquote>
            </section>
            <section>
              <h3 className="text-xs font-medium text-fg-muted">
                Full passage · {formatPageRange(source.page_start, source.page_end)}
                {source.section && ` · ${source.section}`}
              </h3>
              <p className="mt-1.5 rounded-lg bg-surface-muted/60 px-3 py-2 text-sm leading-relaxed whitespace-pre-line text-fg-muted">
                {source.text}
              </p>
            </section>
          </div>
          <footer className="border-t border-border px-5 py-3">
            <a
              href={api.documentFileUrl(passage.documentId, citation.page)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-accent-text hover:underline"
            >
              <ExternalLink className="size-4" aria-hidden />
              Open the PDF at page {citation.page}
            </a>
          </footer>
        </>
      )}
    </dialog>
  )
}
