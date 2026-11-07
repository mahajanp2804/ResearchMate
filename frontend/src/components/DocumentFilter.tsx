import { Check, ChevronDown, Files } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import type { DocumentSummary } from '../types/api'
import { cn, pluralize } from '../utils/format'

interface DocumentFilterProps {
  /** Papers that can be searched (indexed). */
  documents: DocumentSummary[]
  /** Selected paper ids; `null` means all papers. */
  selected: string[] | null
  onChange: (selected: string[] | null) => void
}

/** Choose which papers a search or conversation covers: all of them, or a selection. */
export function DocumentFilter({ documents, selected, onChange }: DocumentFilterProps) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const panelId = useId()

  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent | KeyboardEvent) => {
      const escape = event instanceof KeyboardEvent && event.key === 'Escape'
      const clickedOutside =
        event instanceof MouseEvent && !containerRef.current?.contains(event.target as Node)
      if (escape || clickedOutside) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', close)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', close)
    }
  }, [open])

  const toggle = (id: string) => {
    const current = selected ?? []
    const next = current.includes(id) ? current.filter((item) => item !== id) : [...current, id]
    // Deselecting the last paper means "all papers" rather than "nothing".
    onChange(next.length === 0 ? null : next)
  }

  const label =
    selected === null
      ? `All papers (${documents.length})`
      : selected.length === 1
        ? (documents.find((doc) => doc.id === selected[0])?.title ?? '1 paper')
        : `${pluralize(selected.length, 'paper')} selected`

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-controls={panelId}
        className="inline-flex h-9 w-full max-w-full items-center gap-2 rounded-lg border border-border bg-surface px-3 text-sm text-fg shadow-card hover:bg-surface-muted sm:w-64"
      >
        <Files className="size-4 shrink-0 text-fg-subtle" aria-hidden />
        <span className="min-w-0 flex-1 truncate text-left">{label}</span>
        <ChevronDown className="size-4 shrink-0 text-fg-subtle" aria-hidden />
      </button>

      {open && (
        <div
          id={panelId}
          role="group"
          aria-label="Papers to search"
          className="absolute left-0 z-30 mt-1 w-80 max-w-[calc(100vw-2rem)] overflow-hidden rounded-lg border border-border bg-surface shadow-xl"
        >
          <button
            type="button"
            onClick={() => onChange(null)}
            className="flex w-full items-center gap-2 border-b border-border px-3 py-2.5 text-left text-sm hover:bg-surface-muted"
          >
            <Check
              className={cn('size-4 text-accent-text', selected !== null && 'invisible')}
              aria-hidden
            />
            All papers
          </button>
          <ul className="max-h-72 overflow-y-auto py-1">
            {documents.map((document) => {
              const checked = selected?.includes(document.id) ?? false
              return (
                <li key={document.id}>
                  <label className="flex cursor-pointer items-start gap-2 px-3 py-2 text-sm hover:bg-surface-muted">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggle(document.id)}
                      className="mt-0.5 size-4 accent-[var(--accent)]"
                    />
                    <span className="min-w-0">
                      <span className="line-clamp-2 text-fg">{document.title ?? document.filename}</span>
                      {document.extracted_metadata.author && (
                        <span className="block truncate text-xs text-fg-subtle">
                          {document.extracted_metadata.author}
                        </span>
                      )}
                    </span>
                  </label>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}
