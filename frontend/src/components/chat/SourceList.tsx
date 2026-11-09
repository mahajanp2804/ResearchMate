import { ChevronDown, ExternalLink } from 'lucide-react'
import { api } from '../../services/api'
import type { ChatSource } from '../../types/api'
import { sourceAnchorId } from '../../utils/answerFormat'
import { cn, formatPageRange } from '../../utils/format'

interface SourceListProps {
  messageId: number
  sources: ChatSource[]
  open: boolean
  highlighted: string | null
  onToggle: () => void
  label?: string
}

/** The numbered passages an answer was given, with page links into the PDFs. */
export function SourceList({ messageId, sources, open, highlighted, onToggle, label }: SourceListProps) {
  return (
    <div className="rounded-lg border border-border">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs font-medium text-fg-muted hover:text-fg"
      >
        {label ?? `Sources (${sources.length})`}
        <ChevronDown className={cn('size-3.5 transition-transform', open && 'rotate-180')} aria-hidden />
      </button>
      {open && (
        <ol className="divide-y divide-border border-t border-border">
          {sources.map((source) => (
            <li
              key={source.source_id}
              id={sourceAnchorId(messageId, source.source_id)}
              className={cn(
                'flex gap-3 px-3 py-2.5 transition-colors',
                highlighted === source.source_id && 'bg-accent-soft',
              )}
            >
              <span className="mt-0.5 h-fit shrink-0 rounded-md bg-accent-soft px-1.5 py-0.5 font-mono text-xs font-semibold text-accent-text">
                {source.source_id}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-fg">{source.document_title}</p>
                <p className="text-xs text-fg-subtle">
                  {formatPageRange(source.page_start, source.page_end)}
                  {source.section && ` · ${source.section}`}
                </p>
                <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-fg-muted">{source.text}</p>
                <a
                  href={api.documentFileUrl(source.document_id, source.page_start)}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 inline-flex items-center gap-1 text-xs text-accent-text hover:underline"
                >
                  <ExternalLink className="size-3" aria-hidden /> Open at page {source.page_start}
                </a>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
