import { ChevronDown, ExternalLink } from 'lucide-react'
import { useState } from 'react'
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

function SourceCard({
  messageId,
  source,
  highlighted,
}: {
  messageId: number
  source: ChatSource
  highlighted: boolean
}) {
  const [showPassage, setShowPassage] = useState(false)
  // Cited sources link to the exact page of their supporting excerpt.
  const page = source.excerpt_page ?? source.page_start
  return (
    <li
      id={sourceAnchorId(messageId, source.source_id)}
      className={cn('flex gap-3 px-3 py-2.5 transition-colors', highlighted && 'bg-accent-soft')}
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
        {source.excerpt && (
          <blockquote className="mt-1.5 border-l-2 border-accent pl-2.5 text-xs leading-relaxed text-fg">
            “{source.excerpt}”
            {source.page_start !== source.page_end && (
              <span className="ml-1 text-fg-subtle">(p. {page})</span>
            )}
          </blockquote>
        )}
        {(!source.excerpt || showPassage) && (
          <p
            className={cn(
              'mt-1.5 text-xs leading-relaxed text-fg-muted',
              !showPassage && 'line-clamp-3',
            )}
          >
            {source.text}
          </p>
        )}
        <div className="mt-1 flex flex-wrap items-center gap-3 text-xs">
          <a
            href={api.documentFileUrl(source.document_id, page)}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-accent-text hover:underline"
          >
            <ExternalLink className="size-3" aria-hidden /> Open at page {page}
          </a>
          {source.excerpt && (
            <button
              type="button"
              onClick={() => setShowPassage((value) => !value)}
              className="text-fg-muted hover:text-fg"
            >
              {showPassage ? 'Hide passage' : 'Show full passage'}
            </button>
          )}
        </div>
      </div>
    </li>
  )
}

/**
 * The numbered passages an answer was given. Cited sources come first with
 * the excerpt that supports the answer; the rest are grouped as retrieved
 * but not cited.
 */
export function SourceList({ messageId, sources, open, highlighted, onToggle, label }: SourceListProps) {
  const [showUncited, setShowUncited] = useState(false)
  const cited = sources.filter((source) => source.cited)
  const uncited = sources.filter((source) => !source.cited)
  // Refusals (and answers saved before citations were checked) cite nothing:
  // list every passage as before.
  const primary = cited.length > 0 ? cited : uncited
  const secondary = cited.length > 0 ? uncited : []
  const summary =
    label ??
    (cited.length > 0
      ? `Sources · ${cited.length} cited · ${sources.length} retrieved`
      : `Sources (${sources.length})`)

  return (
    <div className="rounded-lg border border-border">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs font-medium text-fg-muted hover:text-fg"
      >
        {summary}
        <ChevronDown className={cn('size-3.5 transition-transform', open && 'rotate-180')} aria-hidden />
      </button>
      {open && (
        <div className="border-t border-border">
          <ol className="divide-y divide-border">
            {primary.map((source) => (
              <SourceCard
                key={source.source_id}
                messageId={messageId}
                source={source}
                highlighted={highlighted === source.source_id}
              />
            ))}
          </ol>
          {secondary.length > 0 && (
            <div className="border-t border-border">
              <button
                type="button"
                onClick={() => setShowUncited((value) => !value)}
                aria-expanded={showUncited}
                className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs text-fg-subtle hover:text-fg"
              >
                Also retrieved, not cited ({secondary.length})
                <ChevronDown
                  className={cn('size-3.5 transition-transform', showUncited && 'rotate-180')}
                  aria-hidden
                />
              </button>
              {showUncited && (
                <ol className="divide-y divide-border border-t border-border opacity-80">
                  {secondary.map((source) => (
                    <SourceCard
                      key={source.source_id}
                      messageId={messageId}
                      source={source}
                      highlighted={highlighted === source.source_id}
                    />
                  ))}
                </ol>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
