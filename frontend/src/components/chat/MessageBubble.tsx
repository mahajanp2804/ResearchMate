import { Bot, SearchX, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import type { AnswerDetails, ChatMessage } from '../../types/api'
import { sourceAnchorId } from '../../utils/answerFormat'
import { pluralize } from '../../utils/format'
import { CitationWarnings } from '../CitationWarnings'
import { AnswerText } from './AnswerText'
import { SourceList } from './SourceList'

function formatSeconds(milliseconds: number | undefined): string | null {
  return milliseconds === undefined ? null : `${(milliseconds / 1000).toFixed(1)} s`
}

/** How the answer was produced: what was searched, how a follow-up was read, timing. */
function AnswerMeta({ details, question }: { details: AnswerDetails; question?: string }) {
  const parts = [
    `Searched ${pluralize(details.searched_document_count, 'paper')}`,
    formatSeconds(details.timings_ms.total),
    details.model,
  ].filter(Boolean)
  return (
    <div className="space-y-1 text-xs text-fg-subtle">
      {question && details.standalone_question !== question && (
        <p>
          Understood as: <span className="text-fg-muted">“{details.standalone_question}”</span>
        </p>
      )}
      <p>{parts.join(' · ')}</p>
    </div>
  )
}

export function MessageBubble({ message, question }: { message: ChatMessage; question?: string }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [highlighted, setHighlighted] = useState<string | null>(null)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-sm whitespace-pre-wrap text-accent-fg">
          {message.content}
        </p>
      </div>
    )
  }

  const details = message.details
  const outcome = details?.outcome ?? 'answered'
  const sourceIds = message.sources.map((source) => source.source_id)

  const cite = (sourceId: string) => {
    setSourcesOpen(true)
    setHighlighted(sourceId)
    // Wait for the list to render before scrolling to the passage.
    requestAnimationFrame(() =>
      document
        .getElementById(sourceAnchorId(message.id, sourceId))
        ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }),
    )
  }

  return (
    <div className="flex gap-3">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent-text">
        <Bot className="size-4" aria-hidden />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {outcome === 'answered' ? (
          <AnswerText text={message.content} sourceIds={sourceIds} onCite={cite} />
        ) : (
          <div className="rounded-lg border border-warning/30 bg-warning-soft px-4 py-3">
            <p className="flex gap-2 text-sm text-fg">
              <SearchX className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
              {message.content}
            </p>
            <p className="mt-1 pl-6 text-xs text-fg-muted">
              {outcome === 'no_evidence'
                ? 'No passage in the searched papers was relevant, so no answer was generated.'
                : `The model read ${pluralize(message.sources.length, 'passage')} and judged them insufficient to answer.`}
            </p>
          </div>
        )}
        {outcome === 'answered' && details && (
          <CitationWarnings
            status={details.citation_status}
            uncitedStatements={details.uncited_statements}
            removedIds={details.invalid_citation_ids}
            noun="answer"
            evidence="your papers"
          />
        )}
        {details?.truncated && (
          <p className="flex gap-1.5 text-xs text-warning">
            <TriangleAlert className="size-3.5" aria-hidden /> The answer hit the model's output
            limit and may be cut off.
          </p>
        )}
        {message.sources.length > 0 && (
          <SourceList
            messageId={message.id}
            sources={message.sources}
            open={sourcesOpen}
            highlighted={highlighted}
            onToggle={() => setSourcesOpen((value) => !value)}
            label={
              outcome === 'answered'
                ? undefined
                : `Passages considered (${message.sources.length})`
            }
          />
        )}
        {details && <AnswerMeta details={details} question={question} />}
      </div>
    </div>
  )
}
