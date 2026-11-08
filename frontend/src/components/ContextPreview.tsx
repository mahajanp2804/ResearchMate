import { Bot, TriangleAlert } from 'lucide-react'
import type { ContextResponse } from '../types/api'
import { formatNumber, formatPageRange, pluralize } from '../utils/format'
import { Badge } from './ui/Badge'

/** The passages the answer pipeline would send to the LLM, as numbered sources. */
export function ContextPreview({ context }: { context: ContextResponse }) {
  const usedShare = Math.min(100, (context.total_tokens / context.max_tokens) * 100)
  const dropped = [
    context.dropped.duplicates > 0 && pluralize(context.dropped.duplicates, 'near-duplicate'),
    context.dropped.over_budget > 0 && `${context.dropped.over_budget} over the token budget`,
    context.dropped.low_score > 0 && `${context.dropped.low_score} below the evidence threshold`,
  ].filter(Boolean)

  return (
    <section aria-labelledby="context-heading" className="rounded-lg border border-border">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex items-start gap-2">
          <Bot className="mt-0.5 size-4 text-accent-text" aria-hidden />
          <div>
            <h3 id="context-heading" className="text-sm font-semibold text-fg">
              Context sent to the LLM
            </h3>
            <p className="text-xs text-fg-muted">
              {context.reranked ? 'Reranked' : 'Fused'} passages, best first. Answers cite them as
              [S1], [S2], …
            </p>
          </div>
        </div>
        <Badge tone={context.reranked ? 'accent' : 'neutral'}>
          {context.reranked ? 'Hybrid + rerank' : 'Hybrid (reranking off)'}
        </Badge>
      </div>

      {!context.has_evidence ? (
        <p className="flex gap-2 px-4 py-4 text-sm text-fg-muted">
          <TriangleAlert className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
          No passage qualifies as evidence. The assistant would reply that it couldn't find
          sufficient supporting information, without calling the LLM.
        </p>
      ) : (
        <>
          <div className="px-4 pt-3">
            <div className="flex justify-between text-xs text-fg-subtle tabular-nums">
              <span>
                {pluralize(context.passages.length, 'passage')} of up to {context.max_passages}
              </span>
              <span>
                {formatNumber(context.total_tokens)} / {formatNumber(context.max_tokens)} tokens
              </span>
            </div>
            <div
              className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-muted"
              role="meter"
              aria-label="Context token budget used"
              aria-valuenow={context.total_tokens}
              aria-valuemin={0}
              aria-valuemax={context.max_tokens}
            >
              <div className="h-full rounded-full bg-accent" style={{ width: `${usedShare}%` }} />
            </div>
          </div>
          <ol className="divide-y divide-border">
            {context.passages.map((passage, index) => (
              <li key={passage.chunk_id} className="flex gap-3 px-4 py-3">
                <span className="mt-0.5 shrink-0 rounded-md bg-accent-soft px-1.5 py-0.5 font-mono text-xs font-semibold text-accent-text">
                  S{index + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-fg">{passage.document_title}</p>
                  <p className="text-xs text-fg-subtle">
                    {formatPageRange(passage.page_start, passage.page_end)}
                    {passage.section && ` · ${passage.section}`} · {passage.token_count} tokens
                  </p>
                  <p className="mt-1 line-clamp-2 text-xs text-fg-muted">{passage.text}</p>
                </div>
              </li>
            ))}
          </ol>
        </>
      )}

      <p className="border-t border-border px-4 py-2.5 text-xs text-fg-subtle">
        {dropped.length > 0 ? `Left out: ${dropped.join(' · ')}.` : 'Nothing was left out.'}{' '}
        Evidence threshold:{' '}
        {context.min_score === null ? 'off' : `cross-encoder score ≥ ${context.min_score}`}.
      </p>
    </section>
  )
}
