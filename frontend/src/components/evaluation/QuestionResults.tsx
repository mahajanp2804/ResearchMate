import { Check, ChevronDown, X } from 'lucide-react'
import { Fragment, useState } from 'react'
import type { EvaluationRunDetail, QuestionCategory, QuestionResult, SearchMethod } from '../../types/api'
import { CATEGORY_LABELS, METHOD_LABELS, formatPercent, methodColor } from '../../utils/evaluation'
import { cn } from '../../utils/format'

function RankCell({ result, method, topK }: { result: QuestionResult; method: SearchMethod; topK: number }) {
  const retrieved = result.retrieval[method]
  if (!retrieved) return <span className="text-fg-subtle">—</span>
  const rank = retrieved.first_relevant_rank
  return rank ? (
    <span className="inline-flex items-center gap-1 text-fg tabular-nums" title={`First relevant passage at rank ${rank}`}>
      <Check className="size-3.5 text-success" aria-hidden />#{rank}
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-fg-subtle" title={`Not in the top ${topK}`}>
      <X className="size-3.5 text-danger" aria-hidden />
      <span className="sr-only">Not in the top {topK}</span>
      <span aria-hidden>&gt;{topK}</span>
    </span>
  )
}

function Details({ result, methods }: { result: QuestionResult; methods: SearchMethod[] }) {
  const answer = result.answer
  return (
    <div className="space-y-3 text-sm">
      {result.evidence.length > 0 ? (
        <div>
          <p className="text-xs font-medium text-fg-muted">Evidence (from the dataset)</p>
          <ul className="mt-1 space-y-1">
            {result.evidence.map((item, index) => (
              <li key={index} className="border-l-2 border-border pl-2.5 text-fg">
                “{item.text}”
                {item.page && <span className="ml-1 text-xs text-fg-subtle">p. {item.page}</span>}
                <span className="mt-0.5 block text-xs text-fg-subtle">
                  {methods
                    .map((method) => {
                      const rank = result.retrieval[method]?.evidence_ranks[index]
                      return `${METHOD_LABELS[method]}: ${rank ? `#${rank}` : 'not found'}`
                    })
                    .join(' · ')}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="text-fg-muted">
          No evidence: the papers don’t answer this, so it has no retrieval metrics.
        </p>
      )}
      {result.reference_answer && (
        <div>
          <p className="text-xs font-medium text-fg-muted">Reference answer</p>
          <p className="mt-1 text-fg">{result.reference_answer}</p>
        </div>
      )}
      {answer && (
        <div>
          <p className="text-xs font-medium text-fg-muted">
            Generated answer · {answer.refused ? 'refused' : 'answered'}
            {answer.refusal_correct ? '' : ' (wrongly)'}
          </p>
          <p className="mt-1 whitespace-pre-line text-fg">{answer.answer}</p>
          {answer.statements && answer.statements.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs">
              {answer.statements.map((statement, index) => (
                <li key={index} className="flex gap-1.5">
                  {statement.supported ? (
                    <Check className="mt-px size-3.5 shrink-0 text-success" aria-label="Supported" />
                  ) : (
                    <X className="mt-px size-3.5 shrink-0 text-danger" aria-label="Not supported" />
                  )}
                  <span className="text-fg-muted">{statement.text}</span>
                </li>
              ))}
            </ul>
          )}
          {!answer.refused && (
            <p className="mt-2 text-xs text-fg-subtle">
              Faithfulness {formatPercent(answer.faithfulness)} · Answer relevance{' '}
              {formatPercent(answer.answer_relevance)} · Context relevance{' '}
              {formatPercent(answer.context_relevance)} · Correctness{' '}
              {formatPercent(answer.correctness)}
              {answer.judge_error && ` · ${answer.judge_error}`}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

/** Where each method ranked each question's evidence, with the details on demand. */
export function QuestionResults({ run }: { run: EvaluationRunDetail }) {
  const [category, setCategory] = useState<QuestionCategory | 'all'>('all')
  const [open, setOpen] = useState<string | null>(null)
  const methods = run.config.methods
  const results = run.per_question_results.filter(
    (result) => category === 'all' || result.category === category,
  )
  const categories = [...new Set(run.per_question_results.map((result) => result.category))]

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3">
        <p className="text-xs text-fg-muted">
          Rank of the first passage containing the evidence (lower is better).
        </p>
        <label className="flex items-center gap-2 text-xs text-fg-muted">
          Question type
          <select
            value={category}
            onChange={(event) => setCategory(event.target.value as QuestionCategory | 'all')}
            className="h-8 rounded-lg border border-border bg-surface px-2 text-xs text-fg"
          >
            <option value="all">All ({run.per_question_results.length})</option>
            {categories.map((item) => (
              <option key={item} value={item}>
                {CATEGORY_LABELS[item]} (
                {run.per_question_results.filter((result) => result.category === item).length})
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="overflow-x-auto border-t border-border">
        <table className="w-full text-sm">
          <caption className="sr-only">Per-question results</caption>
          <thead>
            <tr className="border-b border-border text-xs text-fg-muted">
              <th scope="col" className="px-5 py-2 text-left font-medium">
                Question
              </th>
              {methods.map((method) => (
                <th key={method} scope="col" className="px-3 py-2 text-left font-medium whitespace-nowrap">
                  <span className="flex items-center gap-1.5">
                    <span className="size-2 rounded-sm" style={{ background: methodColor(method) }} aria-hidden />
                    {METHOD_LABELS[method]}
                  </span>
                </th>
              ))}
              {run.config.answer_quality && (
                <th scope="col" className="px-3 py-2 text-left font-medium">
                  Answer
                </th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {results.map((result) => {
              const expanded = open === result.key
              return (
                <Fragment key={result.key}>
                  <tr className={cn(expanded && 'bg-surface-muted/50')}>
                    <td className="max-w-md px-5 py-2.5">
                      <button
                        type="button"
                        onClick={() => setOpen(expanded ? null : result.key)}
                        aria-expanded={expanded}
                        className="flex w-full items-start gap-2 text-left"
                      >
                        <ChevronDown
                          className={cn(
                            'mt-0.5 size-3.5 shrink-0 text-fg-subtle transition-transform',
                            expanded && 'rotate-180',
                          )}
                          aria-hidden
                        />
                        <span>
                          <span className="text-fg">{result.question}</span>
                          <span className="mt-0.5 block text-xs text-fg-subtle">
                            {result.key} · {CATEGORY_LABELS[result.category]}
                          </span>
                        </span>
                      </button>
                    </td>
                    {methods.map((method) => (
                      <td key={method} className="px-3 py-2.5 align-top">
                        <RankCell result={result} method={method} topK={run.config.top_k} />
                      </td>
                    ))}
                    {run.config.answer_quality && (
                      <td className="px-3 py-2.5 align-top text-xs whitespace-nowrap">
                        {result.answer ? (
                          <span className={result.answer.refusal_correct ? 'text-fg' : 'text-danger'}>
                            {result.answer.refused ? 'Refused' : 'Answered'}
                            {!result.answer.refusal_correct && ' (wrongly)'}
                          </span>
                        ) : (
                          <span className="text-fg-subtle">—</span>
                        )}
                      </td>
                    )}
                  </tr>
                  {expanded && (
                    <tr className="bg-surface-muted/50">
                      <td colSpan={methods.length + 2} className="px-5 pt-1 pb-4 pl-11">
                        <Details result={result} methods={methods} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
