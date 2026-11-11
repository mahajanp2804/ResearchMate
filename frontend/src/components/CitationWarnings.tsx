import { ShieldAlert, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import type { CitationStatus } from '../types/api'

interface CitationWarningsProps {
  status: CitationStatus | null
  /** Claims that cite no source. */
  uncitedStatements: string[]
  /** Source ids the model cited without being given them (removed from the text). */
  removedIds: string[]
  /** What is being checked: "answer", "section", ... */
  noun: string
  /** What the text should be traceable to, e.g. "your papers" or "the paper". */
  evidence: string
}

/** Results of the citation check: uncited text, unsupported claims and removed citations. */
export function CitationWarnings({
  status,
  uncitedStatements,
  removedIds,
  noun,
  evidence,
}: CitationWarningsProps) {
  const [showStatements, setShowStatements] = useState(false)
  const count = uncitedStatements.length

  return (
    <>
      {status === 'uncited' && (
        <p className="flex gap-2 rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-xs text-fg">
          <ShieldAlert className="mt-px size-3.5 shrink-0 text-danger" aria-hidden />
          This {noun} doesn't cite any source, so it can't be traced to {evidence}. Treat it with
          caution.
        </p>
      )}
      {status === 'partially_cited' && count > 0 && (
        <div className="rounded-lg border border-warning/30 bg-warning-soft px-3 py-2 text-xs text-fg">
          <button
            type="button"
            onClick={() => setShowStatements((value) => !value)}
            aria-expanded={showStatements}
            className="flex w-full items-start gap-2 text-left"
          >
            <TriangleAlert className="mt-px size-3.5 shrink-0 text-warning" aria-hidden />
            <span className="flex-1">
              {count === 1
                ? "1 statement isn't backed by a source."
                : `${count} statements aren't backed by a source.`}{' '}
              <span className="text-accent-text">{showStatements ? 'Hide' : 'Show'}</span>
            </span>
          </button>
          {showStatements && (
            <ul className="mt-1.5 list-disc space-y-1 pl-9 text-fg-muted">
              {uncitedStatements.map((statement) => (
                <li key={statement}>{statement}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {removedIds.length > 0 && (
        <p className="text-xs text-fg-subtle">
          Removed {removedIds.length === 1 ? 'a citation' : `${removedIds.length} citations`} to{' '}
          {removedIds.length === 1 ? 'a source' : 'sources'} the model was not given (
          {removedIds.join(', ')}).
        </p>
      )}
    </>
  )
}
