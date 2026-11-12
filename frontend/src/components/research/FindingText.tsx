import { TriangleAlert } from 'lucide-react'
import type { GapAnalysis, GapFinding } from '../../types/api'
import { AnswerText } from '../chat/AnswerText'
import { CitationWarnings } from '../CitationWarnings'
import type { CitedPassage } from './CitationDialog'

interface FindingTextProps {
  finding: GapFinding
  analysis: GapAnalysis
  /** Names the finding in the passage dialog, e.g. "Stated limitation". */
  label: string
  onCite: (passage: CitedPassage) => void
  /** Show the over-claiming warning here (off when the caller places it). */
  showScopeWarning?: boolean
}

/** Shown when a finding's wording claims more than the selected papers can show. */
export function ScopeWarning({ paperCount }: { paperCount: number }) {
  return (
    <p className="flex gap-2 rounded-lg border border-warning/30 bg-warning-soft px-3 py-2 text-xs text-fg">
      <TriangleAlert className="mt-px size-3.5 shrink-0 text-warning" aria-hidden />
      This wording claims more than {paperCount} selected papers can show. Treat it as a statement
      about these papers only.
    </p>
  )
}

/**
 * A gap-analysis finding with citation chips labelled by paper and page
 * ("P2 p. 4"), and the warnings from the evidence checks.
 */
export function FindingText({
  finding,
  analysis,
  label,
  onCite,
  showScopeWarning = true,
}: FindingTextProps) {
  const citations = new Map(finding.citations.map((citation) => [citation.source_id, citation]))
  const sources = new Map(analysis.sources.map((source) => [source.source_id, source]))
  const papers = new Map(analysis.papers.map((paper) => [paper.document_id, paper]))

  const chipLabel = (sourceId: string) => {
    const paper = papers.get(sources.get(sourceId)?.document_id ?? '')
    const page = citations.get(sourceId)?.page
    return paper && page !== undefined ? `${paper.label} p. ${page}` : sourceId
  }
  const cite = (sourceId: string) => {
    const source = sources.get(sourceId)
    const citation = citations.get(sourceId)
    const paper = source && papers.get(source.document_id)
    if (!source || !citation || !paper) return
    onCite({
      documentId: paper.document_id,
      documentTitle: `${paper.label} · ${paper.title}`,
      aspectLabel: label,
      source,
      citation,
    })
  }

  return (
    <div className="space-y-2">
      <AnswerText
        text={finding.text}
        sourceIds={[...citations.keys()]}
        onCite={cite}
        chipLabel={chipLabel}
      />
      <CitationWarnings
        status={finding.citation_status}
        uncitedStatements={finding.uncited_statements}
        removedIds={finding.invalid_citation_ids}
        noun="finding"
        evidence="the selected papers"
      />
      {finding.misattributed_citation_ids.length > 0 && (
        <p className="text-xs text-fg-subtle">
          Removed {finding.misattributed_citation_ids.length === 1 ? 'a citation' : 'citations'} to
          another paper's passage ({finding.misattributed_citation_ids.join(', ')}): a stated
          limitation may only cite its own paper.
        </p>
      )}
      {showScopeWarning && finding.beyond_scope && (
        <ScopeWarning paperCount={analysis.papers.length} />
      )}
    </div>
  )
}
