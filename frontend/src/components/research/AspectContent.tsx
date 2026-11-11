import type { PaperProfile, ProfileAspect } from '../../types/api'
import { AnswerText } from '../chat/AnswerText'
import { CitationWarnings } from '../CitationWarnings'
import type { CitedPassage } from './CitationDialog'

interface AspectContentProps {
  aspect: ProfileAspect
  profile: PaperProfile
  /** Shown when the paper's passages don't state this aspect. */
  notFoundText: string
  /** "section" or "cell", used in warnings. */
  noun: string
  onCite: (passage: CitedPassage) => void
}

/**
 * One summary section or comparison cell: its text with page-labelled
 * citation chips and the citation check's warnings, or a plain "not found".
 */
export function AspectContent({ aspect, profile, notFoundText, noun, onCite }: AspectContentProps) {
  if (aspect.status === 'not_found' || !aspect.text) {
    return <p className="text-sm text-fg-subtle italic">{notFoundText}</p>
  }

  const citations = new Map(aspect.citations.map((citation) => [citation.source_id, citation]))
  const cite = (sourceId: string) => {
    const citation = citations.get(sourceId)
    const source = profile.sources.find((item) => item.source_id === sourceId)
    if (!citation || !source) return
    onCite({
      documentId: profile.document_id,
      documentTitle: profile.document_title,
      aspectLabel: aspect.label,
      source,
      citation,
    })
  }

  return (
    <div className="space-y-2">
      <AnswerText
        text={aspect.text}
        sourceIds={[...citations.keys()]}
        onCite={cite}
        chipLabel={(sourceId) => {
          const page = citations.get(sourceId)?.page
          return page === undefined ? sourceId : `p. ${page}`
        }}
      />
      <CitationWarnings
        status={aspect.citation_status}
        uncitedStatements={aspect.uncited_statements}
        removedIds={aspect.invalid_citation_ids}
        noun={noun}
        evidence="the paper"
      />
    </div>
  )
}
