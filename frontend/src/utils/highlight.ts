// Words too common to be worth highlighting in search results.
const STOPWORDS = new Set(
  'the and for are was were with that this from what which how does did into have has its their they than then there these those when where who why will would about also been can could not our out such use used using'.split(
    ' ',
  ),
)

/** Distinct meaningful words of a query, for highlighting matches in results. */
export function queryTerms(query: string): string[] {
  const words = query.toLowerCase().match(/[a-z0-9][a-z0-9-]*/g) ?? []
  return [...new Set(words)].filter((word) => word.length >= 3 && !STOPWORDS.has(word))
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export type HighlightPart = { text: string; match: boolean }

/** Split text into plain and matching parts; a term also matches its word endings (plurals). */
export function highlightParts(text: string, terms: string[]): HighlightPart[] {
  if (terms.length === 0) return [{ text, match: false }]
  const pattern = new RegExp(`\\b(${terms.map(escapeRegExp).join('|')})\\w*`, 'gi')
  const parts: HighlightPart[] = []
  let last = 0
  for (const found of text.matchAll(pattern)) {
    const start = found.index ?? 0
    if (start > last) parts.push({ text: text.slice(last, start), match: false })
    parts.push({ text: found[0], match: true })
    last = start + found[0].length
  }
  if (last < text.length) parts.push({ text: text.slice(last), match: false })
  return parts
}

/**
 * Start a preview just before the first query match, so a clamped passage
 * shows why it was retrieved instead of a title or heading. Cuts at a word
 * boundary and marks the cut with an ellipsis.
 */
export function snippetAroundMatch(text: string, terms: string[], leadChars = 80): string {
  const parts = highlightParts(text, terms)
  const firstMatch = parts.findIndex((part) => part.match)
  if (firstMatch <= 0) return text
  const offset = parts.slice(0, firstMatch).reduce((length, part) => length + part.text.length, 0)
  if (offset <= leadChars) return text
  const wordStart = text.indexOf(' ', offset - leadChars)
  return `…${text.slice(wordStart + 1).replace(/\s+/g, ' ')}`
}
