/**
 * Links between features that carry a selection of papers, so one feature
 * can hand its papers to another (ask about a paper, analyse the papers just
 * compared, ...). Each page reads `papers` from its URL.
 */

function withPapers(path: string, documentIds: string[]): string {
  return documentIds.length ? `${path}?papers=${documentIds.map(encodeURIComponent).join(',')}` : path
}

/** The Research Assistant with its paper filter set to these papers. */
export const askAboutUrl = (documentIds: string[]) => withPapers('/assistant', documentIds)

export const compareUrl = (documentIds: string[]) => withPapers('/comparison', documentIds)

export const insightsUrl = (documentIds: string[]) => withPapers('/insights', documentIds)

export const summaryUrl = (documentId: string) => `/papers/${encodeURIComponent(documentId)}/summary`

/** Paper ids from a `papers=a,b` query parameter. */
export function papersFromParams(params: URLSearchParams): string[] {
  return (params.get('papers') ?? '').split(',').filter(Boolean)
}
