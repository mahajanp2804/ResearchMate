import type { QuestionCategory, RetrievalSummary, SearchMethod } from '../types/api'

/** Retrieval methods in pipeline order. A method's colour follows it, never its rank. */
export const METHOD_ORDER: SearchMethod[] = ['dense', 'bm25', 'hybrid', 'hybrid_rerank']

export const METHOD_LABELS: Record<SearchMethod, string> = {
  dense: 'Dense',
  bm25: 'BM25',
  hybrid: 'Hybrid (RRF)',
  hybrid_rerank: 'Hybrid + rerank',
}

export function methodColor(method: SearchMethod): string {
  return `var(--series-${METHOD_ORDER.indexOf(method) + 1})`
}

export const CATEGORY_LABELS: Record<QuestionCategory, string> = {
  factual: 'Factual',
  paraphrase: 'Paraphrase',
  keyword: 'Exact term or number',
  multi_paper: 'Needs two papers',
  unanswerable: 'Not answerable',
}

/** Short names for chart axes, where space is tight. */
export const CATEGORY_SHORT_LABELS: Record<QuestionCategory, string> = {
  factual: 'Factual',
  paraphrase: 'Paraphrase',
  keyword: 'Exact term',
  multi_paper: 'Two papers',
  unanswerable: 'Unanswerable',
}

export const CATEGORY_DESCRIPTIONS: Record<QuestionCategory, string> = {
  factual: 'Asks for a fact in roughly the paper’s own words.',
  paraphrase: 'Asks in different words from the paper.',
  keyword: 'Hinges on an exact number, name or setting.',
  multi_paper: 'Needs evidence from two papers.',
  unanswerable: 'The papers don’t answer it; the assistant should refuse.',
}

export function metricValue(summary: RetrievalSummary | undefined, key: string): number | null {
  const value = summary?.[key]
  return typeof value === 'number' ? value : null
}

/** Scores in 0–1 shown with three decimals, as IR papers do. */
export function formatScore(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toFixed(3)
}

export function formatPercent(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`
}
