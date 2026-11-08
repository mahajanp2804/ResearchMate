import { ExternalLink, FileText, Search, Timer } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { useApi } from '../hooks/useApi'
import { ApiError, api } from '../services/api'
import type {
  ContextResponse,
  MethodResults,
  SearchHit,
  SearchMethod,
  SearchResponse,
} from '../types/api'
import { formatPageRange, pluralize } from '../utils/format'
import { queryTerms, snippetAroundMatch } from '../utils/highlight'
import { ContextPreview } from './ContextPreview'
import { DocumentFilter } from './DocumentFilter'
import { Highlighted } from './Highlighted'
import { Badge } from './ui/Badge'
import { Button, ButtonLink } from './ui/Button'
import { Card, CardHeader } from './ui/Card'
import { EmptyState } from './ui/EmptyState'
import { ErrorState, Skeleton, Spinner } from './ui/Feedback'

interface MethodInfo {
  label: string
  description: string
  empty: string
  scoreLabel: string
  digits: number
}

const METHODS: Record<SearchMethod, MethodInfo> = {
  dense: {
    label: 'Dense · semantic',
    description: 'Embeddings + FAISS cosine similarity. Finds paraphrases and related ideas.',
    empty: 'No passages found.',
    scoreLabel: 'Cosine similarity',
    digits: 3,
  },
  bm25: {
    label: 'BM25 · keyword',
    description: 'Exact term matching. Finds names, acronyms, datasets and numbers.',
    empty: 'No passage shares a word with the query, so BM25 has nothing to rank.',
    scoreLabel: 'BM25 score',
    digits: 2,
  },
  hybrid: {
    label: 'Hybrid · RRF',
    description: 'Dense and BM25 lists fused by rank (Reciprocal Rank Fusion).',
    empty: 'Neither retriever found a passage.',
    scoreLabel: 'RRF score: sum of 1/(60 + rank) over the lists that found it',
    digits: 4,
  },
  hybrid_rerank: {
    label: 'Hybrid + rerank',
    description:
      'Fused candidates rescored by a cross-encoder reading question and passage together. This is what the answer pipeline uses.',
    empty: 'Neither retriever found a passage.',
    scoreLabel: 'Cross-encoder score (logit): higher is more relevant',
    digits: 2,
  },
}

const ALL_METHODS: SearchMethod[] = ['dense', 'bm25', 'hybrid', 'hybrid_rerank']

/** "dense #3 · BM25 #1 · was #5 before rerank" for fused and reranked hits. */
function provenance(hit: SearchHit): string | null {
  const parts: string[] = []
  if (hit.source_ranks.dense) parts.push(`dense #${hit.source_ranks.dense}`)
  if (hit.source_ranks.bm25) parts.push(`BM25 #${hit.source_ranks.bm25}`)
  if (hit.source_ranks.hybrid) parts.push(`was #${hit.source_ranks.hybrid} before rerank`)
  return parts.length > 0 ? parts.join(' · ') : null
}

const EXAMPLES = [
  'Which datasets were used?',
  'What are the limitations of the approach?',
  'How do clients with little compute participate?',
]

const TOP_K_OPTIONS = [5, 10, 20]

function HitCard({ hit, method, terms }: { hit: SearchHit; method: SearchMethod; terms: string[] }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <li className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md bg-surface-muted text-xs font-semibold text-fg-muted">
          {hit.rank}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <p className="line-clamp-2 text-sm font-medium text-fg">{hit.document_title}</p>
            <span
              className="shrink-0 font-mono text-xs text-fg-subtle tabular-nums"
              title={METHODS[method].scoreLabel}
            >
              {hit.score.toFixed(METHODS[method].digits)}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-fg-subtle">
            <Badge tone="accent">{formatPageRange(hit.page_start, hit.page_end)}</Badge>
            {hit.section && <span className="truncate">{hit.section}</span>}
          </div>
          {(method === 'hybrid' || method === 'hybrid_rerank') && provenance(hit) && (
            <p className="mt-1 text-xs text-fg-subtle">Found by {provenance(hit)}</p>
          )}
          <p
            className={`mt-2 text-sm leading-relaxed whitespace-pre-line text-fg-muted ${expanded ? '' : 'line-clamp-4'}`}
          >
            <Highlighted text={expanded ? hit.text : snippetAroundMatch(hit.text, terms)} terms={terms} />
          </p>
          <div className="mt-2 flex items-center gap-3 text-xs">
            <button
              type="button"
              onClick={() => setExpanded((value) => !value)}
              className="font-medium text-accent-text hover:underline"
            >
              {expanded ? 'Show less' : 'Show full passage'}
            </button>
            <a
              href={api.documentFileUrl(hit.document_id, hit.page_start)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-fg-muted hover:text-fg"
            >
              <ExternalLink className="size-3" aria-hidden /> Open at page {hit.page_start}
            </a>
          </div>
        </div>
      </div>
    </li>
  )
}

function MethodColumn({ result, terms }: { result: MethodResults; terms: string[] }) {
  const info = METHODS[result.method]
  return (
    <section aria-label={info.label} className="min-w-0">
      <div className="mb-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-fg">{info.label}</h3>
          <span className="inline-flex items-center gap-1 text-xs text-fg-subtle tabular-nums">
            <Timer className="size-3" aria-hidden />
            {result.latency_ms.toFixed(0)} ms
          </span>
        </div>
        <p className="mt-0.5 text-xs text-fg-muted">{info.description}</p>
      </div>
      {result.hits.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border-strong px-4 py-6 text-center text-sm text-fg-muted">
          {info.empty}
        </p>
      ) : (
        <ol className="space-y-3">
          {result.hits.map((hit) => (
            <HitCard key={hit.chunk_id} hit={hit} method={result.method} terms={terms} />
          ))}
        </ol>
      )}
    </section>
  )
}

/** Compare what each retrieval stage finds for the same question. */
export function RetrievalPlayground() {
  const documents = useApi(api.listDocuments)
  const searchable = (documents.data?.items ?? []).filter((doc) => doc.indexing_status === 'indexed')

  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string[] | null>(null)
  const [topK, setTopK] = useState(5)
  const [methods, setMethods] = useState<SearchMethod[]>(ALL_METHODS)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [context, setContext] = useState<ContextResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [searching, setSearching] = useState(false)

  const run = async (text: string) => {
    if (!text.trim()) return
    setSearching(true)
    setError(null)
    // Drop selections of papers deleted since they were chosen.
    const stillSearchable = selected?.filter((id) => searchable.some((doc) => doc.id === id))
    const documentIds = stillSearchable && stillSearchable.length > 0 ? stillSearchable : null
    try {
      // The context preview runs the same question through the answer pipeline.
      const [results, selected] = await Promise.all([
        api.search({ query: text, methods, top_k: topK, document_ids: documentIds }),
        api.context({ query: text, document_ids: documentIds }),
      ])
      setResponse(results)
      setContext(selected)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Search failed. Please try again.')
    } finally {
      setSearching(false)
    }
  }

  const toggleMethod = (method: SearchMethod) => {
    setMethods((current) => {
      const next = current.includes(method)
        ? current.filter((item) => item !== method)
        : ALL_METHODS.filter((item) => item === method || current.includes(item))
      return next.length > 0 ? next : current // keep at least one stage
    })
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    void run(query)
  }

  const header = (
    <CardHeader
      title="Retrieval playground"
      description="Compare what each stage of the retrieval pipeline finds for the same question. Nothing here calls the LLM."
    />
  )

  if (documents.loading && !documents.data) {
    return (
      <Card>
        {header}
        <div className="p-5">
          <Skeleton className="h-9 w-full" />
        </div>
      </Card>
    )
  }

  if (documents.data && searchable.length === 0) {
    return (
      <Card>
        {header}
        <EmptyState
          icon={FileText}
          title="No searchable papers yet"
          description="Upload papers and wait until they show as Searchable, then try a query here."
          action={<ButtonLink to="/papers">Go to papers</ButtonLink>}
        />
      </Card>
    )
  }

  const terms = queryTerms(response?.query ?? '')

  return (
    <Card>
      {header}
      <form onSubmit={submit} className="space-y-3 border-b border-border p-5">
        <label className="relative block">
          <span className="sr-only">Question or search terms</span>
          <Search
            className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-fg-subtle"
            aria-hidden
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask about your papers, e.g. Which datasets were used?"
            maxLength={1000}
            className="h-10 w-full rounded-lg border border-border bg-surface pr-3 pl-9 text-sm text-fg shadow-card placeholder:text-fg-subtle focus:border-accent focus:outline-none"
          />
        </label>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <DocumentFilter documents={searchable} selected={selected} onChange={setSelected} />
          <label className="flex items-center gap-2 text-sm text-fg-muted">
            Results per stage
            <select
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              className="h-9 rounded-lg border border-border bg-surface px-2 text-sm text-fg shadow-card"
            >
              {TOP_K_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <Button type="submit" disabled={searching || !query.trim()} className="sm:ml-auto">
            {searching ? <Spinner className="text-accent-fg" /> : <Search className="size-4" aria-hidden />}
            Search
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Stages to compare">
          <span className="text-xs text-fg-subtle">Compare:</span>
          {ALL_METHODS.map((method) => {
            const active = methods.includes(method)
            return (
              <button
                key={method}
                type="button"
                aria-pressed={active}
                onClick={() => toggleMethod(method)}
                className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                  active
                    ? 'border-accent bg-accent-soft text-accent-text'
                    : 'border-border text-fg-muted hover:bg-surface-muted'
                }`}
              >
                {METHODS[method].label}
              </button>
            )
          })}
        </div>
        {!response && (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs text-fg-subtle">Try:</span>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => {
                  setQuery(example)
                  void run(example)
                }}
                className="rounded-full border border-border px-3 py-1 text-xs text-fg-muted hover:bg-surface-muted hover:text-fg"
              >
                {example}
              </button>
            ))}
          </div>
        )}
      </form>

      <div className="space-y-4 p-5">
        {error && <ErrorState title="Search failed" message={error} />}
        {response && (
          <>
            <p className="text-xs text-fg-subtle">
              Searched {pluralize(response.searched_document_count, 'paper')}
              {response.skipped_document_ids.length > 0 &&
                ` · ${pluralize(response.skipped_document_ids.length, 'selected paper')} skipped because not yet searchable`}
            </p>
            <div className="grid gap-6 lg:grid-cols-2">
              {response.results.map((result) => (
                <MethodColumn key={result.method} result={result} terms={terms} />
              ))}
            </div>
            {context && <ContextPreview context={context} />}
          </>
        )}
        {!response && !error && (
          <p className="text-sm text-fg-muted">
            Results from each stage appear here side by side, with page references, the words
            they matched, and where fused results came from.
          </p>
        )}
      </div>
    </Card>
  )
}
