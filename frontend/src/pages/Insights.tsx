import { ChevronDown, Info, KeyRound, Lightbulb, RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router'
import { CitationDialog, type CitedPassage } from '../components/research/CitationDialog'
import { FindingText } from '../components/research/FindingText'
import { PaperPicker } from '../components/research/PaperPicker'
import { Badge, type BadgeTone } from '../components/ui/Badge'
import { Button, ButtonLink } from '../components/ui/Button'
import { Card, CardHeader } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'
import { ErrorState, Skeleton, Spinner } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { ApiError, api } from '../services/api'
import type { GapAnalysis, GapFinding } from '../types/api'
import { formatRelativeTime, pluralize } from '../utils/format'

// Used until /api/system/info has loaded; the server enforces the real limit.
const DEFAULT_MAX_PAPERS = 5

const KINDS = {
  stated: { label: 'Stated by the papers', tone: 'success' },
  observed: { label: 'Observed across papers', tone: 'accent' },
  inferred: { label: 'Inferred', tone: 'warning' },
} satisfies Record<string, { label: string; tone: BadgeTone }>

const PATTERNS = [
  {
    key: 'recurring_challenges',
    title: 'Recurring challenges',
    description: 'Challenges or limitations that appear in two or more of the papers.',
  },
  {
    key: 'differences',
    title: 'Where the papers differ',
    description: 'Differences in assumptions, data, evaluation or method.',
  },
  {
    key: 'limited_discussion',
    title: 'Limited discussion',
    description: 'Topics that only some papers address, or that get little attention.',
  },
] as const

const NONE_FOUND = 'None found in the retrieved passages.'

function toApiError(cause: unknown): ApiError {
  return cause instanceof ApiError ? cause : new ApiError(0, 'unknown', 'Something went wrong.')
}

function sameIds(first: string[], second: string[]): boolean {
  return first.length === second.length && first.every((id, index) => id === second[index])
}

function Section({
  title,
  description,
  kind,
  children,
}: {
  title: string
  description: string
  kind: keyof typeof KINDS
  children: ReactNode
}) {
  return (
    <Card className="overflow-hidden">
      <CardHeader
        title={title}
        description={description}
        action={<Badge tone={KINDS[kind].tone}>{KINDS[kind].label}</Badge>}
      />
      <div className="divide-y divide-border">{children}</div>
    </Card>
  )
}

function PaperLabels({ ids, analysis }: { ids: string[]; analysis: GapAnalysis }) {
  return (
    <span className="flex flex-wrap gap-1">
      {ids.map((id) => {
        const paper = analysis.papers.find((item) => item.document_id === id)
        return (
          paper && (
            <span
              key={id}
              title={paper.title}
              className="rounded bg-surface-muted px-1.5 font-mono text-[0.7rem] font-semibold text-fg-muted"
            >
              {paper.label}
            </span>
          )
        )
      })}
    </span>
  )
}

function Results({
  analysis,
  onCite,
}: {
  analysis: GapAnalysis
  onCite: (passage: CitedPassage) => void
}) {
  const [showLeftOut, setShowLeftOut] = useState(false)
  const leftOut = analysis.left_out.length + analysis.malformed_items

  const findingList = (findings: GapFinding[], label: (finding: GapFinding) => string) =>
    findings.length === 0 ? (
      <p className="px-5 py-4 text-sm text-fg-subtle italic">{NONE_FOUND}</p>
    ) : (
      findings.map((finding, index) => (
        <div key={index} className="space-y-2 px-5 py-4">
          {finding.title && (
            <h4 className="flex flex-wrap items-center gap-2 text-sm font-semibold text-fg">
              {finding.title}
              <PaperLabels ids={finding.document_ids} analysis={analysis} />
            </h4>
          )}
          <FindingText finding={finding} analysis={analysis} label={label(finding)} onCite={onCite} />
        </div>
      ))
    )

  return (
    <>
      <div className="flex gap-3 rounded-xl border border-accent/30 bg-accent-soft px-4 py-3 text-sm text-fg">
        <Info className="mt-0.5 size-4 shrink-0 text-accent-text" aria-hidden />
        <div className="space-y-1">
          <p>{analysis.scope_note}</p>
          <p className="text-xs text-fg-muted">
            Every finding cites the passages it rests on (select a citation to read it), and was
            checked before being shown: a pattern across papers must cite at least two of them.
          </p>
        </div>
      </div>

      <Card className="px-5 py-4">
        <h3 className="text-xs font-medium text-fg-muted">Papers analysed</h3>
        <ul className="mt-2 space-y-1.5">
          {analysis.papers.map((paper) => (
            <li key={paper.document_id} className="flex items-baseline gap-2 text-sm">
              <span className="font-mono text-xs font-semibold text-fg-muted">{paper.label}</span>
              <Link
                to={`/papers/${paper.document_id}/summary`}
                className="text-fg hover:text-accent-text hover:underline"
              >
                {paper.title}
              </Link>
            </li>
          ))}
        </ul>
      </Card>

      <Section
        title="Stated limitations"
        description="What each paper says about the limits of its own work."
        kind="stated"
      >
        {analysis.papers.map((paper) => {
          const findings = analysis.stated_limitations.filter((finding) =>
            finding.document_ids.includes(paper.document_id),
          )
          return (
            <div key={paper.document_id} className="space-y-3 px-5 py-4">
              <h4 className="text-sm font-semibold text-fg">
                <span className="mr-2 font-mono text-xs text-fg-muted">{paper.label}</span>
                {paper.title}
              </h4>
              {findings.length === 0 ? (
                <p className="text-sm text-fg-subtle italic">
                  No limitation stated in the retrieved passages.
                </p>
              ) : (
                <ul className="space-y-3">
                  {findings.map((finding, index) => (
                    <li key={index} className="border-l-2 border-border pl-3">
                      <FindingText
                        finding={finding}
                        analysis={analysis}
                        label="Stated limitation"
                        onCite={onCite}
                      />
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )
        })}
      </Section>

      <Section
        title="Patterns across these papers"
        description="What the selected papers have in common, where they differ, and what few of them cover."
        kind="observed"
      >
        {PATTERNS.map((pattern) => (
          <div key={pattern.key}>
            <div className="bg-surface-muted/40 px-5 py-2.5">
              <h3 className="text-xs font-semibold tracking-wide text-fg-muted uppercase">
                {pattern.title}
              </h3>
              <p className="text-xs text-fg-subtle">{pattern.description}</p>
            </div>
            <div className="divide-y divide-border">
              {findingList(analysis[pattern.key], (finding) => finding.title ?? pattern.title)}
            </div>
          </div>
        ))}
      </Section>

      <Section
        title="Possible research directions"
        description="Suggestions that follow from the findings above. The evidence is cited; the direction itself is an inference."
        kind="inferred"
      >
        {analysis.opportunities.length === 0 ? (
          <p className="px-5 py-4 text-sm text-fg-subtle italic">{NONE_FOUND}</p>
        ) : (
          analysis.opportunities.map((opportunity, index) => (
            <div key={index} className="space-y-3 px-5 py-4">
              <h4 className="flex flex-wrap items-center gap-2 text-sm font-semibold text-fg">
                {opportunity.title}
                <PaperLabels ids={opportunity.document_ids} analysis={analysis} />
              </h4>
              <div>
                <p className="mb-1 text-xs font-medium text-fg-muted">What the papers show</p>
                <FindingText
                  finding={opportunity}
                  analysis={analysis}
                  label={opportunity.title ?? 'Research direction'}
                  onCite={onCite}
                />
              </div>
              <div className="flex gap-2.5 rounded-lg border border-dashed border-warning/40 px-3 py-2.5">
                <Lightbulb className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
                <div>
                  <p className="text-sm text-fg">{opportunity.suggestion}</p>
                  <p className="mt-1 text-xs text-fg-subtle">
                    Inferred from the evidence above, not stated by the papers.
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </Section>

      {leftOut > 0 && (
        <Card>
          <button
            type="button"
            onClick={() => setShowLeftOut((value) => !value)}
            aria-expanded={showLeftOut}
            className="flex w-full items-center justify-between gap-3 px-5 py-3 text-left text-sm text-fg-muted hover:text-fg"
          >
            <span>
              {pluralize(leftOut, 'proposed finding')} left out because the evidence didn't hold up
            </span>
            <ChevronDown
              className={`size-4 shrink-0 transition-transform ${showLeftOut ? 'rotate-180' : ''}`}
              aria-hidden
            />
          </button>
          {showLeftOut && (
            <ul className="space-y-2 border-t border-border px-5 py-4 text-sm">
              {analysis.left_out.map((item, index) => (
                <li key={index}>
                  <span className="text-fg">{item.title}</span>
                  <span className="block text-xs text-fg-subtle">{item.explanation}</span>
                </li>
              ))}
              {analysis.malformed_items > 0 && (
                <li className="text-xs text-fg-subtle">
                  {pluralize(analysis.malformed_items, 'item')} in the model's reply had the wrong
                  shape and were ignored.
                </li>
              )}
            </ul>
          )}
        </Card>
      )}
    </>
  )
}

export default function Insights() {
  const [searchParams, setSearchParams] = useSearchParams()
  const documents = useApi(api.listDocuments)
  const { data: info } = useApi(api.systemInfo)
  const { data: health } = useApi(api.health)
  const maxPapers = info?.analysis.max_gap_papers ?? DEFAULT_MAX_PAPERS
  const searchable = (documents.data?.items ?? []).filter((doc) => doc.indexing_status === 'indexed')

  // Papers in the URL are analysed again on load (e.g. after a reload); the
  // analysis is saved, so this makes no new model call.
  const [initialIds] = useState(() => (searchParams.get('papers') ?? '').split(',').filter(Boolean))
  const [selected, setSelected] = useState<string[]>(initialIds)
  const [analysis, setAnalysis] = useState<GapAnalysis | null>(null)
  const [running, setRunning] = useState(initialIds.length >= 2)
  const [error, setError] = useState<ApiError | null>(null)
  const [viewing, setViewing] = useState<CitedPassage | null>(null)

  const run = async (ids: string[], refresh = false) => {
    setRunning(true)
    setError(null)
    try {
      setAnalysis(await api.gaps(ids, refresh))
      setSearchParams({ papers: ids.join(',') }, { replace: true })
    } catch (cause) {
      setError(toApiError(cause))
    } finally {
      setRunning(false)
    }
  }

  useEffect(() => {
    if (initialIds.length < 2) return
    const controller = new AbortController()
    api
      .gaps(initialIds, false, controller.signal)
      .then(setAnalysis)
      .catch((cause) => {
        if (!controller.signal.aborted) setError(toApiError(cause))
      })
      .finally(() => {
        if (!controller.signal.aborted) setRunning(false)
      })
    return () => controller.abort()
  }, [initialIds])

  const analysed = analysis?.papers.map((paper) => paper.document_id) ?? []
  const canRun = selected.length >= 2 && !running && !sameIds(selected, analysed)

  if (documents.data && searchable.length < 2) {
    return (
      <Card>
        <EmptyState
          icon={Lightbulb}
          title="Two searchable papers needed"
          description="Upload at least two papers and wait until they show as Searchable, then look for limitations, patterns and possible research directions across them."
          action={<ButtonLink to="/papers">Go to papers</ButtonLink>}
        />
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Choose papers to analyse"
          description="The analysis covers only the papers you select: it finds what they state, what they share or differ on, and what might follow, with a citation behind every finding."
        />
        <div className="space-y-4 p-5">
          {documents.error ? (
            <ErrorState
              title="Couldn't load your papers"
              message={documents.error.message}
              onRetry={documents.reload}
            />
          ) : documents.loading ? (
            <Skeleton className="h-40 w-full" />
          ) : (
            <PaperPicker
              papers={searchable}
              selected={selected}
              max={maxPapers}
              onChange={setSelected}
            />
          )}
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => void run(selected)} disabled={!canRun}>
              {running ? (
                <Spinner className="text-accent-fg" />
              ) : (
                <Lightbulb className="size-4" aria-hidden />
              )}
              Analyse {selected.length >= 2 ? pluralize(selected.length, 'paper') : 'papers'}
            </Button>
            {selected.length < 2 && (
              <span className="text-xs text-fg-subtle">Select at least two papers.</span>
            )}
          </div>
          {health && !health.llm_configured && (
            <p className="flex items-start gap-2 text-xs text-fg-muted">
              <KeyRound className="mt-0.5 size-3.5 shrink-0 text-warning" aria-hidden />
              <span>
                The analysis needs a Gemini API key in <code className="font-mono">backend/.env</code>{' '}
                (see{' '}
                <Link to="/settings" className="text-accent-text hover:underline">
                  Settings
                </Link>
                ).
              </span>
            </p>
          )}
        </div>
      </Card>

      {error && (
        <ErrorState
          title={
            error.code === 'llm_not_configured' ? 'Gemini API key needed' : "Couldn't analyse the papers"
          }
          message={error.message}
        />
      )}

      {running && (
        <Card className="flex items-start gap-3 p-5 text-sm text-fg-muted">
          <Spinner className="mt-0.5 size-5" />
          <div>
            <p className="font-medium text-fg">Analysing {pluralize(selected.length, 'paper')}…</p>
            <p className="mt-1">
              Retrieving each paper's limitations, assumptions, future work and evaluation scope,
              then checking every finding against them. This usually takes 15–45 seconds.
            </p>
          </div>
        </Card>
      )}

      {analysis && !running && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-fg-subtle">
              {analysis.cached ? 'Saved analysis from' : 'Analysed'}{' '}
              {formatRelativeTime(analysis.generated_at)} with {analysis.model} ·{' '}
              {pluralize(analysis.sources.length, 'passage')} read
            </p>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void run(analysed, true)}
              title="Retrieve the passages again and redo the analysis"
            >
              <RefreshCw className="size-3.5" aria-hidden />
              Regenerate
            </Button>
          </div>
          {analysis.truncated && (
            <p className="flex gap-1.5 text-xs text-warning">
              <TriangleAlert className="size-3.5" aria-hidden />
              The model hit its output limit, so some findings may be missing.
            </p>
          )}
          <Results analysis={analysis} onCite={setViewing} />
        </>
      )}

      <CitationDialog passage={viewing} onClose={() => setViewing(null)} />
    </div>
  )
}
