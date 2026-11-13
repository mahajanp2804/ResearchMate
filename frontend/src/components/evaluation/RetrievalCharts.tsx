import type { ReactElement } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { EvaluationRunSummary, QuestionCategory, SearchMethod } from '../../types/api'
import {
  CATEGORY_LABELS,
  METHOD_LABELS,
  formatScore,
  methodColor,
  metricValue,
} from '../../utils/evaluation'

const AXIS = {
  tick: { fill: 'var(--fg-subtle)', fontSize: 12 },
  axisLine: { stroke: 'var(--border-strong)' },
  tickLine: false,
} as const
const SCORE_TICKS = [0, 0.25, 0.5, 0.75, 1]

interface TooltipEntry {
  dataKey?: unknown
  value?: unknown
}

/** Values lead, method names follow, each keyed by a short line in its colour. */
function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: readonly TooltipEntry[]
  label?: unknown
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 font-medium text-fg-muted">{String(label)}</p>
      <ul className="space-y-0.5">
        {payload.map((entry) => {
          const method = entry.dataKey as SearchMethod
          return (
            <li key={method} className="flex items-center gap-2">
              <span
                className="h-0.5 w-3 shrink-0 rounded-full"
                style={{ background: methodColor(method) }}
                aria-hidden
              />
              <span className="font-semibold text-fg tabular-nums">
                {formatScore(typeof entry.value === 'number' ? entry.value : null)}
              </span>
              <span className="text-fg-muted">{METHOD_LABELS[method]}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

/** The methods' shared legend: one swatch per method, in pipeline order. */
export function MethodLegend({ methods, mark }: { methods: SearchMethod[]; mark: 'bar' | 'line' }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted" aria-label="Methods">
      {methods.map((method) => (
        <li key={method} className="flex items-center gap-1.5">
          <span
            className={mark === 'bar' ? 'size-2.5 rounded-sm' : 'h-0.5 w-3.5 rounded-full'}
            style={{ background: methodColor(method) }}
            aria-hidden
          />
          {METHOD_LABELS[method]}
        </li>
      ))}
    </ul>
  )
}

function ChartFrame({
  title,
  description,
  summary,
  children,
}: {
  title: string
  description: string
  /** Text alternative for screen readers; the table view has every number. */
  summary: string
  children: ReactElement
}) {
  return (
    <figure className="min-w-0">
      <figcaption>
        <h3 className="text-sm font-semibold text-fg">{title}</h3>
        <p className="mt-0.5 text-xs text-fg-muted">{description}</p>
      </figcaption>
      <div className="mt-3 h-60" role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </figure>
  )
}

function bars(methods: SearchMethod[]) {
  return methods.map((method) => (
    <Bar
      key={method}
      dataKey={method}
      name={METHOD_LABELS[method]}
      fill={methodColor(method)}
      maxBarSize={24}
      radius={[4, 4, 0, 0]}
      isAnimationActive={false}
    />
  ))
}

/** Precision, recall, hit rate and MRR at cut-off K, one bar per method. */
export function MetricsAtKChart({ run, k }: { run: EvaluationRunSummary; k: number }) {
  const methods = run.config.methods
  const retrieval = run.metrics.retrieval ?? {}
  const rows = [
    { label: `Precision@${k}`, key: `precision@${k}` },
    { label: `Recall@${k}`, key: `recall@${k}` },
    { label: `Hit rate@${k}`, key: `hit@${k}` },
    { label: 'MRR', key: 'mrr' },
  ].map(({ label, key }) => ({
    label,
    ...Object.fromEntries(methods.map((m) => [m, metricValue(retrieval[m], key)])),
  }))
  const best = [...methods].sort(
    (a, b) => (metricValue(retrieval[b], 'mrr') ?? 0) - (metricValue(retrieval[a], 'mrr') ?? 0),
  )[0]

  return (
    <ChartFrame
      title={`Retrieval quality at K = ${k}`}
      description="Higher is better. Precision is capped by how few passages hold each answer."
      summary={`Bar chart of precision, recall and hit rate at ${k} and MRR for each method. Highest MRR: ${METHOD_LABELS[best]}.`}
    >
      <BarChart data={rows} barGap={2} barCategoryGap="22%" margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="label" {...AXIS} />
        <YAxis domain={[0, 1]} ticks={SCORE_TICKS} {...AXIS} axisLine={false} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: 'var(--surface-muted)' }} />
        {bars(methods)}
      </BarChart>
    </ChartFrame>
  )
}

/** Recall as more passages are read: the classic retrieval curve. */
export function RecallCurveChart({ run }: { run: EvaluationRunSummary }) {
  const methods = run.config.methods
  const retrieval = run.metrics.retrieval ?? {}
  const rows = run.config.cutoffs.map((k) => ({
    label: `K = ${k}`,
    ...Object.fromEntries(methods.map((m) => [m, metricValue(retrieval[m], `recall@${k}`)])),
  }))
  return (
    <ChartFrame
      title="Recall as K grows"
      description="Share of each question’s evidence found within the top K passages."
      summary="Line chart of recall at 1, 3, 5 and 10 passages for each method."
    >
      <LineChart data={rows} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="label" {...AXIS} padding={{ left: 12, right: 12 }} />
        <YAxis domain={[0, 1]} ticks={SCORE_TICKS} {...AXIS} axisLine={false} />
        <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'var(--border-strong)' }} />
        {methods.map((method) => (
          <Line
            key={method}
            type="linear"
            dataKey={method}
            name={METHOD_LABELS[method]}
            stroke={methodColor(method)}
            strokeWidth={2}
            dot={{ r: 4, fill: methodColor(method), stroke: 'var(--surface)', strokeWidth: 2 }}
            activeDot={{ r: 5, stroke: 'var(--surface)', strokeWidth: 2 }}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartFrame>
  )
}

/** MRR per question category: where each method wins or loses. */
export function CategoryChart({ run }: { run: EvaluationRunSummary }) {
  const methods = run.config.methods
  const byCategory = run.metrics.retrieval_by_category ?? {}
  const categories = (Object.keys(CATEGORY_LABELS) as QuestionCategory[]).filter(
    (category) => byCategory[methods[0]]?.[category],
  )
  const rows = categories.map((category) => ({
    label: `${CATEGORY_LABELS[category]} (${byCategory[methods[0]]?.[category]?.questions ?? 0})`,
    ...Object.fromEntries(
      methods.map((m) => [m, metricValue(byCategory[m]?.[category], 'mrr')]),
    ),
  }))
  return (
    <ChartFrame
      title="MRR by question type"
      description="How early the first relevant passage appears, per kind of question (number of questions in brackets)."
      summary="Bar chart of mean reciprocal rank per question type for each method."
    >
      <BarChart data={rows} barGap={2} barCategoryGap="22%" margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="label" {...AXIS} interval={0} />
        <YAxis domain={[0, 1]} ticks={SCORE_TICKS} {...AXIS} axisLine={false} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: 'var(--surface-muted)' }} />
        {bars(methods)}
      </BarChart>
    </ChartFrame>
  )
}
