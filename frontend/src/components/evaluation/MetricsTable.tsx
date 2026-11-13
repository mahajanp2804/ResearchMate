import type { EvaluationRunSummary } from '../../types/api'
import { cn } from '../../utils/format'
import { METHOD_LABELS, formatScore, methodColor, metricValue } from '../../utils/evaluation'

/** Every retrieval number of a run, with the best value per column in bold. */
export function MetricsTable({ run }: { run: EvaluationRunSummary }) {
  const methods = run.config.methods
  const retrieval = run.metrics.retrieval ?? {}
  const columns = [
    ...run.config.cutoffs.map((k) => ({ key: `precision@${k}`, label: `P@${k}` })),
    ...run.config.cutoffs.map((k) => ({ key: `recall@${k}`, label: `R@${k}` })),
    { key: 'mrr', label: 'MRR' },
  ]
  const best = Object.fromEntries(
    columns.map(({ key }) => [
      key,
      Math.max(...methods.map((method) => metricValue(retrieval[method], key) ?? -1)),
    ]),
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">Retrieval metrics per method</caption>
        <thead>
          <tr className="border-b border-border text-xs text-fg-muted">
            <th scope="col" className="px-3 py-2 text-left font-medium">
              Method
            </th>
            {columns.map((column) => (
              <th key={column.key} scope="col" className="px-3 py-2 text-right font-medium">
                {column.label}
              </th>
            ))}
            <th scope="col" className="px-3 py-2 text-right font-medium" title="Mean search time per question">
              ms / query
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {methods.map((method) => (
            <tr key={method}>
              <th scope="row" className="px-3 py-2 text-left font-medium whitespace-nowrap text-fg">
                <span className="flex items-center gap-2">
                  <span
                    className="size-2.5 rounded-sm"
                    style={{ background: methodColor(method) }}
                    aria-hidden
                  />
                  {METHOD_LABELS[method]}
                </span>
              </th>
              {columns.map((column) => {
                const value = metricValue(retrieval[method], column.key)
                return (
                  <td
                    key={column.key}
                    className={cn(
                      'px-3 py-2 text-right text-fg-muted tabular-nums',
                      value !== null && value === best[column.key] && 'font-semibold text-fg',
                    )}
                  >
                    {formatScore(value)}
                  </td>
                )
              })}
              <td className="px-3 py-2 text-right text-fg-muted tabular-nums">
                {retrieval[method]?.mean_latency_ms?.toFixed(1) ?? '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
