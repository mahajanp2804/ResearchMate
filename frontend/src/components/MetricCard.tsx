import type { LucideIcon } from 'lucide-react'
import { formatNumber } from '../utils/format'
import { Card } from './ui/Card'
import { Skeleton } from './ui/Feedback'

interface MetricCardProps {
  label: string
  /** `null` while loading or when the value could not be loaded. */
  value: number | null
  loading?: boolean
  icon: LucideIcon
  hint?: string
}

export function MetricCard({ label, value, loading = false, icon: Icon, hint }: MetricCardProps) {
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-fg-muted">{label}</p>
        <Icon className="size-4 text-fg-subtle" aria-hidden />
      </div>
      {value === null && loading ? (
        <Skeleton className="mt-3 h-8 w-16" />
      ) : (
        <p className="mt-2 text-3xl font-semibold tracking-tight text-fg tabular-nums">
          {value === null ? '—' : formatNumber(value)}
        </p>
      )}
      {hint && <p className="mt-1 text-xs text-fg-subtle">{hint}</p>}
    </Card>
  )
}
