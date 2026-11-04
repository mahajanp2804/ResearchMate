import { CircleCheck, CircleX, Copy, FileText } from 'lucide-react'
import type { UploadItem } from '../hooks/useUploadQueue'
import { cn, formatBytes } from '../utils/format'
import { Spinner } from './ui/Feedback'

const STATE_LABEL: Record<UploadItem['state'], string> = {
  queued: 'Waiting…',
  uploading: 'Uploading…',
  processing: 'Extracting text…',
  created: 'Added',
  duplicate: 'Duplicate',
  rejected: 'Not added',
}

function StateIcon({ state }: { state: UploadItem['state'] }) {
  if (state === 'created') return <CircleCheck className="size-4 text-success" aria-hidden />
  if (state === 'duplicate') return <Copy className="size-4 text-warning" aria-hidden />
  if (state === 'rejected') return <CircleX className="size-4 text-danger" aria-hidden />
  if (state === 'queued') return <FileText className="size-4 text-fg-subtle" aria-hidden />
  return <Spinner className="text-accent" />
}

export function UploadQueue({ items }: { items: UploadItem[] }) {
  return (
    <ul className="divide-y divide-border" aria-live="polite">
      {items.map((item) => (
        <li key={item.id} className="flex gap-3 px-5 py-3">
          <div className="mt-0.5">
            <StateIcon state={item.state} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline justify-between gap-3">
              <p className="truncate text-sm font-medium text-fg">{item.name}</p>
              <span
                className={cn(
                  'shrink-0 text-xs',
                  item.state === 'rejected' ? 'text-danger' : 'text-fg-subtle',
                )}
              >
                {STATE_LABEL[item.state]}
              </span>
            </div>
            {(item.state === 'uploading' || item.state === 'processing') && (
              <div
                className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-muted"
                role="progressbar"
                aria-valuenow={Math.round(item.progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`Uploading ${item.name}`}
              >
                <div
                  className={cn(
                    'h-full rounded-full bg-accent transition-[width]',
                    item.state === 'processing' && 'animate-pulse',
                  )}
                  style={{ width: `${Math.max(item.progress * 100, 4)}%` }}
                />
              </div>
            )}
            <p className="mt-1 text-xs text-fg-muted">{item.message ?? formatBytes(item.size)}</p>
          </div>
        </li>
      ))}
    </ul>
  )
}
