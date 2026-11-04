import type { LucideIcon } from 'lucide-react'
import { Badge } from './ui/Badge'
import { Card } from './ui/Card'
import { EmptyState } from './ui/EmptyState'

interface ComingSoonProps {
  icon: LucideIcon
  title: string
  description: string
  phase: number
  capabilities: string[]
}

/** Honest placeholder for a workspace whose backend is not built yet. */
export function ComingSoon({ icon, title, description, phase, capabilities }: ComingSoonProps) {
  return (
    <Card>
      <EmptyState
        icon={icon}
        title={title}
        description={description}
        action={<Badge tone="accent">Planned for roadmap phase {phase}</Badge>}
      />
      <div className="border-t border-border px-6 py-5">
        <p className="mb-3 text-xs font-medium tracking-wide text-fg-subtle uppercase">
          What this workspace will do
        </p>
        <ul className="grid gap-2 sm:grid-cols-2">
          {capabilities.map((capability) => (
            <li key={capability} className="flex gap-2 text-sm text-fg-muted">
              <span className="mt-2 size-1.5 shrink-0 rounded-full bg-accent" aria-hidden />
              {capability}
            </li>
          ))}
        </ul>
      </div>
    </Card>
  )
}
