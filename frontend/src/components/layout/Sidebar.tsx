import { X } from 'lucide-react'
import { NavLink } from 'react-router'
import { useApi } from '../../hooks/useApi'
import { NAV_ITEMS } from '../../navigation'
import { api } from '../../services/api'
import { cn } from '../../utils/format'

const HEALTH_POLL_MS = 15_000

function BackendStatus() {
  const { data, error, loading } = useApi(api.health, { pollMs: HEALTH_POLL_MS })

  let tone = 'bg-fg-subtle'
  let label = 'Checking API…'
  if (error) {
    tone = 'bg-danger'
    label = 'API offline'
  } else if (data) {
    tone = data.status === 'ok' ? 'bg-success' : 'bg-warning'
    label = data.status === 'ok' ? `API connected · v${data.version}` : 'API degraded'
  }

  return (
    <div className="flex items-center gap-2 px-3 text-xs text-fg-muted" aria-live="polite">
      <span className={cn('size-2 rounded-full', tone, loading && !data && 'animate-pulse')} />
      {label}
    </div>
  )
}

interface SidebarProps {
  open: boolean
  onClose: () => void
}

export function Sidebar({ open, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile backdrop */}
      <div
        className={cn(
          'fixed inset-0 z-30 bg-black/40 transition-opacity lg:hidden',
          open ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
        onClick={onClose}
        aria-hidden
      />
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border bg-surface',
          'transition-transform lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex h-16 items-center justify-between px-5">
          <div className="flex items-center gap-2.5">
            <img src="/favicon.svg" alt="" className="size-7" />
            <span className="text-[15px] font-semibold tracking-tight text-fg">ResearchMate</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-fg-muted hover:bg-surface-muted lg:hidden"
            aria-label="Close navigation"
          >
            <X className="size-5" />
          </button>
        </div>

        <nav className="flex-1 space-y-0.5 px-3 py-2" aria-label="Main">
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              onClick={onClose}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-accent-soft text-accent-text'
                    : 'text-fg-muted hover:bg-surface-muted hover:text-fg',
                )
              }
            >
              <Icon className="size-4 shrink-0" aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-border py-4">
          <BackendStatus />
        </div>
      </aside>
    </>
  )
}
