import { Menu } from 'lucide-react'
import { useLocation } from 'react-router'
import { findNavItem } from '../../navigation'
import { ThemeToggle } from './ThemeToggle'

export function Header({ onMenuClick }: { onMenuClick: () => void }) {
  const { pathname } = useLocation()
  const item = findNavItem(pathname)

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-border bg-bg/85 px-4 backdrop-blur sm:px-8">
      <button
        type="button"
        onClick={onMenuClick}
        className="-ml-1 rounded-md p-1.5 text-fg-muted hover:bg-surface-muted lg:hidden"
        aria-label="Open navigation"
      >
        <Menu className="size-5" />
      </button>
      <div className="min-w-0 flex-1">
        <h1 className="truncate text-base font-semibold tracking-tight text-fg">
          {item?.label ?? 'Not found'}
        </h1>
        {item && <p className="hidden truncate text-xs text-fg-muted sm:block">{item.description}</p>}
      </div>
      <ThemeToggle />
    </header>
  )
}
