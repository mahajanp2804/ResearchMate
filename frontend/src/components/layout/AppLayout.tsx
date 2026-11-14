import { useEffect, useState } from 'react'
import { Outlet, useLocation } from 'react-router'
import { findNavItem } from '../../navigation'
import { Header } from './Header'
import { Sidebar } from './Sidebar'

export function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { pathname } = useLocation()

  useEffect(() => {
    const item = findNavItem(pathname)
    document.title = item ? `${item.label} · ResearchMate` : 'ResearchMate'
  }, [pathname])

  return (
    <div className="min-h-screen">
      {/* First stop for keyboard users: jumps past the navigation. */}
      <a
        href="#main"
        className="sr-only z-50 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-accent-fg focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        Skip to content
      </a>
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="lg:pl-64">
        <Header onMenuClick={() => setSidebarOpen(true)} />
        <main
          id="main"
          tabIndex={-1}
          className="mx-auto w-full max-w-7xl px-4 py-6 focus:outline-none sm:px-8 sm:py-8"
        >
          <Outlet />
        </main>
      </div>
    </div>
  )
}
