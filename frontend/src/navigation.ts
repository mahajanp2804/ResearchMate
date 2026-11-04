import {
  FileText,
  FlaskConical,
  GitCompareArrows,
  LayoutDashboard,
  Lightbulb,
  MessageSquareText,
  Settings,
  type LucideIcon,
} from 'lucide-react'

export interface NavItem {
  path: string
  label: string
  description: string
  icon: LucideIcon
}

// Single source for the sidebar, page headers and document titles.
export const NAV_ITEMS: NavItem[] = [
  {
    path: '/',
    label: 'Dashboard',
    description: 'Overview of your research workspace',
    icon: LayoutDashboard,
  },
  {
    path: '/papers',
    label: 'My Research Papers',
    description: 'Upload, inspect and manage your paper library',
    icon: FileText,
  },
  {
    path: '/assistant',
    label: 'Research Assistant',
    description: 'Ask questions and get answers backed by citations',
    icon: MessageSquareText,
  },
  {
    path: '/comparison',
    label: 'Paper Comparison',
    description: 'Compare methods, datasets and results side by side',
    icon: GitCompareArrows,
  },
  {
    path: '/insights',
    label: 'Research Insights',
    description: 'Limitations, recurring patterns and potential research gaps',
    icon: Lightbulb,
  },
  {
    path: '/evaluation',
    label: 'RAG Evaluation',
    description: 'Measure retrieval quality across pipeline configurations',
    icon: FlaskConical,
  },
  {
    path: '/settings',
    label: 'Settings',
    description: 'Models, retrieval parameters and appearance',
    icon: Settings,
  },
]

export function findNavItem(pathname: string): NavItem | undefined {
  return NAV_ITEMS.find((item) =>
    item.path === '/' ? pathname === '/' : pathname.startsWith(item.path),
  )
}
