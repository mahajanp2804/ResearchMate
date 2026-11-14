import { createBrowserRouter, RouterProvider } from 'react-router'
import { AppLayout } from './components/layout/AppLayout'
import { ThemeProvider } from './context/ThemeContext'
import Assistant from './pages/Assistant'
import Comparison from './pages/Comparison'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Insights from './pages/Insights'
import NotFound from './pages/NotFound'
import PaperSummary from './pages/PaperSummary'
import RouteError from './pages/RouteError'
import Settings from './pages/Settings'

const router = createBrowserRouter([
  {
    element: <AppLayout />,
    // Last resort if the layout itself fails.
    errorElement: <RouteError />,
    children: [
      {
        // A page that fails shows the error inside the layout, so navigation still works.
        errorElement: <RouteError />,
        children: [
          { index: true, element: <Dashboard /> },
          { path: 'papers', element: <Documents /> },
          { path: 'papers/:documentId/summary', element: <PaperSummary /> },
          { path: 'assistant', element: <Assistant /> },
          { path: 'comparison', element: <Comparison /> },
          { path: 'insights', element: <Insights /> },
          {
            path: 'evaluation',
            // Loaded on first visit: it brings the charting library, which no other page needs.
            lazy: async () => ({ Component: (await import('./pages/Evaluation')).default }),
          },
          { path: 'settings', element: <Settings /> },
          { path: '*', element: <NotFound /> },
        ],
      },
    ],
  },
])

export default function App() {
  return (
    <ThemeProvider>
      <RouterProvider router={router} />
    </ThemeProvider>
  )
}
