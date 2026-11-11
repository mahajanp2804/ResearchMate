import { createBrowserRouter, RouterProvider } from 'react-router'
import { AppLayout } from './components/layout/AppLayout'
import { ThemeProvider } from './context/ThemeContext'
import Assistant from './pages/Assistant'
import Comparison from './pages/Comparison'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Evaluation from './pages/Evaluation'
import Insights from './pages/Insights'
import NotFound from './pages/NotFound'
import PaperSummary from './pages/PaperSummary'
import Settings from './pages/Settings'

const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'papers', element: <Documents /> },
      { path: 'papers/:documentId/summary', element: <PaperSummary /> },
      { path: 'assistant', element: <Assistant /> },
      { path: 'comparison', element: <Comparison /> },
      { path: 'insights', element: <Insights /> },
      { path: 'evaluation', element: <Evaluation /> },
      { path: 'settings', element: <Settings /> },
      { path: '*', element: <NotFound /> },
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
