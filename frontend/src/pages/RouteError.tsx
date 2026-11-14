import { RefreshCw, TriangleAlert } from 'lucide-react'
import { isRouteErrorResponse, useRouteError } from 'react-router'
import { Button, ButtonLink } from '../components/ui/Button'
import { Card } from '../components/ui/Card'

/**
 * Shown instead of a page that failed to render or load (for example a
 * chunk that could not be downloaded), so one broken page never blanks the
 * whole app. The error itself is logged to the console by the router.
 */
export default function RouteError() {
  const error = useRouteError()
  const detail = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? error.message
      : null

  return (
    <Card className="mx-auto max-w-xl">
      <div role="alert" className="flex flex-col items-center px-6 py-12 text-center">
        <div className="mb-4 flex size-11 items-center justify-center rounded-xl bg-danger-soft text-danger">
          <TriangleAlert className="size-5" aria-hidden />
        </div>
        <h2 className="text-sm font-semibold text-fg">This page ran into a problem</h2>
        <p className="mt-1 max-w-sm text-sm text-fg-muted">
          Reloading usually fixes it. Your papers, conversations and results are saved on the
          server and are not affected.
        </p>
        {detail && (
          <p className="mt-3 max-w-sm rounded-md bg-surface-muted px-3 py-1.5 font-mono text-xs break-words text-fg-subtle">
            {detail}
          </p>
        )}
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <Button onClick={() => window.location.reload()}>
            <RefreshCw className="size-4" aria-hidden />
            Reload the page
          </Button>
          <ButtonLink to="/" variant="secondary">
            Back to dashboard
          </ButtonLink>
        </div>
      </div>
    </Card>
  )
}
