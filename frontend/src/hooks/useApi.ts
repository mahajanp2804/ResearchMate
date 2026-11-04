import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ApiError } from '../services/api'

interface ApiState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

/**
 * Runs an API call on mount (and whenever `reload` is called), aborting the
 * request if the component unmounts. With `pollMs`, the call repeats quietly
 * in the background without flipping `loading` back on.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  options: { pollMs?: number } = {},
): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [reloadToken, setReloadToken] = useState(0)
  // Keep the latest fetcher without re-running the effect when callers pass
  // an inline function.
  const fetcherRef = useRef(fetcher)
  useLayoutEffect(() => {
    fetcherRef.current = fetcher
  })

  useEffect(() => {
    const controller = new AbortController()

    const run = async () => {
      try {
        const result = await fetcherRef.current(controller.signal)
        setData(result)
        setError(null)
      } catch (cause) {
        if (controller.signal.aborted) return
        setError(
          cause instanceof ApiError
            ? cause
            : new ApiError(0, 'unknown_error', 'Something went wrong while loading data.'),
        )
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    void run()
    const timer = options.pollMs ? window.setInterval(run, options.pollMs) : undefined

    return () => {
      controller.abort()
      if (timer) window.clearInterval(timer)
    }
  }, [reloadToken, options.pollMs])

  const reload = useCallback(() => {
    setLoading(true)
    setReloadToken((token) => token + 1)
  }, [])

  return { data, error, loading, reload }
}
