import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ApiError } from '../services/api'

interface ApiState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

interface ApiOptions<T> {
  /** Repeat the call every `pollMs` milliseconds, quietly (no `loading` flicker). */
  pollMs?: number
  /** With `pollMs`, only poll while this returns true for the latest data. */
  pollWhile?: (data: T) => boolean
}

/**
 * Runs an API call on mount (and whenever `reload` is called), aborting the
 * request if the component unmounts.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  options: ApiOptions<T> = {},
): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [reloadToken, setReloadToken] = useState(0)
  // Keep the latest callbacks and data without re-running the effect when
  // callers pass inline functions.
  const fetcherRef = useRef(fetcher)
  const pollWhileRef = useRef(options.pollWhile)
  const dataRef = useRef<T | null>(null)
  useLayoutEffect(() => {
    fetcherRef.current = fetcher
    pollWhileRef.current = options.pollWhile
  })

  useEffect(() => {
    const controller = new AbortController()

    const run = async () => {
      try {
        const result = await fetcherRef.current(controller.signal)
        dataRef.current = result
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

    const poll = () => {
      const shouldPoll = pollWhileRef.current
      if (shouldPoll && (dataRef.current === null || !shouldPoll(dataRef.current))) return
      void run()
    }

    void run()
    const timer = options.pollMs ? window.setInterval(poll, options.pollMs) : undefined

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
