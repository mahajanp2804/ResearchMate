import type { ApiErrorBody, DashboardStats, HealthResponse, SystemInfo } from '../types/api'

// Empty by default: in development Vite proxies /api to FastAPI, and in Docker
// the web server does the same. Set VITE_API_BASE_URL only for a remote API.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly details: Record<string, unknown>

  constructor(status: number, code: string, message: string, details: Record<string, unknown> = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }

  get isNetworkError(): boolean {
    return this.status === 0
  }
}

function isErrorBody(value: unknown): value is ApiErrorBody {
  return (
    typeof value === 'object' &&
    value !== null &&
    'error' in value &&
    typeof (value as ApiErrorBody).error?.message === 'string'
  )
}

const unreachableError = () =>
  new ApiError(
    0,
    'network_error',
    'Cannot reach the ResearchMate API. Make sure the backend is running on port 8000.',
  )

// Gateway statuses without our JSON error envelope come from a proxy (Vite in
// development, the web server in Docker) that could not reach FastAPI.
const PROXY_FAILURE_STATUSES = new Set([502, 503, 504])

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw unreachableError()
  }

  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    if (isErrorBody(body)) {
      throw new ApiError(response.status, body.error.code, body.error.message, body.error.details)
    }
    if (PROXY_FAILURE_STATUSES.has(response.status)) throw unreachableError()
    throw new ApiError(response.status, 'http_error', `Request failed with status ${response.status}`)
  }
  return body as T
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>('/api/health', { signal }),
  systemInfo: (signal?: AbortSignal) => request<SystemInfo>('/api/system/info', { signal }),
  dashboardStats: (signal?: AbortSignal) =>
    request<DashboardStats>('/api/dashboard/stats', { signal }),
}
