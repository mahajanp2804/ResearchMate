import type {
  ApiErrorBody,
  ChatResponse,
  ChunkListResponse,
  ComparisonResponse,
  ContextResponse,
  ConversationDetail,
  ConversationListResponse,
  DashboardStats,
  DocumentDetail,
  DocumentListResponse,
  HealthResponse,
  PaperProfile,
  ReindexResponse,
  SearchRequest,
  SearchResponse,
  SystemInfo,
  UploadResponse,
} from '../types/api'

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

function toApiError(status: number, body: unknown): ApiError {
  if (isErrorBody(body)) {
    return new ApiError(status, body.error.code, body.error.message, body.error.details)
  }
  if (PROXY_FAILURE_STATUSES.has(status)) return unreachableError()
  return new ApiError(status, 'http_error', `Request failed with status ${status}`)
}

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

  // 204 responses have no body; json() rejects and we return null.
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw toApiError(response.status, body)
  return body as T
}

/**
 * Upload one PDF. Uses XMLHttpRequest because fetch cannot report upload
 * progress. `onProgress` receives the fraction of bytes sent (0–1).
 */
function uploadDocument(file: File, onProgress?: (fraction: number) => void): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('files', file, file.name)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE_URL}/api/documents/upload`)
    xhr.responseType = 'json'
    xhr.setRequestHeader('Accept', 'application/json')
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as UploadResponse)
      else reject(toApiError(xhr.status, xhr.response))
    }
    xhr.onerror = () => reject(unreachableError())
    xhr.send(form)
  })
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>('/api/health', { signal }),
  systemInfo: (signal?: AbortSignal) => request<SystemInfo>('/api/system/info', { signal }),
  dashboardStats: (signal?: AbortSignal) =>
    request<DashboardStats>('/api/dashboard/stats', { signal }),

  listDocuments: (signal?: AbortSignal) =>
    request<DocumentListResponse>('/api/documents?limit=500', { signal }),
  getDocument: (documentId: string, signal?: AbortSignal) =>
    request<DocumentDetail>(`/api/documents/${encodeURIComponent(documentId)}`, { signal }),
  deleteDocument: (documentId: string) =>
    request<null>(`/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' }),
  /** Bring a paper up to date; `force` rebuilds chunks and vectors even if current. */
  reindexDocument: (documentId: string, force = false) =>
    request<ReindexResponse>(
      `/api/documents/${encodeURIComponent(documentId)}/reindex${force ? '?force=true' : ''}`,
      { method: 'POST' },
    ),
  listChunks: (documentId: string, offset: number, limit: number, signal?: AbortSignal) =>
    request<ChunkListResponse>(
      `/api/documents/${encodeURIComponent(documentId)}/chunks?offset=${offset}&limit=${limit}`,
      { signal },
    ),
  uploadDocument,
  /** Run pipeline stages (dense, bm25, hybrid, hybrid_rerank) and get each one's ranked passages. */
  search: (body: SearchRequest, signal?: AbortSignal) =>
    request<SearchResponse>('/api/retrieval/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    }),
  /** The passages the answer pipeline would send to the LLM for this question. */
  context: (body: { query: string; document_ids?: string[] | null }, signal?: AbortSignal) =>
    request<ContextResponse>('/api/retrieval/context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    }),
  /** Ask a question, continuing `conversation_id` if given; `document_ids` null = all papers. */
  ask: (body: { question: string; conversation_id?: string | null; document_ids?: string[] | null }) =>
    request<ChatResponse>('/api/chat/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  listConversations: (signal?: AbortSignal) =>
    request<ConversationListResponse>('/api/chat/conversations?limit=200', { signal }),
  getConversation: (conversationId: string, signal?: AbortSignal) =>
    request<ConversationDetail>(`/api/chat/conversations/${encodeURIComponent(conversationId)}`, {
      signal,
    }),
  deleteConversation: (conversationId: string) =>
    request<null>(`/api/chat/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'DELETE',
    }),
  /** The saved summary of a paper; 404 (`summary_not_found`) if it has none yet. */
  getSummary: (documentId: string, signal?: AbortSignal) =>
    request<PaperProfile>(`/api/research/summaries/${encodeURIComponent(documentId)}`, { signal }),
  /** Generate (or return the saved) nine-section summary; `refresh` regenerates it. */
  summarize: (documentId: string, refresh = false) =>
    request<PaperProfile>(
      `/api/research/summarize/${encodeURIComponent(documentId)}${refresh ? '?refresh=true' : ''}`,
      { method: 'POST' },
    ),
  /** Compare papers aspect by aspect, in the given column order. */
  compare: (documentIds: string[], refresh = false) =>
    request<ComparisonResponse>('/api/research/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document_ids: documentIds, refresh }),
    }),
  /** URL of the original PDF, optionally opened at a 1-based page. */
  documentFileUrl: (documentId: string, page?: number) =>
    `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/file${page ? `#page=${page}` : ''}`,
}
