// Mirrors backend/app/schemas. Keep in sync when response models change.

export type ProcessingStatus = 'pending' | 'processing' | 'completed' | 'failed'
export type IndexingStatus = 'not_indexed' | 'indexing' | 'indexed' | 'failed'

export type ActivityType =
  | 'document_uploaded'
  | 'document_deleted'
  | 'document_reindexed'
  | 'question_asked'
  | 'summary_generated'
  | 'comparison_run'
  | 'gap_analysis_run'
  | 'evaluation_run'

export interface HealthResponse {
  status: 'ok' | 'degraded'
  version: string
  database: 'ok' | 'error'
  llm_configured: boolean
}

export interface SystemInfo {
  app_name: string
  version: string
  environment: string
  embedding_model_name: string
  embedding_device: string
  vector_index: {
    vector_count: number
    /** Model and dimension of the stored vectors; null until a paper is indexed. */
    model_name: string | null
    dimension: number | null
  }
  max_upload_size_mb: number
  max_files_per_upload: number
  max_pages_per_document: number
  chunking: {
    chunk_size_tokens: number
    chunk_overlap_tokens: number
  }
  retrieval: {
    dense_top_k: number
    bm25_top_k: number
    rrf_k: number
    reranker_enabled: boolean
    reranker_model_name: string
    rerank_candidates: number
    final_context_chunks: number
    max_context_tokens: number
  }
  llm: {
    provider: string
    model: string
    configured: boolean
    temperature: number
    max_output_tokens: number
  }
}

export interface RecentDocument {
  id: string
  filename: string
  title: string | null
  page_count: number | null
  processing_status: ProcessingStatus
  indexing_status: IndexingStatus
  created_at: string
}

export interface ActivityItem {
  id: number
  event_type: ActivityType
  description: string
  document_id: string | null
  created_at: string
}

export interface DashboardStats {
  total_papers: number
  indexed_papers: number
  total_chunks: number
  total_questions: number
  total_conversations: number
  recent_documents: RecentDocument[]
  recent_activity: ActivityItem[]
}

/** Metadata read from the PDF; every key is optional. */
export interface ExtractedMetadata {
  author?: string
  subject?: string
  keywords?: string
  creator?: string
  producer?: string
  pdf_version?: string
  pdf_title?: string
  created?: string
  modified?: string
  title_source?: 'metadata' | 'first_page' | 'filename'
}

export interface DocumentSummary {
  id: string
  filename: string
  title: string | null
  page_count: number | null
  file_size_bytes: number
  processing_status: ProcessingStatus
  indexing_status: IndexingStatus
  status_message: string | null
  chunk_count: number
  extracted_metadata: ExtractedMetadata
  upload_date: string
  updated_at: string
}

export interface DocumentDetail extends DocumentSummary {
  file_hash: string
  embedding_model: string | null
  chunking_signature: string | null
  indexed_at: string | null
  pages_without_text: number[]
}

/**
 * What re-indexing did: nothing (`unchanged`), re-embedded existing chunks,
 * rebuilt chunks and vectors, or failed (see `document.status_message`).
 */
export interface ReindexResponse {
  action: 'unchanged' | 'reembedding' | 'rechunked' | 'failed'
  document: DocumentDetail
}

/** A passage the paper was split into; the unit that retrieval returns. */
export interface Chunk {
  id: number
  chunk_index: number
  page_start: number
  page_end: number
  section: string | null
  token_count: number
  char_count: number
  text: string
}

export interface ChunkListResponse {
  items: Chunk[]
  total: number
}

export interface DocumentListResponse {
  items: DocumentSummary[]
  total: number
}

export interface ErrorDetail {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface UploadResult {
  filename: string
  status: 'created' | 'duplicate' | 'rejected'
  document: DocumentSummary | null
  error: ErrorDetail | null
}

export interface UploadResponse {
  results: UploadResult[]
  created: number
  duplicates: number
  rejected: number
}

export interface ApiErrorBody {
  error: {
    code: string
    message: string
    details?: Record<string, unknown>
  }
}

// --- Retrieval ---------------------------------------------------------------

/** dense and bm25 retrieve; hybrid fuses them (RRF); hybrid_rerank rescores with a cross-encoder. */
export type SearchMethod = 'dense' | 'bm25' | 'hybrid' | 'hybrid_rerank'

export interface SearchRequest {
  query: string
  methods?: SearchMethod[]
  top_k?: number
  /** Omit or null to search every searchable paper. */
  document_ids?: string[] | null
}

export interface SearchHit {
  rank: number
  /**
   * Cosine similarity (dense), BM25 score, RRF score (hybrid) or cross-encoder
   * logit (hybrid_rerank). Only comparable within one method's list.
   */
  score: number
  chunk_id: number
  document_id: string
  document_title: string
  chunk_index: number
  page_start: number
  page_end: number
  section: string | null
  text: string
  token_count: number
  /** Rank in each stage that saw this passage, e.g. { dense: 3, bm25: 1, hybrid: 2 }. */
  source_ranks: Partial<Record<'dense' | 'bm25' | 'hybrid', number>>
}

export interface MethodResults {
  method: SearchMethod
  hits: SearchHit[]
  latency_ms: number
}

export interface SearchResponse {
  query: string
  results: MethodResults[]
  searched_document_count: number
  skipped_document_ids: string[]
}

/** The passages the answer pipeline would send to the LLM, as sources S1..Sn. */
export interface ContextResponse {
  query: string
  reranked: boolean
  has_evidence: boolean
  passages: SearchHit[]
  total_tokens: number
  max_tokens: number
  max_passages: number
  /** Evidence threshold on cross-encoder scores; null when disabled. */
  min_score: number | null
  dropped: { low_score: number; duplicates: number; over_budget: number }
  searched_document_count: number
  skipped_document_ids: string[]
  latency_ms: number
}
