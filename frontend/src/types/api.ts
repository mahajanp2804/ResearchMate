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
  max_upload_size_mb: number
  max_files_per_upload: number
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

export interface ApiErrorBody {
  error: {
    code: string
    message: string
    details?: Record<string, unknown>
  }
}
