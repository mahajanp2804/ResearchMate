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
    /** Evidence threshold on cross-encoder scores; null when off. */
    min_rerank_score: number | null
    final_context_chunks: number
    max_context_tokens: number
    /** Guaranteed context slots per paper for questions about a few selected papers. */
    min_passages_per_paper: number
  }
  /** Paper profiles (behind summaries and comparisons) and gap analysis. */
  analysis: {
    profile_passages_per_aspect: number
    profile_max_context_tokens: number
    max_compare_papers: number
    gap_passages_per_question: number
    gap_max_tokens_per_paper: number
    max_gap_papers: number
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

// --- Chat --------------------------------------------------------------------

export type CitationStatus = 'cited' | 'partially_cited' | 'uncited'

/** A passage given to the LLM, numbered the way the answer cites it. */
export interface ChatSource {
  source_id: string
  chunk_id: number
  document_id: string
  document_title: string
  page_start: number
  page_end: number
  section: string | null
  text: string
  score: number
  /** Whether the answer cites this source; others were retrieved but unused. */
  cited: boolean
  /** For cited sources: the sentence(s) supporting the answer, and their exact page. */
  excerpt: string | null
  excerpt_page: number | null
  /** Share of the citing statements' key terms found in the source (0–1). */
  support: number | null
}

/**
 * How an answer was produced. `no_evidence`: refused without calling the LLM;
 * `model_declined`: the LLM judged the sources insufficient.
 */
export interface AnswerDetails {
  outcome: 'answered' | 'no_evidence' | 'model_declined'
  standalone_question: string
  model: string | null
  context_tokens: number
  truncated: boolean
  timings_ms: Record<string, number>
  searched_document_count: number
  skipped_document_ids: string[]
  /** null for refusals (and for answers saved before citations were checked). */
  citation_status: CitationStatus | null
  /** Ids the model cited that were not among the sources; removed from the answer. */
  invalid_citation_ids: string[]
  /** Claims in the answer that cite no source. */
  uncited_statements: string[]
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
  sources: ChatSource[]
  details: AnswerDetails | null
}

export interface ConversationSummary {
  id: string
  title: string
  document_ids: string[] | null
  message_count: number
  created_at: string
  updated_at: string
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[]
}

export interface ConversationListResponse {
  items: ConversationSummary[]
  total: number
}

export interface ChatResponse {
  conversation: ConversationSummary
  user_message: ChatMessage
  assistant_message: ChatMessage
}

// --- Summaries and comparisons -----------------------------------------------

/** A source an aspect cites, with the sentence that supports it. */
export interface ProfileCitation {
  source_id: string
  excerpt: string
  /** The page the excerpt is on (exact even when the passage spans pages). */
  page: number
  support: number
}

/** A passage of the paper that a summary or comparison was built from. */
export interface ProfileSource {
  source_id: string
  chunk_id: number
  page_start: number
  page_end: number
  section: string | null
  text: string
}

/** A summary section or a comparison cell. */
export interface ProfileAspect {
  key: string
  label: string
  /** `not_found`: the paper's retrieved passages don't state this; `text` is null. */
  status: 'found' | 'not_found'
  text: string | null
  citations: ProfileCitation[]
  citation_status: CitationStatus | null
  /** Ids the model cited that were not among the sources; removed from the text. */
  invalid_citation_ids: string[]
  /** Claims in the text that cite no source. */
  uncited_statements: string[]
}

/** One paper's aspects: its summary sections, or its column of a comparison. */
export interface PaperProfile {
  document_id: string
  document_title: string
  aspects: ProfileAspect[]
  sources: ProfileSource[]
  model: string
  generated_at: string
  /** True when it was saved earlier and no LLM call was made. */
  cached: boolean
  truncated: boolean
}

export interface ComparisonResponse {
  /** The rows, in order. */
  aspects: { key: string; label: string }[]
  /** One column per paper, in the requested order. */
  papers: PaperProfile[]
}

// --- Research gaps -------------------------------------------------------------

export interface GapPaper {
  document_id: string
  title: string
  /** P1, P2, ...: how findings and passages refer to the paper. */
  label: string
}

/** A passage the analysis was given, numbered across all papers. */
export interface GapSource extends ProfileSource {
  document_id: string
}

export interface GapFinding {
  /** Short label; null for stated limitations. */
  title: string | null
  /** The finding (for an opportunity, its evidence) with [S#] citations. */
  text: string
  /** The papers it cites. */
  document_ids: string[]
  citations: ProfileCitation[]
  citation_status: CitationStatus
  uncited_statements: string[]
  /** Removed: ids that were not among the sources. */
  invalid_citation_ids: string[]
  /** Removed from a stated limitation: ids of another paper's passages. */
  misattributed_citation_ids: string[]
  /** Its wording claims more than the selected papers can show. */
  beyond_scope: boolean
}

export interface GapOpportunity extends GapFinding {
  /** The inferred direction; not stated by the papers. */
  suggestion: string
}

export interface LeftOutFinding {
  kind: string
  title: string
  reason: 'no_valid_citation' | 'cites_other_paper' | 'cites_one_paper'
  explanation: string
}

export interface GapAnalysis {
  papers: GapPaper[]
  scope_note: string
  /** Stated by the papers about their own work. */
  stated_limitations: GapFinding[]
  /** Observed across the selected papers. */
  recurring_challenges: GapFinding[]
  differences: GapFinding[]
  limited_discussion: GapFinding[]
  /** Inferred from the findings above. */
  opportunities: GapOpportunity[]
  /** Findings that failed the evidence checks. */
  left_out: LeftOutFinding[]
  malformed_items: number
  sources: GapSource[]
  model: string
  generated_at: string
  cached: boolean
  truncated: boolean
}
