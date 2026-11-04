import { CircleCheck, KeyRound, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { ThemeToggle } from '../components/layout/ThemeToggle'
import { Badge } from '../components/ui/Badge'
import { Card, CardHeader } from '../components/ui/Card'
import { ErrorState, Skeleton } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'

function SettingRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 px-5 py-3 sm:flex-row sm:items-center sm:justify-between">
      <dt className="text-sm text-fg-muted">{label}</dt>
      <dd className="font-mono text-xs break-all text-fg sm:text-right">{value}</dd>
    </div>
  )
}

function SettingsSkeleton() {
  return (
    <div className="space-y-3 p-5">
      {[0, 1, 2, 3].map((row) => (
        <Skeleton key={row} className="h-6 w-full" />
      ))}
    </div>
  )
}

export default function Settings() {
  const { data: info, error, loading, reload } = useApi(api.systemInfo)

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader title="Appearance" description="Choose how ResearchMate looks on this device" />
        <div className="px-5 py-4">
          <ThemeToggle showLabels />
        </div>
      </Card>

      {error && (
        <ErrorState title="Couldn't load configuration" message={error.message} onRetry={reload} />
      )}

      {info && !info.llm.configured && (
        <div className="flex items-start gap-3 rounded-xl border border-warning/30 bg-warning-soft px-4 py-3">
          <KeyRound className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
          <div className="text-sm">
            <p className="font-medium text-fg">Gemini API key not configured</p>
            <p className="mt-0.5 text-fg-muted">
              Paper uploads and search work without it, but answers, summaries and comparisons
              need an LLM. Add <code className="font-mono text-xs">GEMINI_API_KEY</code> to{' '}
              <code className="font-mono text-xs">backend/.env</code> and restart the backend.
            </p>
          </div>
        </div>
      )}

      <p className="text-xs text-fg-subtle">
        These values come from the backend configuration (environment variables or{' '}
        <code className="font-mono">backend/.env</code>) and are read-only here.
      </p>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Models" description="Embedding, reranking and generation" />
          {loading && !info ? (
            <SettingsSkeleton />
          ) : (
            info && (
              <dl className="divide-y divide-border">
                <SettingRow label="Embedding model" value={info.embedding_model_name} />
                <SettingRow
                  label="Reranker"
                  value={
                    info.retrieval.reranker_enabled ? info.retrieval.reranker_model_name : 'Disabled'
                  }
                />
                <SettingRow label="LLM provider" value={info.llm.provider} />
                <SettingRow label="LLM model" value={info.llm.model} />
                <SettingRow
                  label="LLM status"
                  value={
                    info.llm.configured ? (
                      <Badge tone="success">
                        <CircleCheck className="size-3" aria-hidden /> API key configured
                      </Badge>
                    ) : (
                      <Badge tone="warning">
                        <TriangleAlert className="size-3" aria-hidden /> Not configured
                      </Badge>
                    )
                  }
                />
                <SettingRow label="Temperature" value={info.llm.temperature} />
                <SettingRow label="Max output tokens" value={info.llm.max_output_tokens} />
              </dl>
            )
          )}
        </Card>

        <Card>
          <CardHeader title="Retrieval pipeline" description="Hybrid search and context selection" />
          {loading && !info ? (
            <SettingsSkeleton />
          ) : (
            info && (
              <dl className="divide-y divide-border">
                <SettingRow label="Dense candidates (FAISS top-k)" value={info.retrieval.dense_top_k} />
                <SettingRow label="Keyword candidates (BM25 top-k)" value={info.retrieval.bm25_top_k} />
                <SettingRow label="RRF constant (k)" value={info.retrieval.rrf_k} />
                <SettingRow label="Candidates reranked" value={info.retrieval.rerank_candidates} />
                <SettingRow label="Chunks sent to the LLM" value={info.retrieval.final_context_chunks} />
                <SettingRow label="Context token budget" value={info.retrieval.max_context_tokens} />
              </dl>
            )
          )}
        </Card>

        <Card>
          <CardHeader title="Document processing" description="Chunking and upload limits" />
          {loading && !info ? (
            <SettingsSkeleton />
          ) : (
            info && (
              <dl className="divide-y divide-border">
                <SettingRow label="Chunk size" value={`${info.chunking.chunk_size_tokens} tokens`} />
                <SettingRow
                  label="Chunk overlap"
                  value={`${info.chunking.chunk_overlap_tokens} tokens`}
                />
                <SettingRow label="Max file size" value={`${info.max_upload_size_mb} MB`} />
                <SettingRow label="Max files per upload" value={info.max_files_per_upload} />
              </dl>
            )
          )}
        </Card>

        <Card>
          <CardHeader title="About" />
          {loading && !info ? (
            <SettingsSkeleton />
          ) : (
            info && (
              <dl className="divide-y divide-border">
                <SettingRow label="Application" value={info.app_name} />
                <SettingRow label="Version" value={info.version} />
                <SettingRow label="Environment" value={info.environment} />
              </dl>
            )
          )}
        </Card>
      </div>
    </div>
  )
}
