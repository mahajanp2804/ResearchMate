import { useCallback, useState } from 'react'
import { ExperimentSetup } from '../components/evaluation/ExperimentSetup'
import { RunHistory } from '../components/evaluation/RunHistory'
import { RunResults } from '../components/evaluation/RunResults'
import { RetrievalPlayground } from '../components/RetrievalPlayground'
import { ErrorState, Skeleton } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'
import type { DatasetListResponse, EvaluationRunList } from '../types/api'

const POLL_MS = 2000

const papersIndexing = (list: DatasetListResponse) =>
  list.items.some((dataset) =>
    dataset.papers.some((paper) => paper.indexing_status === 'indexing' || paper.indexing_status === 'not_indexed'),
  )
const runActive = (list: EvaluationRunList) =>
  list.items.some((run) => run.status === 'pending' || run.status === 'running')

export default function Evaluation() {
  const datasets = useApi(api.evaluationDatasets, { pollMs: POLL_MS, pollWhile: papersIndexing })
  const runs = useApi(api.evaluationRuns, { pollMs: POLL_MS, pollWhile: runActive })
  const { data: health } = useApi(api.health)
  const [chosenRunId, setChosenRunId] = useState<string | null>(null)

  const items = runs.data?.items ?? []
  // Show the chosen run, or else the latest one.
  const selectedId = items.some((run) => run.id === chosenRunId) ? chosenRunId : (items[0]?.id ?? null)
  const reloadRuns = runs.reload
  const onFinished = useCallback(() => reloadRuns(), [reloadRuns])
  const dataset = datasets.data?.items[0]

  return (
    <div className="space-y-6">
      {datasets.error ? (
        <ErrorState
          title="Couldn't load the evaluation datasets"
          message={datasets.error.message}
          onRetry={datasets.reload}
        />
      ) : !dataset ? (
        <Skeleton className="h-72 w-full rounded-xl" />
      ) : (
        <ExperimentSetup
          dataset={dataset}
          llmConfigured={health?.llm_configured ?? false}
          busy={runs.data ? runActive(runs.data) : false}
          onInstalled={datasets.reload}
          onStarted={(run) => {
            setChosenRunId(run.id)
            runs.reload()
          }}
        />
      )}

      {selectedId && <RunResults key={selectedId} runId={selectedId} onFinished={onFinished} />}

      {items.length > 1 && (
        <RunHistory
          runs={items}
          selectedId={selectedId}
          onSelect={setChosenRunId}
          onDeleted={(runId) => {
            if (runId === chosenRunId) setChosenRunId(null)
            runs.reload()
          }}
        />
      )}

      <RetrievalPlayground />
    </div>
  )
}
