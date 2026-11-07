import { FlaskConical } from 'lucide-react'
import { ComingSoon } from '../components/ComingSoon'
import { RetrievalPlayground } from '../components/RetrievalPlayground'

export default function Evaluation() {
  return (
    <div className="space-y-6">
      <RetrievalPlayground />
      <ComingSoon
        icon={FlaskConical}
        phase={11}
        title="Evaluation experiments"
        description="Run a curated question set against the pipeline and measure how well each retrieval configuration finds the right evidence."
        capabilities={[
          'Precision@K, Recall@K and MRR',
          'Dense vs. BM25 vs. hybrid vs. hybrid + rerank',
          'Faithfulness, answer and context relevance (LLM-judged)',
          'Retrieval and answer metrics reported separately',
          'Stored runs you can compare over time',
          'Charts of every experiment',
        ]}
      />
    </div>
  )
}
