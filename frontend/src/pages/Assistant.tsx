import { MessageSquareText } from 'lucide-react'
import { ComingSoon } from '../components/ComingSoon'

export default function Assistant() {
  return (
    <ComingSoon
      icon={MessageSquareText}
      phase={7}
      title="Research assistant"
      description="Ask questions about one paper, a selection, or your whole library. Answers come only from your papers and cite the pages they rely on."
      capabilities={[
        'Hybrid semantic + keyword retrieval',
        'Cross-encoder reranking of evidence',
        'Citations with document, page and excerpt',
        'Follow-up questions with conversation history',
        'Restrict a conversation to selected papers',
        'Says so plainly when the papers lack evidence',
      ]}
    />
  )
}
