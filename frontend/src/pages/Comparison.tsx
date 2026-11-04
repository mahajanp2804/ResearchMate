import { GitCompareArrows } from 'lucide-react'
import { ComingSoon } from '../components/ComingSoon'

export default function Comparison() {
  return (
    <ComingSoon
      icon={GitCompareArrows}
      phase={9}
      title="Paper comparison"
      description="Pick two or more papers and see them side by side, with every cell traced back to its source."
      capabilities={[
        'Objectives, problem statements and methods',
        'Algorithms, datasets and evaluation metrics',
        'Results, strengths and limitations',
        'Future directions from each paper',
        '"Not reported" instead of guessed values',
        'Follow-up questions about the comparison',
      ]}
    />
  )
}
