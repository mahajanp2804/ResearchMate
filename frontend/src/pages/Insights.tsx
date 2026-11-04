import { Lightbulb } from 'lucide-react'
import { ComingSoon } from '../components/ComingSoon'

export default function Insights() {
  return (
    <ComingSoon
      icon={Lightbulb}
      phase={10}
      title="Research insights"
      description="Evidence-based gap analysis across the papers you select, scoped to your library rather than the whole field."
      capabilities={[
        'Limitations the authors state explicitly',
        'Challenges that recur across papers',
        'Where the papers disagree or differ',
        'Topics with little discussion',
        'Candidate research directions, labelled as inferred',
        'A citation behind every finding',
      ]}
    />
  )
}
