import { Compass } from 'lucide-react'
import { ButtonLink } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { EmptyState } from '../components/ui/EmptyState'

export default function NotFound() {
  return (
    <Card>
      <EmptyState
        icon={Compass}
        title="Page not found"
        description="The page you were looking for doesn't exist."
        action={<ButtonLink to="/">Back to dashboard</ButtonLink>}
      />
    </Card>
  )
}
