import { MessageSquarePlus, Trash2 } from 'lucide-react'
import type { ConversationSummary } from '../../types/api'
import { cn, formatRelativeTime } from '../../utils/format'
import { Button } from '../ui/Button'
import { Skeleton } from '../ui/Feedback'

interface ConversationListProps {
  conversations: ConversationSummary[] | null
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (conversation: ConversationSummary) => void
}

export function ConversationList({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
}: ConversationListProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="p-3">
        <Button variant="secondary" className="w-full" onClick={onNew}>
          <MessageSquarePlus className="size-4" aria-hidden />
          New conversation
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {conversations === null ? (
          <div className="space-y-2 px-1">
            {[0, 1, 2].map((row) => (
              <Skeleton key={row} className="h-11 w-full" />
            ))}
          </div>
        ) : conversations.length === 0 ? (
          <p className="px-2 py-4 text-xs text-fg-subtle">Your conversations will appear here.</p>
        ) : (
          <ul className="space-y-0.5" aria-label="Conversations">
            {conversations.map((conversation) => (
              <li key={conversation.id} className="group relative">
                <button
                  type="button"
                  onClick={() => onSelect(conversation.id)}
                  aria-current={conversation.id === activeId ? 'true' : undefined}
                  className={cn(
                    'w-full rounded-lg px-3 py-2 pr-9 text-left transition-colors',
                    conversation.id === activeId
                      ? 'bg-accent-soft text-accent-text'
                      : 'text-fg-muted hover:bg-surface-muted hover:text-fg',
                  )}
                >
                  <span className="block truncate text-sm">{conversation.title}</span>
                  <span className="block text-xs text-fg-subtle">
                    {formatRelativeTime(conversation.updated_at)}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(conversation)}
                  aria-label={`Delete conversation “${conversation.title}”`}
                  className="absolute top-1/2 right-2 -translate-y-1/2 rounded-md p-1 text-fg-subtle opacity-0 group-hover:opacity-100 hover:bg-danger-soft hover:text-danger focus:opacity-100"
                >
                  <Trash2 className="size-3.5" aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
