import { Bot, MessageSquareText } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'
import { ApiError, api } from '../../services/api'
import type { ChatMessage } from '../../types/api'
import { Composer } from '../chat/Composer'
import { MessageBubble } from '../chat/MessageBubble'
import { ErrorState, Spinner } from '../ui/Feedback'

const SUGGESTIONS = [
  'What are the main differences in methodology?',
  'Compare the experimental results of these papers.',
  'Which limitations do these papers share?',
]

/**
 * Questions about the compared papers. Answers come from the normal grounded
 * pipeline limited to these papers, which gives every paper a share of the
 * passages, and are saved as a conversation in the Research Assistant.
 */
export function FollowUpChat({ documentIds }: { documentIds: string[] }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [draft, setDraft] = useState('')

  const ask = async (text: string) => {
    const question = text.trim()
    if (!question || pending) return
    setPending(question)
    setDraft('')
    setError(null)
    try {
      const response = await api.ask({
        question,
        conversation_id: conversationId,
        document_ids: documentIds,
      })
      setConversationId(response.conversation.id)
      setMessages((current) => [...current, response.user_message, response.assistant_message])
    } catch (cause) {
      setError(cause instanceof ApiError ? cause : new ApiError(0, 'unknown', 'Something went wrong.'))
      setDraft(question) // keep the question so it can be retried
    } finally {
      setPending(null)
    }
  }

  return (
    <div className="space-y-4">
      {messages.length === 0 && !pending && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => void ask(suggestion)}
              className="rounded-full border border-border px-3 py-1.5 text-xs text-fg-muted hover:bg-surface-muted hover:text-fg"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
      {messages.length > 0 && (
        <div className="space-y-6">
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              question={message.role === 'assistant' ? messages[index - 1]?.content : undefined}
            />
          ))}
        </div>
      )}
      {pending && (
        <div className="space-y-4">
          <div className="flex justify-end">
            <p className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-sm whitespace-pre-wrap text-accent-fg">
              {pending}
            </p>
          </div>
          <div className="flex items-center gap-3 text-sm text-fg-muted">
            <div className="flex size-8 items-center justify-center rounded-lg bg-accent-soft text-accent-text">
              <Bot className="size-4" aria-hidden />
            </div>
            <Spinner />
            Searching the compared papers…
          </div>
        </div>
      )}
      {error && <ErrorState title="Couldn't get an answer" message={error.message} />}
      <Composer value={draft} onChange={setDraft} onSubmit={() => void ask(draft)} busy={pending !== null}>
        {conversationId && (
          <Link
            to={`/assistant?c=${conversationId}`}
            className="inline-flex items-center gap-1.5 text-xs text-accent-text hover:underline"
          >
            <MessageSquareText className="size-3.5" aria-hidden />
            Continue in Research Assistant
          </Link>
        )}
      </Composer>
    </div>
  )
}
