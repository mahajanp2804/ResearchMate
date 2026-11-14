import { Bot, FileText, History, KeyRound, MessageSquareText } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { Composer } from '../components/chat/Composer'
import { ConversationList } from '../components/chat/ConversationList'
import { MessageBubble } from '../components/chat/MessageBubble'
import { DocumentFilter } from '../components/DocumentFilter'
import { ButtonLink } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { EmptyState } from '../components/ui/EmptyState'
import { ErrorState, Spinner } from '../components/ui/Feedback'
import { useApi } from '../hooks/useApi'
import { ApiError, api } from '../services/api'
import type { ChatMessage, ConversationSummary } from '../types/api'
import { cn } from '../utils/format'
import { papersFromParams } from '../utils/links'

const EXAMPLES = [
  'What is the main objective of this research?',
  'Explain the methodology used.',
  'Which datasets were used?',
  'What are the limitations of the proposed approach?',
  'Compare the experimental results of these papers.',
  'What future research directions are mentioned?',
]

interface LoadedConversation {
  id: string
  messages: ChatMessage[]
}

function EmptyChat({ hasPapers, onExample }: { hasPapers: boolean; onExample: (q: string) => void }) {
  if (!hasPapers) {
    return (
      <EmptyState
        icon={FileText}
        title="No searchable papers yet"
        description="Upload papers and wait until they show as Searchable. Answers come only from your papers."
        action={<ButtonLink to="/papers">Go to papers</ButtonLink>}
      />
    )
  }
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center py-10 text-center">
      <div className="mb-4 flex size-11 items-center justify-center rounded-xl bg-accent-soft text-accent-text">
        <MessageSquareText className="size-5" aria-hidden />
      </div>
      <h2 className="text-base font-semibold text-fg">Ask about your papers</h2>
      <p className="mt-1 text-sm text-fg-muted">
        Answers use only passages retrieved from your papers and cite them as [S1], [S2], … If
        the papers don't cover a question, the assistant says so instead of guessing.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onExample(example)}
            className="rounded-full border border-border px-3 py-1.5 text-xs text-fg-muted hover:bg-surface-muted hover:text-fg"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function Assistant() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeId = searchParams.get('c')

  const conversations = useApi(api.listConversations)
  const documents = useApi(api.listDocuments)
  const { data: health } = useApi(api.health)
  const searchable = (documents.data?.items ?? []).filter((doc) => doc.indexing_status === 'indexed')

  const [loaded, setLoaded] = useState<LoadedConversation | null>(null)
  const [loadError, setLoadError] = useState<ApiError | null>(null)
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null)
  const [askError, setAskError] = useState<ApiError | null>(null)
  const [draft, setDraft] = useState('')
  // A link can open the assistant scoped to papers (?papers=a,b), e.g. "Ask about this paper".
  const [selected, setSelected] = useState<string[] | null>(() => {
    const ids = papersFromParams(searchParams)
    return ids.length > 0 ? ids : null
  })
  const [historyOpen, setHistoryOpen] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<ConversationSummary | null>(null)
  const [deleting, setDeleting] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  const messages = activeId && loaded?.id === activeId ? loaded.messages : []
  const loadingConversation = activeId !== null && loaded?.id !== activeId && !loadError

  // Open the conversation named in the URL (also restores it after a reload).
  useEffect(() => {
    if (!activeId || loaded?.id === activeId) return
    const controller = new AbortController()
    api
      .getConversation(activeId, controller.signal)
      .then((conversation) => {
        setLoaded({ id: conversation.id, messages: conversation.messages })
        setSelected(conversation.document_ids)
        setLoadError(null)
      })
      .catch((cause) => {
        if (!controller.signal.aborted) setLoadError(cause instanceof ApiError ? cause : null)
      })
    return () => controller.abort()
  }, [activeId, loaded?.id])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, pendingQuestion])

  const openConversation = (id: string | null) => {
    setAskError(null)
    setLoadError(null)
    setHistoryOpen(false)
    setSearchParams(id ? { c: id } : {})
    if (!id) setSelected(null)
  }

  const ask = async (text: string) => {
    const question = text.trim()
    if (!question || pendingQuestion) return
    setPendingQuestion(question)
    setDraft('')
    setAskError(null)
    // Drop selections of papers deleted or no longer searchable.
    const stillSearchable = selected?.filter((id) => searchable.some((doc) => doc.id === id))
    try {
      const response = await api.ask({
        question,
        conversation_id: activeId,
        document_ids: stillSearchable && stillSearchable.length > 0 ? stillSearchable : null,
      })
      const earlier = loaded?.id === response.conversation.id ? loaded.messages : []
      setLoaded({
        id: response.conversation.id,
        messages: [...earlier, response.user_message, response.assistant_message],
      })
      if (response.conversation.id !== activeId) setSearchParams({ c: response.conversation.id })
      conversations.reload()
    } catch (cause) {
      setAskError(cause instanceof ApiError ? cause : new ApiError(0, 'unknown', 'Something went wrong.'))
      setDraft(question) // keep the question so it can be retried
    } finally {
      setPendingQuestion(null)
    }
  }

  const confirmDelete = async () => {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await api.deleteConversation(pendingDelete.id)
      if (pendingDelete.id === activeId) openConversation(null)
      conversations.reload()
    } finally {
      setDeleting(false)
      setPendingDelete(null)
    }
  }

  const activeTitle = conversations.data?.items.find((item) => item.id === activeId)?.title
  const list = (
    <ConversationList
      conversations={conversations.data?.items ?? null}
      activeId={activeId}
      onSelect={openConversation}
      onNew={() => openConversation(null)}
      onDelete={setPendingDelete}
    />
  )

  return (
    <div className="grid h-[calc(100dvh-7rem)] gap-6 sm:h-[calc(100dvh-8rem)] lg:grid-cols-[16rem_minmax(0,1fr)]">
      <Card className="hidden overflow-hidden lg:block">{list}</Card>

      <Card className="flex min-h-0 flex-col overflow-hidden">
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <button
            type="button"
            onClick={() => setHistoryOpen((value) => !value)}
            aria-expanded={historyOpen}
            className="rounded-md p-1.5 text-fg-muted hover:bg-surface-muted lg:hidden"
            aria-label="Conversation history"
          >
            <History className="size-4" />
          </button>
          <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-fg">
            {activeId ? (activeTitle ?? 'Conversation') : 'New conversation'}
          </h2>
        </div>

        {historyOpen && <div className="max-h-72 border-b border-border lg:hidden">{list}</div>}

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6" aria-live="polite">
          {loadError ? (
            <ErrorState
              title="Couldn't open this conversation"
              message={loadError.message}
              onRetry={() => openConversation(null)}
            />
          ) : loadingConversation ? (
            <div className="flex justify-center py-10">
              <Spinner className="size-5" />
            </div>
          ) : messages.length === 0 && !pendingQuestion ? (
            <EmptyChat
              hasPapers={!documents.data || searchable.length > 0}
              onExample={(question) => void ask(question)}
            />
          ) : (
            <div className="mx-auto max-w-3xl space-y-6">
              {messages.map((message, index) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  question={message.role === 'assistant' ? messages[index - 1]?.content : undefined}
                />
              ))}
              {pendingQuestion && (
                <>
                  <div className="flex justify-end">
                    <p className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-sm whitespace-pre-wrap text-accent-fg">
                      {pendingQuestion}
                    </p>
                  </div>
                  <div className="flex items-center gap-3 text-sm text-fg-muted">
                    <div className="flex size-8 items-center justify-center rounded-lg bg-accent-soft text-accent-text">
                      <Bot className="size-4" aria-hidden />
                    </div>
                    <Spinner />
                    Searching your papers and writing an answer…
                  </div>
                </>
              )}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="space-y-3 border-t border-border p-4">
          {askError && (
            <ErrorState
              title={
                askError.code === 'llm_not_configured' ? 'Gemini API key needed' : "Couldn't get an answer"
              }
              message={askError.message}
            />
          )}
          {health && !health.llm_configured && !askError && (
            <p className="flex items-start gap-2 text-xs text-fg-muted">
              <KeyRound className="mt-0.5 size-3.5 shrink-0 text-warning" aria-hidden />
              <span>
                Answers need a Gemini API key in <code className="font-mono">backend/.env</code>{' '}
                (see <Link to="/settings" className="text-accent-text hover:underline">Settings</Link>).
                Questions your papers can't answer are still recognised without one.
              </span>
            </p>
          )}
          <Composer value={draft} onChange={setDraft} onSubmit={() => void ask(draft)} busy={pendingQuestion !== null}>
            <div className={cn(searchable.length === 0 && 'pointer-events-none opacity-50')}>
              <DocumentFilter documents={searchable} selected={selected} onChange={setSelected} />
            </div>
          </Composer>
        </div>
      </Card>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete this conversation?"
        description={`“${pendingDelete?.title ?? ''}” and all its messages will be removed. This can't be undone.`}
        confirmLabel="Delete conversation"
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}
