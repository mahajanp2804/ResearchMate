import { SendHorizonal } from 'lucide-react'
import { useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react'
import { Button } from '../ui/Button'
import { Spinner } from '../ui/Feedback'

interface ComposerProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  busy: boolean
  disabled?: boolean
  /** Controls shown beside the send button, e.g. the paper filter. */
  children?: ReactNode
}

const MAX_HEIGHT_PX = 200

/** Question box: Enter sends, Shift+Enter adds a line. */
export function Composer({ value, onChange, onSubmit, busy, disabled = false, children }: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Grow with the text up to a limit, then scroll.
  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_HEIGHT_PX)}px`
  }, [value])

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      if (!busy && value.trim()) onSubmit()
    }
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        if (!busy && value.trim()) onSubmit()
      }}
      className="rounded-xl border border-border bg-surface p-2 shadow-card focus-within:border-accent"
    >
      <label className="sr-only" htmlFor="question">
        Ask a question about your papers
      </label>
      <textarea
        id="question"
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        disabled={disabled}
        maxLength={1000}
        placeholder="Ask a question about your papers…"
        className="block w-full resize-none bg-transparent px-2 py-1.5 text-sm text-fg placeholder:text-fg-subtle focus:outline-none disabled:opacity-60"
      />
      <div className="mt-1 flex flex-wrap items-center gap-2">
        {children}
        <Button type="submit" size="sm" disabled={busy || disabled || !value.trim()} className="ml-auto">
          {busy ? <Spinner className="text-accent-fg" /> : <SendHorizonal className="size-3.5" aria-hidden />}
          Ask
        </Button>
      </div>
    </form>
  )
}
