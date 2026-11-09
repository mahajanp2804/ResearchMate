import { parseBlocks, parseInline } from '../../utils/answerFormat'
import { cn } from '../../utils/format'

interface AnswerTextProps {
  text: string
  /** Source ids the answer may cite, e.g. ["S1", "S2"]. */
  sourceIds: string[]
  onCite: (sourceId: string) => void
}

function Inline({ text, sourceIds, onCite }: AnswerTextProps) {
  return (
    <>
      {parseInline(text).map((part, index) => {
        if (part.kind === 'bold') return <strong key={index}>{part.text}</strong>
        if (part.kind === 'code') {
          return (
            <code key={index} className="rounded bg-surface-muted px-1 py-0.5 font-mono text-[0.85em]">
              {part.text}
            </code>
          )
        }
        if (part.kind === 'cite') {
          return (
            <span key={index} className="whitespace-nowrap">
              {part.ids.map((id) => {
                const known = sourceIds.includes(id)
                return (
                  <button
                    key={id}
                    type="button"
                    disabled={!known}
                    onClick={() => onCite(id)}
                    title={known ? `Show source ${id}` : `${id} is not one of the sources`}
                    className={cn(
                      'mx-0.5 rounded px-1 align-baseline font-mono text-[0.75em] font-semibold',
                      known
                        ? 'bg-accent-soft text-accent-text hover:bg-accent hover:text-accent-fg'
                        : 'bg-danger-soft text-danger line-through',
                    )}
                  >
                    {id}
                  </button>
                )
              })}
            </span>
          )
        }
        return <span key={index}>{part.text}</span>
      })}
    </>
  )
}

/** Renders an answer's light Markdown with clickable [S#] citation chips. */
export function AnswerText(props: AnswerTextProps) {
  return (
    <div className="space-y-3 text-sm leading-relaxed text-fg">
      {parseBlocks(props.text).map((block, index) => {
        if (block.kind === 'heading') {
          return (
            <h4 key={index} className="font-semibold">
              <Inline {...props} text={block.text} />
            </h4>
          )
        }
        if (block.kind === 'list') {
          const List = block.ordered ? 'ol' : 'ul'
          return (
            <List
              key={index}
              className={cn('space-y-1 pl-5', block.ordered ? 'list-decimal' : 'list-disc')}
            >
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  <Inline {...props} text={item} />
                </li>
              ))}
            </List>
          )
        }
        return (
          <p key={index}>
            <Inline {...props} text={block.text} />
          </p>
        )
      })}
    </div>
  )
}
