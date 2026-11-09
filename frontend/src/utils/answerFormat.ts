/**
 * A deliberately small parser for the light Markdown the LLM writes
 * (paragraphs, bullet and numbered lists, headings, **bold**, `code`) plus
 * [S1]-style citation markers. Output is plain data rendered as React
 * elements, never as HTML, so text from papers or the model cannot inject
 * markup into the page.
 */

export type InlinePart =
  | { kind: 'text'; text: string }
  | { kind: 'bold'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'cite'; ids: string[] }

export type Block =
  | { kind: 'paragraph'; text: string }
  | { kind: 'heading'; text: string }
  | { kind: 'list'; ordered: boolean; items: string[] }

// **bold**, `code`, and citations like [S1], [S1, S3] or [S2; S4].
const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[S\d+(?:\s*[,;]\s*S\d+)*\])/g
const BULLET = /^\s*[-*•]\s+/
const NUMBERED = /^\s*\d+[.)]\s+/
const HEADING = /^\s*#{1,6}\s+/

export function parseInline(text: string): InlinePart[] {
  const parts: InlinePart[] = []
  let last = 0
  for (const match of text.matchAll(INLINE)) {
    const start = match.index ?? 0
    if (start > last) parts.push({ kind: 'text', text: text.slice(last, start) })
    const token = match[0]
    if (token.startsWith('**')) parts.push({ kind: 'bold', text: token.slice(2, -2) })
    else if (token.startsWith('`')) parts.push({ kind: 'code', text: token.slice(1, -1) })
    else parts.push({ kind: 'cite', ids: token.slice(1, -1).split(/\s*[,;]\s*/) })
    last = start + token.length
  }
  if (last < text.length) parts.push({ kind: 'text', text: text.slice(last) })
  return parts
}

export function parseBlocks(answer: string): Block[] {
  const blocks: Block[] = []
  let paragraph: string[] = []
  let list: { ordered: boolean; items: string[] } | null = null

  const flush = () => {
    if (paragraph.length) blocks.push({ kind: 'paragraph', text: paragraph.join(' ') })
    if (list) blocks.push({ kind: 'list', ...list })
    paragraph = []
    list = null
  }

  for (const raw of answer.split('\n')) {
    const line = raw.trim()
    if (!line) {
      flush()
    } else if (HEADING.test(line)) {
      flush()
      blocks.push({ kind: 'heading', text: line.replace(HEADING, '') })
    } else if (BULLET.test(line) || NUMBERED.test(line)) {
      const ordered = NUMBERED.test(line)
      if (paragraph.length || (list && list.ordered !== ordered)) flush()
      list ??= { ordered, items: [] }
      list.items.push(line.replace(ordered ? NUMBERED : BULLET, ''))
    } else if (list && raw.startsWith(' ')) {
      // An indented line continues the previous list item.
      list.items[list.items.length - 1] += ` ${line}`
    } else {
      if (list) flush()
      paragraph.push(line)
    }
  }
  flush()
  return blocks
}

/** DOM id of a source card, so a citation chip can scroll to it. */
export function sourceAnchorId(messageId: number, sourceId: string): string {
  return `message-${messageId}-${sourceId}`
}
