import { highlightParts } from '../utils/highlight'

/** Text with query terms marked, so it is visible why a passage was retrieved. */
export function Highlighted({ text, terms }: { text: string; terms: string[] }) {
  return (
    <>
      {highlightParts(text, terms).map((part, index) =>
        part.match ? (
          <mark key={index} className="rounded-sm bg-warning-soft px-0.5 text-fg">
            {part.text}
          </mark>
        ) : (
          part.text
        ),
      )}
    </>
  )
}
