import type { DocumentSummary } from '../../types/api'
import { cn, pluralize } from '../../utils/format'

interface PaperPickerProps {
  papers: DocumentSummary[]
  /** Selected ids, in the order they were picked (the column order). */
  selected: string[]
  max: number
  onChange: (selected: string[]) => void
}

/** Checkbox list of searchable papers; the pick order becomes the column order. */
export function PaperPicker({ papers, selected, max, onChange }: PaperPickerProps) {
  const full = selected.length >= max
  const toggle = (id: string) =>
    onChange(selected.includes(id) ? selected.filter((item) => item !== id) : [...selected, id])

  return (
    <fieldset>
      <legend className="sr-only">Papers to compare</legend>
      <ul className="max-h-72 divide-y divide-border overflow-y-auto rounded-lg border border-border">
        {papers.map((paper) => {
          const checked = selected.includes(paper.id)
          const disabled = !checked && full
          const position = selected.indexOf(paper.id) + 1
          return (
            <li key={paper.id}>
              <label
                className={cn(
                  'flex cursor-pointer items-start gap-3 px-3 py-2.5 hover:bg-surface-muted/60',
                  disabled && 'cursor-not-allowed opacity-50 hover:bg-transparent',
                )}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled}
                  onChange={() => toggle(paper.id)}
                  className="mt-0.5 size-4 shrink-0 accent-accent"
                />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm text-fg">{paper.title ?? paper.filename}</span>
                  <span className="block truncate text-xs text-fg-subtle">
                    {[
                      paper.extracted_metadata.author,
                      paper.page_count !== null && pluralize(paper.page_count, 'page'),
                    ]
                      .filter(Boolean)
                      .join(' · ')}
                  </span>
                </span>
                {checked && (
                  <span
                    className="shrink-0 rounded-md bg-accent-soft px-1.5 text-xs font-semibold text-accent-text tabular-nums"
                    title="Column"
                  >
                    {position}
                  </span>
                )}
              </label>
            </li>
          )
        })}
      </ul>
      <p className="mt-2 text-xs text-fg-subtle">
        {selected.length} of up to {max} selected
        {full && ' · deselect a paper to choose another'}
      </p>
    </fieldset>
  )
}
