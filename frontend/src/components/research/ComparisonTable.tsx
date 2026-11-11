import { BookOpenText, ExternalLink } from 'lucide-react'
import { Link } from 'react-router'
import { api } from '../../services/api'
import type { ComparisonResponse } from '../../types/api'
import { AspectContent } from './AspectContent'
import type { CitedPassage } from './CitationDialog'

interface ComparisonTableProps {
  comparison: ComparisonResponse
  onCite: (passage: CitedPassage) => void
}

/**
 * Aspects as rows, papers as columns. Every cell cites its own paper's
 * passages; a cell the paper doesn't cover reads "Not reported". Scrolls
 * sideways on narrow screens with the aspect names kept in view.
 */
export function ComparisonTable({ comparison, onCite }: ComparisonTableProps) {
  const { aspects, papers } = comparison
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-border bg-surface-muted/50">
            <th
              scope="col"
              className="sticky left-0 z-10 w-28 bg-surface-muted px-4 py-3 text-xs font-medium text-fg-muted sm:w-40"
            >
              Aspect
            </th>
            {papers.map((paper, index) => (
              <th
                key={paper.document_id}
                scope="col"
                className="min-w-[16rem] px-4 py-3 align-top font-normal"
              >
                <span className="text-xs font-medium text-fg-subtle">Paper {index + 1}</span>
                <span className="mt-0.5 line-clamp-2 block text-sm font-semibold text-fg">
                  {paper.document_title}
                </span>
                <span className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs">
                  <Link
                    to={`/papers/${paper.document_id}/summary`}
                    className="inline-flex items-center gap-1 text-accent-text hover:underline"
                  >
                    <BookOpenText className="size-3" aria-hidden />
                    Summary
                  </Link>
                  <a
                    href={api.documentFileUrl(paper.document_id)}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-accent-text hover:underline"
                  >
                    <ExternalLink className="size-3" aria-hidden />
                    PDF
                  </a>
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {aspects.map((row) => (
            <tr key={row.key}>
              <th
                scope="row"
                className="sticky left-0 z-10 bg-surface px-4 py-3 align-top text-xs font-semibold text-fg"
              >
                {row.label}
              </th>
              {papers.map((paper) => {
                const cell = paper.aspects.find((aspect) => aspect.key === row.key)
                return (
                  <td key={paper.document_id} className="px-4 py-3 align-top">
                    {cell ? (
                      <AspectContent
                        aspect={cell}
                        profile={paper}
                        notFoundText="Not reported"
                        noun="cell"
                        onCite={onCite}
                      />
                    ) : (
                      <span className="text-sm text-fg-subtle italic">Not reported</span>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
