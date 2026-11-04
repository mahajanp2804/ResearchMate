import { FileText } from 'lucide-react'
import { ComingSoon } from '../components/ComingSoon'

export default function Documents() {
  return (
    <ComingSoon
      icon={FileText}
      phase={2}
      title="Your paper library"
      description="Upload PDFs here. Each paper is checked, its text is extracted page by page, and it is indexed for search."
      capabilities={[
        'Upload several PDFs at once, with progress',
        'Detect duplicate uploads by file hash',
        'Extract titles, metadata and page-level text',
        'Flag scanned PDFs that have no extractable text',
        'Track processing and indexing status',
        'Delete or re-index papers',
      ]}
    />
  )
}
