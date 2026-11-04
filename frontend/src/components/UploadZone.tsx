import { FileUp } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'
import { cn } from '../utils/format'

interface UploadZoneProps {
  maxSizeMb: number
  maxPages: number
  disabled?: boolean
  onFiles: (files: File[]) => void
}

export function UploadZone({ maxSizeMb, maxPages, disabled = false, onFiles }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    const files = Array.from(event.dataTransfer.files)
    if (files.length > 0) onFiles(files)
  }

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={cn(
        'flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors',
        dragging ? 'border-accent bg-accent-soft' : 'border-border-strong bg-surface',
        disabled && 'opacity-60',
      )}
    >
      <div className="mb-3 flex size-11 items-center justify-center rounded-xl bg-accent-soft text-accent-text">
        <FileUp className="size-5" aria-hidden />
      </div>
      <p className="text-sm font-medium text-fg">
        Drag PDFs here, or{' '}
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="font-semibold text-accent-text hover:underline disabled:pointer-events-none"
        >
          browse your files
        </button>
      </p>
      <p className="mt-1 text-xs text-fg-subtle">
        PDF with selectable text · up to {maxSizeMb} MB and {maxPages} pages each · scanned PDFs
        need OCR, which is not supported yet
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(event) => {
          const files = Array.from(event.target.files ?? [])
          if (files.length > 0) onFiles(files)
          // Reset so choosing the same file again still fires onChange.
          event.target.value = ''
        }}
      />
    </div>
  )
}
