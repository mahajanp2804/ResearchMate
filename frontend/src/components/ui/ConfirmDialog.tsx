import { useEffect, useRef } from 'react'
import { Button } from './Button'
import { Spinner } from './Feedback'

interface ConfirmDialogProps {
  open: boolean
  title: string
  description: string
  confirmLabel: string
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
}

/** Modal confirmation built on <dialog>, which handles focus trapping and Esc. */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  busy = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])

  return (
    <dialog
      ref={dialogRef}
      onCancel={(event) => {
        event.preventDefault()
        if (!busy) onCancel()
      }}
      className="m-auto w-[calc(100%-2rem)] max-w-md rounded-xl border border-border bg-surface p-0 text-fg shadow-xl backdrop:bg-black/40"
    >
      <div className="px-6 pt-6 pb-4">
        <h2 className="text-base font-semibold">{title}</h2>
        <p className="mt-2 text-sm text-fg-muted">{description}</p>
      </div>
      <div className="flex justify-end gap-2 border-t border-border bg-surface-muted/50 px-6 py-4">
        <Button variant="secondary" onClick={onCancel} disabled={busy}>
          Cancel
        </Button>
        <Button
          onClick={onConfirm}
          disabled={busy}
          className="bg-danger-solid text-white hover:bg-danger-solid-hover"
          autoFocus
        >
          {busy && <Spinner className="text-white" />}
          {confirmLabel}
        </Button>
      </div>
    </dialog>
  )
}
