import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import { ApiError, api } from '../services/api'

export type UploadState = 'queued' | 'uploading' | 'processing' | 'created' | 'duplicate' | 'rejected'

export interface UploadItem {
  id: number
  name: string
  size: number
  state: UploadState
  /** Fraction of bytes sent, 0–1. */
  progress: number
  message?: string
}

const ACTIVE_STATES: UploadState[] = ['queued', 'uploading', 'processing']

export function isActiveUpload(item: UploadItem): boolean {
  return ACTIVE_STATES.includes(item.state)
}

/**
 * Uploads files one at a time, one request per file, so each file gets its
 * own progress bar and its own result. The server extracts text during the
 * request, so uploading in sequence also keeps the laptop responsive.
 */
export function useUploadQueue({ maxSizeMb, onUploaded }: { maxSizeMb: number; onUploaded: () => void }) {
  const [items, setItems] = useState<UploadItem[]>([])
  const pendingRef = useRef<{ id: number; file: File }[]>([])
  const runningRef = useRef(false)
  const nextIdRef = useRef(1)
  const onUploadedRef = useRef(onUploaded)
  useLayoutEffect(() => {
    onUploadedRef.current = onUploaded
  })

  const update = useCallback((id: number, patch: Partial<UploadItem>) => {
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }, [])

  const drain = useCallback(async () => {
    if (runningRef.current) return
    runningRef.current = true
    while (pendingRef.current.length > 0) {
      const { id, file } = pendingRef.current.shift()!
      update(id, { state: 'uploading', progress: 0 })
      try {
        const response = await api.uploadDocument(file, (fraction) =>
          // Once every byte is sent, the server is extracting text.
          update(id, { progress: fraction, state: fraction >= 1 ? 'processing' : 'uploading' }),
        )
        const result = response.results[0]
        const title = result.document?.title ?? result.document?.filename
        if (result.status === 'created') {
          update(id, { state: 'created', progress: 1, message: `Added as “${title}”` })
        } else if (result.status === 'duplicate') {
          update(id, { state: 'duplicate', progress: 1, message: `Already in your library as “${title}”` })
        } else {
          update(id, { state: 'rejected', message: result.error?.message ?? 'The file was rejected.' })
        }
        onUploadedRef.current()
      } catch (error) {
        update(id, {
          state: 'rejected',
          message: error instanceof ApiError ? error.message : 'Upload failed. Please try again.',
        })
      }
    }
    runningRef.current = false
  }, [update])

  const addFiles = useCallback(
    (files: File[]) => {
      const added: UploadItem[] = files.map((file) => {
        const item: UploadItem = {
          id: nextIdRef.current++,
          name: file.name,
          size: file.size,
          state: 'queued',
          progress: 0,
        }
        // Obvious problems are caught before uploading; the server re-checks everything.
        if (!file.name.toLowerCase().endsWith('.pdf')) {
          return { ...item, state: 'rejected', message: 'Only PDF files are supported.' }
        }
        if (file.size > maxSizeMb * 1024 * 1024) {
          return { ...item, state: 'rejected', message: `Larger than the ${maxSizeMb} MB limit.` }
        }
        pendingRef.current.push({ id: item.id, file })
        return item
      })
      setItems((current) => [...added, ...current])
      void drain()
    },
    [drain, maxSizeMb],
  )

  const clearFinished = useCallback(() => {
    setItems((current) => current.filter(isActiveUpload))
  }, [])

  return { items, addFiles, clearFinished, busy: items.some(isActiveUpload) }
}
