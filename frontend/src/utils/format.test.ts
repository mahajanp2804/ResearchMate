import { describe, expect, it } from 'vitest'
import { formatScore, formatPercent, methodColor } from './evaluation'
import { cn, formatBytes, formatPageRange, formatRelativeTime, pluralize } from './format'

describe('format helpers', () => {
  it('formats page ranges as citations do', () => {
    expect(formatPageRange(3, 3)).toBe('p. 3')
    expect(formatPageRange(3, 4)).toBe('pp. 3–4')
  })

  it('pluralises with the count', () => {
    expect(pluralize(1, 'paper')).toBe('1 paper')
    expect(pluralize(2, 'passage')).toBe('2 passages')
    expect(pluralize(0, 'analysis', 'analyses')).toBe('0 analyses')
  })

  it('formats file sizes', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })

  it('describes times relative to now', () => {
    const now = new Date('2026-11-15T12:00:00Z')
    expect(formatRelativeTime('2026-11-15T11:59:30Z', now)).toBe('just now')
    expect(formatRelativeTime('2026-11-15T11:00:00Z', now)).toMatch(/hour/)
  })

  it('joins class names, skipping falsy ones', () => {
    expect(cn('a', false, null, undefined, 'b')).toBe('a b')
  })
})

describe('evaluation helpers', () => {
  it('shows scores with three decimals and missing ones as a dash', () => {
    expect(formatScore(0.7864)).toBe('0.786')
    expect(formatScore(null)).toBe('—')
    expect(formatPercent(0.456)).toBe('46%')
  })

  it('gives each method a fixed colour slot', () => {
    expect(methodColor('dense')).toBe('var(--series-1)')
    expect(methodColor('hybrid_rerank')).toBe('var(--series-4)')
  })
})
