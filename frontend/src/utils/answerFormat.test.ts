import { describe, expect, it } from 'vitest'
import { parseBlocks, parseInline, sourceAnchorId } from './answerFormat'

describe('parseInline', () => {
  it('splits bold, code and citation markers from plain text', () => {
    expect(parseInline('Uses **FedProx** with `mu = 0.01` [S1].')).toEqual([
      { kind: 'text', text: 'Uses ' },
      { kind: 'bold', text: 'FedProx' },
      { kind: 'text', text: ' with ' },
      { kind: 'code', text: 'mu = 0.01' },
      { kind: 'text', text: ' ' },
      { kind: 'cite', ids: ['S1'] },
      { kind: 'text', text: '.' },
    ])
  })

  it('reads grouped citations separated by commas or semicolons', () => {
    expect(parseInline('[S1, S3][S2; S4]')).toEqual([
      { kind: 'cite', ids: ['S1', 'S3'] },
      { kind: 'cite', ids: ['S2', 'S4'] },
    ])
  })

  it('keeps markup from papers or the model as plain text', () => {
    // The parts are rendered as React text, so this can never become HTML.
    const parts = parseInline('<img src=x onerror=alert(1)> and <script>x</script>')
    expect(parts).toEqual([{ kind: 'text', text: '<img src=x onerror=alert(1)> and <script>x</script>' }])
  })

  it('leaves unmatched markers and brackets alone', () => {
    expect(parseInline('a **b and [figure 2] c')).toEqual([
      { kind: 'text', text: 'a **b and [figure 2] c' },
    ])
  })
})

describe('parseBlocks', () => {
  it('groups paragraphs, headings and lists', () => {
    const answer = [
      '## Summary',
      'FedProx adds a proximal term',
      'to the local objective [S1].',
      '',
      '- MNIST [S2]',
      '- FEMNIST',
      '  with non-IID splits [S3]',
      '1. First step',
      '2. Second step',
      'Closing line.',
    ].join('\n')
    expect(parseBlocks(answer)).toEqual([
      { kind: 'heading', text: 'Summary' },
      { kind: 'paragraph', text: 'FedProx adds a proximal term to the local objective [S1].' },
      { kind: 'list', ordered: false, items: ['MNIST [S2]', 'FEMNIST with non-IID splits [S3]'] },
      { kind: 'list', ordered: true, items: ['First step', 'Second step'] },
      { kind: 'paragraph', text: 'Closing line.' },
    ])
  })

  it('returns nothing for an empty answer', () => {
    expect(parseBlocks('\n  \n')).toEqual([])
  })
})

it('builds a stable anchor id for a source card', () => {
  expect(sourceAnchorId(12, 'S3')).toBe('message-12-S3')
})
