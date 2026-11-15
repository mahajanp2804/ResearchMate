import { describe, expect, it } from 'vitest'
import { askAboutUrl, compareUrl, insightsUrl, papersFromParams, summaryUrl } from './links'

describe('links between features', () => {
  it('carries the selected papers in the URL', () => {
    expect(askAboutUrl(['a1'])).toBe('/assistant?papers=a1')
    expect(compareUrl(['a1', 'b2'])).toBe('/comparison?papers=a1,b2')
    expect(insightsUrl([])).toBe('/insights')
    expect(summaryUrl('a1')).toBe('/papers/a1/summary')
  })

  it('reads the papers back, ignoring empty entries', () => {
    const url = new URL(`http://localhost${compareUrl(['a1', 'b2'])}`)
    expect(papersFromParams(url.searchParams)).toEqual(['a1', 'b2'])
    expect(papersFromParams(new URLSearchParams('papers=,a1,'))).toEqual(['a1'])
    expect(papersFromParams(new URLSearchParams(''))).toEqual([])
  })

  it('escapes ids that are not URL-safe', () => {
    expect(askAboutUrl(['a b&c'])).toBe('/assistant?papers=a%20b%26c')
  })
})
