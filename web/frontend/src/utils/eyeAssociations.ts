import type { Silhouette } from './silhouette'
export type Expression = 'affectionate-tired' | 'amused' | 'wonder' | 'neutral'
export interface EyeAssociation {
  version: 1
  concept: string
  expression: Expression
  silhouette: Silhouette
  confirmed_at: string
}
export const ASSOCIATION_KEY = 'wizard-eye-associations-v1'
export function conceptKey(value: string): string {
  return value.replace(/^txt:word_/, '').normalize('NFKC').trim().toLocaleLowerCase().replace(/\s+/g, ' ')
}
export function readAssociations(): EyeAssociation[] {
  try {
    const values = JSON.parse(localStorage.getItem(ASSOCIATION_KEY) || '[]')
    if (!Array.isArray(values)) return []
    return values.slice(0, 32).filter(value => value?.version === 1 && typeof value.concept === 'string'
      && value.concept.length <= 120 && ['affectionate-tired', 'amused', 'wonder', 'neutral'].includes(value.expression)
      && typeof value.silhouette?.path === 'string' && value.silhouette.path.length <= 128000
      && /^[MLZ0-9 .-]+$/.test(value.silhouette.path)
      && Number.isInteger(value.silhouette.width) && value.silhouette.width > 0 && value.silhouette.width <= 512
      && Number.isInteger(value.silhouette.height) && value.silhouette.height > 0 && value.silhouette.height <= 512
      && Array.isArray(value.silhouette.vector) && value.silhouette.vector.length === 16
      && value.silhouette.vector.every((number: unknown) => typeof number === 'number' && Number.isFinite(number) && number >= 0 && number <= 1))
  } catch { return [] }
}
export function saveAssociation(value: EyeAssociation): EyeAssociation[] {
  if (!value.concept || value.concept.length > 120) throw new Error('Enter a concept label of 1–120 characters.')
  if (value.silhouette.path.length > 128000) throw new Error('The outline is too complex. Use a simpler cutout.')
  const values = [value, ...readAssociations().filter(previous => conceptKey(previous.concept) !== conceptKey(value.concept))].slice(0, 32)
  localStorage.setItem(ASSOCIATION_KEY, JSON.stringify(values))
  return values
}
export function matchAssociation(concepts: string[], values: EyeAssociation[]): EyeAssociation | undefined {
  const keys = new Set(concepts.map(conceptKey))
  return values.find(value => keys.has(conceptKey(value.concept)))
}
