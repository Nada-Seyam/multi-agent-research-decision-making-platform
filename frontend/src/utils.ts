import type { DiscussionSnapshot } from './api'
import { AGENT_COLORS } from './theme'

export type View = 'discussion' | 'analytics' | 'conclusion'
export const prettyAgent = (id: string) => id.replace(/^persona[_-]?/i, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase())
const objectiveText = (snapshot: DiscussionSnapshot) => typeof snapshot.objective === 'string' ? snapshot.objective : snapshot.objective.goal || 'Multi-agent decision discussion'
/** Pulls the discussion topic out of an uploaded objective file: a "Topic:" / "Discussion topic:" / "Title:" line (value on the same line or on the next line), else a plain first-line title. */
export function extractTopic(text: string): string | null {
  const tidy = (line: string) => line.replace(/^[\s>#*_`•\-–—]*(?:\d+[.)]\s+)?/, '').replace(/[*_`]+/g, '').trim()
  const value = (raw: string) => { const v = raw.trim().replace(/^["“'‘]+|["”'’]+$/g, '').trim(); return v && v.length <= 300 && !/^[\w &/-]{2,40}:$/.test(v) ? v : null }
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).map(tidy)
  for (let i = 0; i < lines.length; i++) {
    const m = /^(?:the\s+)?(?:discussion\s+)?(topic|title|subject)(s?)\s*(?:[:：\-–—=]\s*(.*))?$/i.exec(lines[i])
    if (!m) continue
    const inline = m[3] ? value(m[3]) : null
    if (inline) return inline
    if (m[2]) continue // a "Discussion topics" list heading, not a single topic
    const next = lines.slice(i + 1).find(Boolean)
    const fromNext = next ? value(next) : null
    if (fromNext) return fromNext
  }
  const first = lines.find(Boolean)
  const sectionWord = /^(objective|goal|final goal|constraints?|evidence|requirements?|background|context|overview|summary)\b/i
  return first && first.length <= 160 && !/:\s*$/.test(first) && !sectionWord.test(first) ? value(first) : null
}
/** Title shown on the discussion page: the discussion topic (backend value if the snapshot carries one, else what the user typed or the uploaded file's topic). Falls back to the objective only if no topic is known. */
export const topicText = (snapshot: DiscussionSnapshot, known: string | null) => {
  const clean = (v: unknown) => typeof v === 'string' && v.trim() ? v.trim() : null
  const pick = (o: unknown) => o && typeof o === 'object' ? clean((o as Record<string, unknown>).topic) ?? clean((o as Record<string, unknown>).discussion_topic) : null
  return pick(snapshot) ?? pick(snapshot.objective) ?? clean(known) ?? objectiveText(snapshot)
}
export const allMessages = (snapshot: DiscussionSnapshot) => snapshot.rounds.flatMap(round => round.messages).sort((a, b) => a.sequence - b.sequence)
export const agentColor = (agent: string, participants: string[]) => AGENT_COLORS[Math.max(0, participants.indexOf(agent)) % AGENT_COLORS.length]
export const last = <T,>(items: T[]): T | undefined => items[items.length - 1]
