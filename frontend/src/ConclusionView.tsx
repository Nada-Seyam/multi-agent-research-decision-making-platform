import { type CSSProperties, type ReactNode } from 'react'
import { AlertCircle, ArrowRight, CheckCircle2, Clock3, ShieldCheck } from 'lucide-react'
import type { DiscussionSnapshot } from './api'
import { C } from './theme'
import { prettyAgent } from './utils'

const CONCLUSION_SECTIONS = [
  'Recommendations',
  'Technical Claims and Assumptions',
  'Unresolved Conditions, Evidence Gaps, and Risks',
]

type Tone = 'positive' | 'slate' | 'gold' | 'negative' | 'accent'
const TONE_COLOR: Record<Tone, string> = { positive: C.positive, slate: C.slate, gold: C.gold, negative: C.negative, accent: C.accentStrong }
const CONCLUSION_SECTION_META: Record<string, { tone: Tone; Icon: typeof CheckCircle2 }> = {
  'recommendations': { tone: 'positive', Icon: CheckCircle2 },
  'technical claims and assumptions': { tone: 'slate', Icon: ShieldCheck },
  'unresolved conditions, evidence gaps, and risks': { tone: 'gold', Icon: AlertCircle },
}
function consensusTone(status?: string | null): { tone: Tone; label: string } {
  if (!status) return { tone: 'gold', label: 'Recorded' }
  const s = status.toLowerCase(), label = status.replace(/_/g, ' ')
  if (/no|none|fail|split|deadlock/.test(s)) return { tone: 'negative', label }
  if (/partial|majority|mixed/.test(s)) return { tone: 'gold', label }
  return { tone: 'positive', label }
}

function normalizeConclusion(text: string) {
  const sectionNames = CONCLUSION_SECTIONS.join('|').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return text
    .replace(/\\([*_#`-])/g, '$1')
    .replace(/\r\n?/g, '\n')
    .replace(new RegExp(`\\s*(\\*\\*(?:${sectionNames})\\*\\*)\\s*`, 'gi'), '\n\n$1\n\n')
    .replace(/[ \t]+\*[ \t]+(?=\*\*)/g, '\n- ')
    .replace(/^\s*\*\s+/gm, '- ')
    .trim()
}

function inlineConclusion(text: string, keyPrefix: string): ReactNode[] {
  return text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g).filter(Boolean).map((part, index) => {
    const key = `${keyPrefix}-${index}`
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={key}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={key}>{part.slice(1, -1)}</code>
    return part.replace(/\*\*/g, '')
  })
}

function ConclusionText({ text }: { text: string }) {
  const blocks = normalizeConclusion(text).split(/\n\s*\n/).filter(Boolean)
  let tone: Tone | null = null
  return <div className="conclusion-text">{blocks.map((block, blockIndex) => {
    const lines = block.split('\n').map(line => line.trim()).filter(Boolean)
    const heading = lines.length === 1 ? lines[0].match(/^#{1,6}\s+(.+)$/) : null
    const boldHeading = lines.length === 1 ? lines[0].match(/^\*\*([^*]+)\*\*:?$/) : null
    const plainHeading = lines.length === 1 && CONCLUSION_SECTIONS.some(title => title.toLowerCase() === lines[0].replace(/:$/, '').toLowerCase())
    const headingTitle = heading?.[1] ?? boldHeading?.[1] ?? (plainHeading ? lines[0].replace(/:$/, '') : null)
    if (headingTitle != null) {
      const meta = CONCLUSION_SECTION_META[headingTitle.replace(/:$/, '').toLowerCase()]
      tone = meta?.tone ?? null
      const Icon = meta?.Icon
      return <h3 key={blockIndex} style={meta ? { color: TONE_COLOR[meta.tone], borderBottomColor: TONE_COLOR[meta.tone] + '55' } : undefined}>
        {Icon && <Icon size={13} />}{headingTitle}
      </h3>
    }
    if (lines.every(line => /^[-*]\s+/.test(line))) return (
      <ul key={blockIndex} style={tone ? ({ '--tone': TONE_COLOR[tone] } as CSSProperties) : undefined}>
        {lines.map((line, lineIndex) => <li key={lineIndex}>{inlineConclusion(line.replace(/^[-*]\s+/, ''), `${blockIndex}-${lineIndex}`)}</li>)}
      </ul>
    )
    return <p key={blockIndex} className={blockIndex === 0 ? 'lede' : undefined}>{inlineConclusion(lines.join(' '), `${blockIndex}`)}</p>
  })}</div>
}

export function ConclusionView({ snapshot }: { snapshot: DiscussionSnapshot }) {
  const d = snapshot.decision_summary, support = d?.support_personas || [], dissent = Object.entries(d?.dissenting_positions || {}), ratio = snapshot.participants.length ? support.length / snapshot.participants.length : 0
  const status = consensusTone(d?.consensus_status)
  return <div className="conclusion-layout fade-up">
    <article className="conclusion-card">
      <div className="conclusion-head-row">
        <div className="eyebrow"><CheckCircle2 size={13} style={{ verticalAlign: -2, marginRight: 6, color: C.positive }} />Discussion complete</div>
        <span className={`status-badge tone-${status.tone}`}>{status.label}</span>
      </div>
      <h2>{d?.preferred_candidate || 'Final conclusion'}</h2>
      <ConclusionText text={snapshot.conclusion || 'The discussion completed without a written conclusion.'} />
    </article>
    <aside className="decision-side">
      <div className="decision-card tone-slate">
        <div className="eyebrow"><ShieldCheck size={13} color={C.slate} />Decision signal</div>
        <h3>{d?.consensus_status?.replace(/_/g, ' ') || 'Recorded'}</h3>
        <div className="bar-track"><div className="bar-fill" style={{ width: `${ratio * 100}%`, background: C.slate }} /></div>
        <p className="metric-label">{support.length} of {snapshot.participants.length} agents explicitly support the preferred candidate.</p>
      </div>
      <div className="decision-card tone-positive">
        <div className="eyebrow"><CheckCircle2 size={13} color={C.positive} />Support</div>
        <ul className="plain-list">{support.length ? support.map(a => <li key={a}>{prettyAgent(a)}</li>) : <li>No explicit support list was produced.</li>}</ul>
      </div>
      {dissent.length > 0 && <div className="decision-card tone-negative">
        <div className="eyebrow"><AlertCircle size={13} color={C.negative} />Preserved dissent</div>
        <ul className="plain-list">{dissent.map(([a, p]) => <li key={a}><strong>{prettyAgent(a)}:</strong> {p}</li>)}</ul>
      </div>}
      <div className="decision-card tone-gold">
        <div className="eyebrow"><Clock3 size={13} color={C.gold} />Unresolved uncertainties</div>
        <ul className="plain-list">{d?.unresolved_uncertainties?.length ? d.unresolved_uncertainties.map(i => <li key={i}>{i}</li>) : <li>No unresolved uncertainties were recorded.</li>}</ul>
      </div>
      {d?.strongest_alternative && <div className="decision-card tone-accent">
        <div className="eyebrow"><ArrowRight size={13} color={C.accent} />Strongest alternative</div>
        <h3>{d.strongest_alternative}</h3>
      </div>}
    </aside>
  </div>
}
