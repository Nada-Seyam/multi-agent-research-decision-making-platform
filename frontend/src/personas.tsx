import { useEffect, useId, useState } from 'react'
import { C } from './theme'
import { prettyAgent } from './utils'
import type { ParticipantSelectionMode } from './api'

/* ============================================================================
   PERSONA PREVIEW (Setup page only)
   Shared setup metadata describing the six personas the backend can seat for
   a discussion. The setup page uses these ids for custom roster selection;
   the backend remains authoritative and validates every submitted id.
   ========================================================================== */
type CriterionId = 'capability' | 'latency' | 'reliability' | 'auditability' | 'operational_simplicity' | 'cost_efficiency'
export const CRITERIA: { id: CriterionId; label: string }[] = [
  { id: 'capability', label: 'Capability' },
  { id: 'latency', label: 'Latency' },
  { id: 'reliability', label: 'Reliability' },
  { id: 'auditability', label: 'Auditability' },
  { id: 'operational_simplicity', label: 'Simplicity' },
  { id: 'cost_efficiency', label: 'Cost' },
]
export interface PersonaInfo { id: string; name: string; role: string; code: string; marker: string; focus: string; priorities: Record<CriterionId, number> }
export const PERSONAS: PersonaInfo[] = [
  { id: 'capability_maximizer', name: 'Capability Maximizer', role: 'Principal Research Architect', code: 'CM', marker: '#B85C44',
    focus: 'Argues for model capability, scaling headroom, compute efficiency, and long-context performance.',
    priorities: { capability: 5, latency: 4, reliability: 2, auditability: 1, operational_simplicity: 1, cost_efficiency: 1 } },
  { id: 'practical_architect', name: 'Practical Systems Architect', role: 'Staff Systems Engineer', code: 'PA', marker: '#71856B',
    focus: 'Argues for reliability, simplicity, and predictable performance that is easy to deploy and maintain.',
    priorities: { capability: 2, latency: 2, reliability: 5, auditability: 2, operational_simplicity: 5, cost_efficiency: 3 } },
  { id: 'safety_governance_advocate', name: 'Safety and Governance Advocate', role: 'Head of AI Safety', code: 'SG', marker: '#A84E45',
    focus: 'Argues for accountability, observability, privacy, robustness, and strong governance safeguards.',
    priorities: { capability: 1, latency: 1, reliability: 4, auditability: 5, operational_simplicity: 2, cost_efficiency: 2 } },
  { id: 'product_user_advocate', name: 'Product and User Advocate', role: 'Head of Product', code: 'PU', marker: '#B38A3E',
    focus: 'Argues for user value, usability, accessibility, and solutions people can adopt successfully.',
    priorities: { capability: 3, latency: 3, reliability: 4, auditability: 1, operational_simplicity: 4, cost_efficiency: 2 } },
  { id: 'cost_sustainability_analyst', name: 'Cost and Sustainability Analyst', role: 'Finance & Sustainability Lead', code: 'CS', marker: '#40545A',
    focus: 'Argues for total cost of ownership, resource efficiency, and predictable, sustainable spend.',
    priorities: { capability: 1, latency: 2, reliability: 3, auditability: 2, operational_simplicity: 3, cost_efficiency: 5 } },
  { id: 'strategic_consensus_builder', name: 'Strategic Consensus Builder', role: 'VP of Strategy', code: 'SB', marker: '#68706A',
    focus: 'Argues for balanced, implementable solutions that preserve strategic flexibility across perspectives.',
    priorities: { capability: 3, latency: 2, reliability: 3, auditability: 3, operational_simplicity: 3, cost_efficiency: 2 } },
]

export function PriorityGlyph({ persona }: { persona: PersonaInfo }) {
  return <div className="priority-glyph">{CRITERIA.map(c => (
    <div key={c.id} className="priority-bar" title={`${c.label}: ${persona.priorities[c.id]}/5`}
      style={{ height: `${(persona.priorities[c.id] / 5) * 15}px`, background: persona.marker }} />
  ))}</div>
}

function PersonaCard({ persona, selectable = false, selected = false, onToggle }: { persona: PersonaInfo; selectable?: boolean; selected?: boolean; onToggle?: () => void }) {
  const content = <>
      <div className="persona-card-head">
        <span className="persona-avatar" style={{ border: `1.5px solid ${persona.marker}`, color: persona.marker, background: `${persona.marker}14` }}>{persona.code}</span>
        <div style={{ minWidth: 0 }}>
          <div className="persona-card-name">{persona.name}</div>
          <div className="persona-card-role">{persona.role}</div>
        </div>
      </div>
      <p className="persona-card-focus">{persona.focus}</p>
      <PriorityGlyph persona={persona} />
      {selectable && <span className="persona-selection-state">{selected ? 'Selected' : 'Add to room'}</span>}
    </>
  if (selectable) return <button type="button" className={`persona-card persona-card-selectable ${selected ? 'selected' : ''}`} onClick={onToggle} aria-pressed={selected}>{content}</button>
  return <div className="persona-card">{content}</div>
}

/** Static circular preview of the six-persona panel. No live discussion state
 *  is involved. Rather than a hover-driven tooltip (the full detail already
 *  lives in the "Available personas" column), this quietly self-cycles
 *  through each persona with a soft, ambient highlight. */
export function PersonaTopologyPreview({ participantCount = 4, mode = 'random' }: { participantCount?: number; mode?: ParticipantSelectionMode }) {
  const size = 260
  const cx = size / 2, cy = size / 2, r = size * 0.34, nodeR = size * 0.105
  const n = PERSONAS.length
  const [activeIndex, setActiveIndex] = useState(0)
  const uid = 'pp' + useId().replace(/[:]/g, '')
  useEffect(() => {
    const timer = window.setInterval(() => setActiveIndex(i => (i + 1) % n), 2600)
    return () => window.clearInterval(timer)
  }, [n])
  const active = PERSONAS[activeIndex]
  const points = PERSONAS.map((p, i) => {
    const angle = (i * 2 * Math.PI) / n - Math.PI / 2
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), p, angle }
  })
  const padAngle = (nodeR + 7) / r
  function arcFor(a: typeof points[number], b: typeof points[number]) {
    let angleA = a.angle, angleB = b.angle
    if (angleB < angleA) angleB += 2 * Math.PI
    const sa = angleA + padAngle, sb = angleB - padAngle
    const x1 = cx + r * Math.cos(sa), y1 = cy + r * Math.sin(sa)
    const x2 = cx + r * Math.cos(sb), y2 = cy + r * Math.sin(sb)
    return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r.toFixed(2)} ${r.toFixed(2)} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`
  }
  const centerClear = nodeR + size * 0.09
  return (
    <div className="persona-preview">
      <svg className="persona-graph-svg" viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <marker id={`${uid}-arrow`} viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill={C.borderStrong} /></marker>
          <marker id={`${uid}-arrow-active`} viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill={C.slate} /></marker>
        </defs>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.border} strokeWidth="1" strokeDasharray="1.5 5" />
        {points.map((p, i) => {
          const next = points[(i + 1) % n]
          const connected = active.id === p.p.id || active.id === next.p.id
          return (
            <path key={i} className="persona-arc" d={arcFor(p, next)} fill="none" stroke={connected ? C.slate : C.borderStrong} strokeWidth={connected ? 1.6 : 1}
              opacity={connected ? 1 : 0.4} markerEnd={`url(#${connected ? `${uid}-arrow-active` : `${uid}-arrow`})`} />
          )
        })}
        <circle cx={cx} cy={cy} r={centerClear} fill={C.bg} stroke={C.border} strokeWidth="1" />
        <text x={cx} y={cy - 6} textAnchor="middle" fontSize="9.5" fontFamily={C.mono} fill={C.textFaint} letterSpacing="0.06em">PANEL PREVIEW</text>
        <text x={cx} y={cy + 11} textAnchor="middle" fontSize="15" fontFamily={C.serif} fill={C.text}>{participantCount} agents</text>
        <text x={cx} y={cy + centerClear - 9} textAnchor="middle" fontSize="8.5" fontFamily={C.mono} fill={C.textFaint}>{mode === 'random' ? 'SEEDED RANDOM PANEL' : 'CUSTOM ROSTER'}</text>
        {points.map(({ x, y, p }) => {
          const isActive = active.id === p.id
          return (
            <g key={p.id} transform={`translate(${x},${y})`}>
              {isActive && <circle className="persona-node-ring" r={nodeR + 5} fill="none" stroke={p.marker} strokeWidth="1" opacity="0.5" />}
              <circle className="persona-node-circle" r={nodeR} fill={isActive ? `${p.marker}22` : C.surface} stroke={p.marker} strokeWidth={isActive ? 2.25 : 1.4} />
              <text textAnchor="middle" dy={4.5} fontSize={nodeR * 0.5} fontWeight={700} fill={C.text} fontFamily={C.mono}>{p.code}</text>
              <text textAnchor="middle" y={nodeR + 15} fontSize="9" fill={isActive ? C.text : C.textFaint} fontFamily={C.mono} className="persona-node-label">{p.role.split(' ')[0].toUpperCase()}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

interface PersonaColumnProps { mode: ParticipantSelectionMode; participantCount: number; selectedIds: string[]; onMode: (mode: ParticipantSelectionMode) => void; onParticipantCount: (count: number) => void; onToggle: (id: string) => void }
export function PersonaColumn({ mode, participantCount, selectedIds, onMode, onParticipantCount, onToggle }: PersonaColumnProps) {
  return (
    <aside className="persona-column">
      <div className="persona-column-head">
        <div className="eyebrow">Available personas</div>
        <span className="persona-section-count">6 total</span>
      </div>
      <div className="persona-mode-toggle" role="group" aria-label="Participant selection method">
        <button type="button" className={mode === 'random' ? 'active' : ''} onClick={() => onMode('random')}>Random</button>
        <button type="button" className={mode === 'custom' ? 'active' : ''} onClick={() => onMode('custom')}>Choose personas</button>
      </div>
      {mode === 'random' ? <div className="persona-random-settings"><p className="persona-column-note">The backend uses the discussion seed to draw a reproducible random panel.</p><label htmlFor="participant-count">Agents in the room</label><select id="participant-count" className="input" value={participantCount} onChange={event => onParticipantCount(Number(event.target.value))}>{[3, 4, 5, 6].map(count => <option key={count} value={count}>{count} agents</option>)}</select></div>
        : <p className={`persona-column-note persona-custom-count ${selectedIds.length < 3 ? 'invalid' : ''}`}><strong>{selectedIds.length} selected</strong> · choose at least 3 so the discussion forms a graph.</p>}
      <div className="persona-grid persona-grid-stack">{PERSONAS.map(persona => <PersonaCard key={persona.id} persona={persona} selectable={mode === 'custom'} selected={selectedIds.includes(persona.id)} onToggle={() => onToggle(persona.id)} />)}</div>
    </aside>
  )
}

const personaKey = (id: string) => id.toLowerCase().replace(/^persona[_-]?/, '').replace(/[-\s]+/g, '_')
export function personaFor(agent: string): PersonaInfo | undefined {
  const key = personaKey(agent)
  return PERSONAS.find(p => p.id === key) || PERSONAS.find(p => p.code.toLowerCase() === key)
    || (key.length > 3 ? PERSONAS.find(p => key.includes(p.id) || p.id.includes(key)) : undefined)
}
/** Shorter card titles for the tablet/phone variants, where the full persona name would wrap. */
export const displayName = (agent: string) => personaFor(agent)?.name ?? prettyAgent(agent)
export const initials = (name: string) => { const parts = name.split(' ').filter(Boolean); return (parts.length > 1 ? parts[0][0] + parts[1][0] : name.slice(0, 2)).toUpperCase() }
