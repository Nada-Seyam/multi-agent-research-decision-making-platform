import { type CSSProperties, useEffect, useMemo, useRef, useState } from 'react'
import { ArrowRight, MessageSquare, Network } from 'lucide-react'
import type { DiscussionSnapshot } from './api'
import { CRITERIA, type PersonaInfo, PriorityGlyph, displayName, initials, personaFor } from './personas'
import { C, FLOW } from './theme'
import { agentColor, allMessages, last, prettyAgent } from './utils'

interface Pt { x: number; y: number }
type AgentRole = 'idle' | 'speaking' | 'receiving'
/** full: desktop cards · lite: no role/glyph (tablet) · mini: avatar + name only (phone) */
type CardVariant = 'full' | 'lite' | 'mini'
type EdgePath = { d: string; head: string } | null

const TAU = Math.PI * 2
const HEAD_LEN = 12, HEAD_HALF = 6
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const num = (n: number) => n.toFixed(1)

const SHORT_NAMES: Record<string, string> = { safety_governance_advocate: 'Safety Advocate', product_user_advocate: 'Product Advocate', cost_sustainability_analyst: 'Cost Analyst', practical_architect: 'Practical Architect', strategic_consensus_builder: 'Consensus Builder' }
function arrowHead(tip: Pt, u: Pt) {
  const bx = tip.x - u.x * HEAD_LEN, by = tip.y - u.y * HEAD_LEN, nx = -u.y, ny = u.x
  return `M ${num(tip.x)} ${num(tip.y)} L ${num(bx + nx * HEAD_HALF)} ${num(by + ny * HEAD_HALF)} L ${num(bx - nx * HEAD_HALF)} ${num(by - ny * HEAD_HALF)} Z`
}

interface RingArgs { cx: number; cy: number; rx: number; ry: number; from: number; to: number; a: Pt; b: Pt; hw: number; hh: number }
/** Clockwise elliptical arc between two neighbouring cards, trimmed to the card borders. */
function ringEdge({ cx, cy, rx, ry, from, to, a, b, hw, hh }: RingArgs): EdgePath {
  const at = (t: number): Pt => ({ x: cx + rx * Math.cos(from + t), y: cy + ry * Math.sin(from + t) })
  const span = (((to - from) % TAU) + TAU) % TAU || TAU
  const inside = (p: Pt, c: Pt) => Math.abs(p.x - c.x) <= hw && Math.abs(p.y - c.y) <= hh
  const step = 0.004
  let s0 = 0
  while (s0 < span && inside(at(s0), a)) s0 += step
  let s1 = span
  while (s1 > s0 && inside(at(s1), b)) s1 -= step
  if (s1 - s0 < 0.05) return null
  const p0 = at(s0), tip = at(s1), ang = from + s1
  const tx = -rx * Math.sin(ang), ty = ry * Math.cos(ang), tl = Math.hypot(tx, ty) || 1
  const u = { x: tx / tl, y: ty / tl }
  const end = { x: tip.x - u.x * HEAD_LEN * 0.7, y: tip.y - u.y * HEAD_LEN * 0.7 }
  return { d: `M ${num(p0.x)} ${num(p0.y)} A ${num(rx)} ${num(ry)} 0 ${s1 - s0 > Math.PI ? 1 : 0} 1 ${num(end.x)} ${num(end.y)}`, head: arrowHead(tip, u) }
}

/** How a message travels the one-way loop: number of hops, when (ms) each recipient is reached, and how long the message stays on screen. */
function routeFor(agents: string[], sender: string, recipients: string[]) {
  const n = agents.length, from = agents.indexOf(sender)
  const arrive: Record<string, number> = {}
  let hops = 0
  if (from >= 0 && n > 1) recipients.forEach(r => {
    const to = agents.indexOf(r)
    if (to < 0 || to === from) return
    const h = (to - from + n) % n
    hops = Math.max(hops, h)
    arrive[r] = FLOW.delay + h * FLOW.hop - 100
  })
  return { from, hops, arrive, life: FLOW.delay + Math.max(1, hops) * FLOW.hop + FLOW.hold }
}

/** Glowing dot that travels along one hop. SMIL timelines start when the SVG document loads, so the
 *  animations are declared `indefinite` and started explicitly each time a packet mounts. */
function Packet({ d, delay }: { d: string; delay: number }) {
  const ref = useRef<SVGGElement>(null)
  useEffect(() => {
    ref.current?.querySelectorAll<SVGAnimationElement>('set, animateMotion').forEach(a => a.beginElementAt(delay / 1000))
  }, [])
  const motion = <animateMotion path={d} begin="indefinite" dur={`${FLOW.hop}ms`} fill="freeze" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines=".45 0 .2 1" />
  const visible = <set attributeName="opacity" to="1" begin="indefinite" dur={`${FLOW.hop}ms`} />
  return (
    <g ref={ref} className="ag-packet-wrap">
      <circle className="ag-packet-halo" r="11" opacity="0" fillOpacity="0.2">{visible}{motion}</circle>
      <circle className="ag-packet" r="4.5" opacity="0">{visible}{motion}</circle>
    </g>
  )
}

interface AgentCardProps { agent: string; x: number; y: number; w: number; h: number; variant: CardVariant; color: string; fromColor: string; persona?: PersonaInfo; role: AgentRole; arrive: number; selected: boolean; pulseKey: string; onSelect: () => void }
function AgentCard({ agent, x, y, w, h, variant, color, fromColor, persona, role, arrive, selected, pulseKey, onSelect }: AgentCardProps) {
  const full = variant === 'full', mini = variant === 'mini'
  const name = !full && persona && SHORT_NAMES[persona.id] || displayName(agent)
  const code = persona?.code ?? initials(prettyAgent(agent))
  const drivers = persona && !mini ? [...CRITERIA].sort((p, q) => persona.priorities[q.id] - persona.priorities[p.id]).slice(0, 2) : []
  return (
    <button type="button" className={`ag-card ${role} ${variant}${selected ? ' selected' : ''}`}
      style={{ left: x - w / 2, top: y - h / 2, width: w, height: h, '--c': color, '--from': fromColor, '--arrive': `${arrive}ms` } as CSSProperties}
      onClick={onSelect} aria-pressed={selected} aria-label={`${name}${role !== 'idle' ? `, ${role}` : ''}`}>
      {role === 'speaking' && <span key={pulseKey} className="ag-ripple" aria-hidden />}
      {role === 'receiving' && <span key={pulseKey} className="ag-arrive" aria-hidden />}
      <span className="ag-tag speaking" aria-hidden><span className="ag-wave"><i /><i /><i /></span>Speaking</span>
      <span className="ag-tag receiving" aria-hidden>Receiving</span>
      <span className="ag-top">
        <span className="ag-avatar" style={{ background: `${color}14` }}>{code}</span>
        <span className="ag-who"><span className="ag-name">{name}</span>{full && <span className="ag-role">{persona?.role ?? 'Participant'}</span>}</span>
      </span>
      {!mini && drivers.length > 0 && persona && <>
        <span className="ag-rule" />
        <span className="ag-drivers">
          <span className="ag-row"><span className="ag-eyebrow">Key priorities</span>{full && <PriorityGlyph persona={persona} />}</span>
          <span className="ag-chips">{drivers.map(d => <span key={d.id} className="ag-chip">{d.label}</span>)}</span>
        </span>
      </>}
    </button>
  )
}

type DiscussionMessage = DiscussionSnapshot['rounds'][number]['messages'][number]
type ChatMessage = DiscussionMessage & { content?: string; text?: string; message?: string; summary?: string }

/** Right-hand column of the Agent communication panel: the actual transcript,
 *  grouped by round, rendered as a chat feed. Reads directly from
 *  `snapshot.rounds` — no separate API call — and relies on stable message
 *  keys so only newly-arrived messages play their entrance animation; older
 *  ones stay put. Auto-scrolls to the newest message as rounds complete. */
const CHAT_CLAMP_CHARS = 200

/** Convert model-authored Markdown into readable transcript text. The live
 *  transcript is intentionally plain text, so formatting tokens should never
 *  leak into a bubble regardless of which round produced the message. */
function cleanTranscriptText(value: string) {
  return value
    .replace(/&#x20;|&#32;|&nbsp;/gi, ' ')
    .replace(/\\([*_#`~$])/g, '$1')
    .replace(/\r\n?/g, '\n')
    .replace(/\*+/g, '')
    .replace(/_{2,}/g, '')
    .replace(/^\s*#{1,6}\s*/gm, '')
    .replace(/`+/g, '')
    .replace(/\$([^$\n]+)\$/g, '$1')
    .replace(/~{2}/g, '')
    .replace(/^\s*[-+]\s+/gm, '')
    .replace(/\b([A-Za-z0-9]+(?:_[A-Za-z0-9]+)+)\b/g, match => match.replace(/_/g, ' '))
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/** One transcript message rendered as a chat bubble, tinted to the sender's
 *  persona color. Long messages are clamped to a few lines with a "Read
 *  more" toggle so the feed stays scannable; short ones render in full with
 *  no button. Each bubble owns its own expanded/collapsed state. */
function ChatBubble({ m, color, code, name, recipientsLabel, style }: { m: ChatMessage; color: string; code: string; name: string; recipientsLabel: string; style?: CSSProperties }) {
  const [expanded, setExpanded] = useState(false)
  const rawBody = m.content ?? m.text ?? m.message ?? m.summary
  const body = rawBody ? cleanTranscriptText(rawBody) : undefined
  const long = !!body && body.length > CHAT_CLAMP_CHARS
  return (
    <div className="ag-chat-bubble" style={style}>
      <span className="ag-chat-avatar" style={{ background: `${color}18`, color, border: `1px solid ${color}` }}>{code}</span>
      <div className="ag-chat-body">
        <div className="ag-chat-meta"><strong style={{ color }}>{name}</strong><span className="ag-chat-to"><ArrowRight size={10} />{recipientsLabel}</span></div>
        {body && (
          <div className="ag-chat-msg" style={{ background: `linear-gradient(135deg, ${color}22, ${color}0d)`, border: `1px solid ${color}45`, borderLeft: `3px solid ${color}` }}>
            <p className={`ag-chat-text${expanded ? ' expanded' : ''}`}>{body}</p>
            {long && <button type="button" className="ag-chat-more" style={{ color }} onClick={() => setExpanded(v => !v)}>{expanded ? 'Show less' : 'Read more…'}</button>}
          </div>
        )}
      </div>
    </div>
  )
}

function MessageChat({ snapshot }: { snapshot: DiscussionSnapshot }) {
  const listRef = useRef<HTMLDivElement>(null)
  const sourceAgents = snapshot.graph.agents.length ? snapshot.graph.agents : snapshot.participants
  const total = allMessages(snapshot).length

  /* Keep the feed live without hijacking the user's scroll position.
     If the user is reading older messages, new messages arrive in the feed
     but the viewport stays exactly where they left it. */
  useEffect(() => {
    const el = listRef.current
    if (!el) return

    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    const wasFollowingLive = distanceFromBottom < 90

    if (wasFollowingLive) {
      window.requestAnimationFrame(() => {
        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
      })
    }
  }, [total])

  return (
    <div className="ag-chat">
      <div className="ag-chat-head"><MessageSquare size={13} />Live transcript<span className="ag-chat-count">{total}</span></div>
      <div className="ag-chat-list" ref={listRef}>
        {snapshot.rounds.length === 0 && <div className="ag-chat-empty">Messages will appear here as each round completes.</div>}
        {snapshot.rounds.map((round, ri) => {
          const messages = (round.messages as ChatMessage[]).slice().sort((a, b) => a.sequence - b.sequence)
          if (!messages.length) return null
          return (
            <div key={ri} className="ag-chat-round">
              <div className="ag-chat-round-label"><span />Round {ri + 1}<span /></div>
              {messages.map(m => {
                const persona = personaFor(m.sender)
                const color = persona?.marker ?? agentColor(m.sender, sourceAgents)
                const recipientsLabel = m.recipients.length ? m.recipients.map(displayName).join(', ') : 'the room'
                const messagePosition = Math.max(0, messages.findIndex(item => item.sequence === m.sequence))
                return (
                  <ChatBubble key={m.sequence} m={m} color={color} code={persona?.code ?? initials(prettyAgent(m.sender))}
                    name={displayName(m.sender)} recipientsLabel={recipientsLabel}
                    style={{ '--chat-delay': `${Math.min(messagePosition * 55, 220)}ms` } as CSSProperties} />
                )
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function AgentGraph({ snapshot, selected, onSelect }: { snapshot: DiscussionSnapshot; selected: string | null; onSelect: (a: string) => void }) {
  const sourceAgents = snapshot.graph.agents.length ? snapshot.graph.agents : snapshot.participants
  const agents = useMemo(() => {
    if (sourceAgents.length < 3) return sourceAgents
    const neighbors = Object.fromEntries(sourceAgents.map(agent => [agent, new Set<string>()]))
    snapshot.graph.edges.forEach(([source, target]) => { neighbors[source]?.add(target); neighbors[target]?.add(source) })
    const start = sourceAgents.find(agent => neighbors[agent]?.size === 1) || sourceAgents[0]
    const ordered: string[] = [], seen = new Set<string>()
    let current: string | undefined = start
    while (current && !seen.has(current)) {
      ordered.push(current); seen.add(current)
      current = [...(neighbors[current] || [])].find(agent => !seen.has(agent))
    }
    return ordered.length === sourceAgents.length ? ordered : sourceAgents
  }, [sourceAgents.join('|'), snapshot.graph.edges.map(edge => edge.join('>')).join('|')])

  /* ---- measure the stage so the ring, cards and arrows share one pixel space ---- */
  const stageRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(960)
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const measure = () => setWidth(Math.round(el.clientWidth) || 960)
    measure()
    if (typeof ResizeObserver === 'undefined') { window.addEventListener('resize', measure); return () => window.removeEventListener('resize', measure) }
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const layout = useMemo(() => {
    const n = agents.length
    const variant: CardVariant = width >= 900 ? 'full' : width >= 620 ? 'lite' : 'mini'
    const cardW = variant === 'full' ? clamp(width * 0.23, 216, 272) : variant === 'lite' ? clamp(width * 0.27, 190, 224) : clamp(width * 0.37, 124, 172)
    const cardH = variant === 'full' ? 146 : variant === 'lite' ? 118 : 64
    const rx = Math.max(0, (width - cardW) / 2 - (variant === 'mini' ? 8 : 16))
    let ry = variant === 'full' ? clamp(width * 0.19, 180, 220) : variant === 'lite' ? clamp(width * 0.21, 140, 180) : 112
    if (n === 2) ry *= 0.85
    if (n >= 5) ry = Math.max(ry, cardH * 1.15)
    const height = Math.round(2 * ry + cardH + (variant === 'full' ? 44 : 30))
    const cx = width / 2, cy = height / 2
    const angle = (i: number) => -Math.PI / 2 + (TAU * i) / n
    const pos: Record<string, Pt> = Object.fromEntries(agents.map((agent, i) => [agent, n === 1 ? { x: cx, y: cy } : { x: cx + rx * Math.cos(angle(i)), y: cy + ry * Math.sin(angle(i)) }]))
    const pad = variant === 'mini' ? 6 : 10
    const hw = cardW / 2 + pad, hh = cardH / 2 + pad
    /* ring[i] is the one-way arrow from agents[i] to the next agent clockwise */
    const ring: EdgePath[] = n < 2 ? [] : agents.map((source, i) => {
      const next = agents[(i + 1) % n]
      return ringEdge({ cx, cy, rx, ry, from: angle(i), to: angle(i + 1), a: pos[source], b: pos[next], hw, hh })
    })
    return { n, variant, cardW, cardH, height, cx, cy, rx, ry, pos, ring, hubWidth: Math.min(320, 2 * rx - cardW - 28) }
  }, [width, agents.join('|')])

  /* ---- message replay: each message stays as long as its trip around the loop takes ---- */
  const messages = allMessages(snapshot)
  const [messageIndex, setMessageIndex] = useState(0)
  const previousMessageCount = useRef(0)
  useEffect(() => {
    if (messages.length > previousMessageCount.current) setMessageIndex(previousMessageCount.current)
    else if (messageIndex >= messages.length) setMessageIndex(0)
    previousMessageCount.current = messages.length
  }, [messages.length, messageIndex])
  const message = messages[messageIndex] || last(messages)
  const messageKey = `${messageIndex}:${message?.sequence ?? 'none'}`
  const route = message ? routeFor(agents, message.sender, message.recipients) : null
  const life = route?.life ?? FLOW.delay + FLOW.hop + FLOW.hold
  useEffect(() => {
    if (messages.length < 2) return
    const timer = window.setTimeout(() => setMessageIndex(current => (current + 1) % messages.length), life)
    return () => window.clearTimeout(timer)
  }, [messageIndex, messages.length, life])
  const recipients = message?.recipients.map(displayName).join(' & ')

  const colorFor = (agent: string) => personaFor(agent)?.marker ?? agentColor(agent, sourceAgents)
  const senderColor = message ? colorFor(message.sender) : C.accent
  const roleOf = (agent: string): AgentRole => !message ? 'idle' : message.sender === agent ? 'speaking' : message.recipients.includes(agent) ? 'receiving' : 'idle'
  const hubInside = layout.variant !== 'mini' && layout.hubWidth >= 230

  const hub = (
    <div className="ag-hub-inner">
      {message
        ? <div className="ag-hub-line" key={messageKey}><div className="ag-hub-from" style={{ color: senderColor }}>{displayName(message.sender)}</div><div className="ag-hub-to"><ArrowRight size={13} />{recipients || 'the room'}</div></div>
        : <div className="ag-hub-line"><div className="ag-hub-from">Agents are preparing</div></div>}
    </div>
  )

  return (
    <div className="panel ag-panel">
      <div className="panel-head">
        <div className="panel-title"><Network size={16} />Agent communication</div>
        <div className="ag-legend"><span><i style={{ background: senderColor }} />Speaking</span><span><i style={{ background: C.slate }} />Receiving</span></div>
      </div>
      <div className="ag-body">
      <div className="ag-stage" ref={stageRef} style={{ height: layout.height }}>
        <svg className="ag-svg" viewBox={`0 0 ${width} ${layout.height}`} role="img" aria-label="Agent communication graph">
          {layout.n > 1 && <ellipse className="ag-track" cx={layout.cx} cy={layout.cy} rx={layout.rx} ry={layout.ry} />}
          {layout.ring.map((g, i) => g && <g key={i}><path className="ag-edge" d={g.d} /><path className="ag-arrow" d={g.head} /></g>)}
          {route && Array.from({ length: route.hops }, (_, k) => {
            const g = layout.ring[(route.from + k) % layout.n]
            if (!g) return null
            const start = FLOW.delay + k * FLOW.hop
            const timing = { '--c': senderColor, '--life': `${route.life}ms`, '--t': `${FLOW.hop}ms`, '--d': `${start}ms`, '--dh': `${start + FLOW.hop - 120}ms` } as CSSProperties
            return (
              <g key={`${messageKey}-${k}`} className="ag-live" style={timing}>
                <path className="ag-live-glow" d={g.d} pathLength={100} />
                <path className="ag-live-line" d={g.d} pathLength={100} />
                <path className="ag-live-comet" d={g.d} pathLength={100} />
                <path className="ag-live-head" d={g.head} />
                <Packet d={g.d} delay={start} />
              </g>
            )
          })}
        </svg>
        {agents.map(agent => { const p = layout.pos[agent]; return <AgentCard key={agent} agent={agent} x={p.x} y={p.y} w={layout.cardW} h={layout.cardH} variant={layout.variant} color={colorFor(agent)} fromColor={senderColor} persona={personaFor(agent)} role={roleOf(agent)} arrive={route?.arrive[agent] ?? 0} selected={selected === agent} pulseKey={messageKey} onSelect={() => onSelect(agent)} /> })}
        {hubInside && <div className="ag-hub" style={{ width: layout.hubWidth }}>{hub}</div>}
      </div>
      <MessageChat snapshot={snapshot} />
      </div>
      {!hubInside && <div className="ag-caption">{hub}</div>}
    </div>
  )
}
