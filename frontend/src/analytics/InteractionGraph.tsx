import { useEffect, useId, useMemo, useState } from 'react'
import type { DiscussionSnapshot } from '../api'
import { displayName, initials, personaFor } from '../personas'
import { C } from '../theme'
import { allMessages, prettyAgent } from '../utils'
import { analyticsColorFor } from './colors'

const TAU = Math.PI * 2

function orderCycle(agents: string[], rawEdges: unknown[]) {
  if (agents.length < 3) return agents
  const neighbors: Record<string, string[]> = Object.fromEntries(agents.map(agent => [agent, []]))
  rawEdges.forEach(edge => {
    let source = '', target = ''
    if (Array.isArray(edge)) {
      source = typeof edge[0] === 'string' ? edge[0] : ''
      target = typeof edge[1] === 'string' ? edge[1] : ''
    } else if (edge && typeof edge === 'object') {
      const e = edge as Record<string, unknown>
      source = typeof e.source === 'string' ? e.source : typeof e.from === 'string' ? e.from : ''
      target = typeof e.target === 'string' ? e.target : typeof e.to === 'string' ? e.to : ''
    }
    if (source && target && neighbors[source] && !neighbors[source].includes(target)) neighbors[source].push(target)
  })
  const start = agents.find(agent => neighbors[agent]?.length === 1) || agents[0]
  const ordered: string[] = []
  const seen = new Set<string>()
  let current: string | undefined = start
  while (current && !seen.has(current)) {
    ordered.push(current)
    seen.add(current)
    current = neighbors[current]?.find(agent => !seen.has(agent)) || agents.find(agent => !seen.has(agent))
  }
  return ordered.length === agents.length ? ordered : agents
}

function directedStrength(messages: Array<{ sender: string; recipients: string[] }>, source: string, target: string, rawEdges: unknown[]) {
  let strength = 0
  messages.forEach(message => {
    if (message.sender === source && message.recipients?.includes(target)) strength += 1
  })
  rawEdges.forEach(edge => {
    if (Array.isArray(edge) && edge[0] === source && edge[1] === target && typeof edge[2] === 'number') strength = Math.max(strength, edge[2])
    if (edge && typeof edge === 'object' && !Array.isArray(edge)) {
      const e = edge as Record<string, unknown>
      const from = typeof e.source === 'string' ? e.source : typeof e.from === 'string' ? e.from : ''
      const to = typeof e.target === 'string' ? e.target : typeof e.to === 'string' ? e.to : ''
      const raw = typeof e.weight === 'number' ? e.weight : typeof e.strength === 'number' ? e.strength : typeof e.count === 'number' ? e.count : 0
      if (from === source && to === target) strength = Math.max(strength, raw)
    }
  })
  return strength
}

export function InteractionGraph({ snapshot }: { snapshot: DiscussionSnapshot }) {
  const sourceAgents = snapshot.graph?.agents?.length ? snapshot.graph.agents : snapshot.participants
  const rawEdges = (snapshot.graph?.edges || []) as unknown[]
  const messages = allMessages(snapshot) as Array<{ sender: string; recipients: string[] }>
  const agents = useMemo(() => orderCycle(sourceAgents, rawEdges), [sourceAgents.join('|'), rawEdges.map(edge => JSON.stringify(edge)).join('|')])
  const uid = 'ig' + useId().replace(/[:]/g, '')
  const [activeHop, setActiveHop] = useState(0)
  const [focus, setFocus] = useState<string | null>(null)

  const hops = useMemo(() => {
    const n = agents.length
    if (n < 2) return []
    return agents.map((source, i) => {
      const target = agents[(i + 1) % n]
      return { source, target, strength: directedStrength(messages, source, target, rawEdges), index: i }
    })
  }, [agents.join('|'), messages.map(m => `${m.sender}:${m.recipients?.join(',')}`).join('|'), rawEdges.map(edge => JSON.stringify(edge)).join('|')])

  useEffect(() => {
    if (hops.length < 2 || focus) return
    const timer = window.setInterval(() => setActiveHop(i => (i + 1) % hops.length), 1800)
    return () => window.clearInterval(timer)
  }, [hops.length, focus])

  if (agents.length < 2) return <div className="interaction-empty">Interaction data will appear when the discussion produces agent relationships.</div>

  const width = 680, height = 360, cx = width / 2, cy = height / 2
  const rx = 132, ry = 132
  const nodeR = 22
  const n = agents.length
  const angleAt = (i: number) => -Math.PI / 2 + (TAU * i) / n
  const pos = agents.map((agent, i) => {
    const angle = angleAt(i)
    return { agent, angle, x: cx + rx * Math.cos(angle), y: cy + ry * Math.sin(angle) }
  })
  const currentHop = focus ? Math.max(0, agents.indexOf(focus)) : activeHop

  const arcFor = (from: number, to: number) => {
    let span = to - from
    if (span <= 0) span += TAU
    const pad = 0.38
    const a0 = from + pad, a1 = from + span - pad
    const p0x = cx + rx * Math.cos(a0), p0y = cy + ry * Math.sin(a0)
    const p1x = cx + rx * Math.cos(a1), p1y = cy + ry * Math.sin(a1)
    const large = a1 - a0 > Math.PI ? 1 : 0
    const tx = -rx * Math.sin(a1), ty = ry * Math.cos(a1)
    const tl = Math.hypot(tx, ty) || 1
    const ux = tx / tl, uy = ty / tl
    const midA = (a0 + a1) / 2
    return {
      d: `M ${p0x.toFixed(1)} ${p0y.toFixed(1)} A ${rx} ${ry} 0 ${large} 1 ${p1x.toFixed(1)} ${p1y.toFixed(1)}`,
      tipX: p1x, tipY: p1y, ux, uy,
      labelX: cx + (rx + 18) * Math.cos(midA),
      labelY: cy + (ry + 18) * Math.sin(midA),
    }
  }

  return (
    <div className="interaction-graph">
      <div className="interaction-graph-stage">
        <svg className="interaction-graph-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="One-way cyclic agent interaction graph">
          <defs>
            <radialGradient id={`${uid}-core`}><stop offset="0%" stopColor="#FFFaf3" /><stop offset="100%" stopColor={C.surfaceSunken} /></radialGradient>
            <marker id={`${uid}-arrow`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill={C.slate} /></marker>
            {hops.map(hop => (
              <marker key={hop.index} id={`${uid}-arrow-${hop.index}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" markerUnits="userSpaceOnUse" orient="auto">
                <path d="M0 0 L10 5 L0 10 z" fill={analyticsColorFor(hop.source, agents)} />
              </marker>
            ))}
          </defs>
          <ellipse className="interaction-orbit" cx={cx} cy={cy} rx={rx} ry={ry} />
          {hops.map(hop => {
            const from = angleAt(hop.index)
            const to = angleAt(hop.index + 1)
            const arc = arcFor(from, to)
            const color = analyticsColorFor(hop.source, agents)
            const live = currentHop === hop.index
            const widthStroke = 2.8
            return (
              <g key={`${hop.source}-${hop.target}`} opacity={focus && !live ? 0.28 : 1}>
                <path className="interaction-edge" d={arc.d} stroke={`${color}22`} strokeWidth="9" />
                <path className="interaction-edge" d={arc.d} stroke={color} strokeWidth={widthStroke} opacity={live ? 1 : 0.78} markerEnd={`url(#${uid}-arrow-${hop.index})`} />
                <path className="interaction-edge-flow" d={arc.d} stroke={color} strokeWidth="1.6" opacity={live ? 0.95 : 0.35} />
                {live && <path className="interaction-comet" d={arc.d} stroke={color} strokeWidth="6" />}
              </g>
            )
          })}
          {pos.map((p, i) => {
            const color = analyticsColorFor(p.agent, agents)
            const persona = personaFor(p.agent)
            const live = currentHop === i || currentHop === (i - 1 + n) % n
            return (
              <g key={p.agent} className="interaction-node" transform={`translate(${p.x},${p.y})`} onClick={() => setFocus(current => current === p.agent ? null : p.agent)} opacity={focus && focus !== p.agent && !live ? 0.38 : 1}>
                <circle className="interaction-node-ring" r={nodeR + 8} stroke={color} style={{ animationDelay: `${i * 0.35}s` }} />
                <circle className="interaction-node-core" r={nodeR} fill={`url(#${uid}-core)`} stroke={color} />
                <text className="interaction-node-code" textAnchor="middle" dy="4" fill={color}>{persona?.code ?? initials(prettyAgent(p.agent))}</text>
                <text className="interaction-node-name" textAnchor="middle" y={nodeR + 18}>{displayName(p.agent)}</text>
                <text className="interaction-node-role" textAnchor="middle" y={nodeR + 32}>{persona?.role ?? 'Participant'}</text>
              </g>
            )
          })}
        </svg>
      </div>
      <div className="interaction-graph-footer">
        <div className="interaction-legend">
          <span className="interaction-legend-item"><i className="interaction-line medium" />One-way flow</span>
          <span className="interaction-legend-item"><i className="interaction-line strong" />Interaction strength</span>
          
        </div>
        <div className="interaction-summary"><strong>{agents.length}</strong> agents <span>·</span> <strong>{hops.length}</strong> directed hops</div>
      </div>
    </div>
  )
}
