import { useId } from 'react'
import type { AgreementRecord, StanceRecord } from '../api'
import { displayName, initials, personaFor } from '../personas'
import { C } from '../theme'
import { prettyAgent } from '../utils'
import { analyticsColorFor } from './colors'

function polar(cx: number, cy: number, r: number, deg: number) {
  const a = ((deg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) }
}

function donutSlice(cx: number, cy: number, r: number, rInner: number, start: number, end: number) {
  if (end - start <= 0.04) return ''
  const large = end - start > 180 ? 1 : 0
  const o1 = polar(cx, cy, r, start), o2 = polar(cx, cy, r, end)
  const i1 = polar(cx, cy, rInner, end), i2 = polar(cx, cy, rInner, start)
  return `M ${o1.x.toFixed(2)} ${o1.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${o2.x.toFixed(2)} ${o2.y.toFixed(2)} L ${i1.x.toFixed(2)} ${i1.y.toFixed(2)} A ${rInner} ${rInner} 0 ${large} 0 ${i2.x.toFixed(2)} ${i2.y.toFixed(2)} Z`
}

export function AgreementGauge({ score }: { score: number | null | undefined }) {
  const pct = score == null ? null : Math.round(Math.max(0, Math.min(1, score)) * 100)
  const r = 38, cx = 46, cy = 46
  const circ = 2 * Math.PI * r
  const filled = pct == null ? 0 : (pct / 100) * circ
  const tone = pct == null ? C.borderStrong : pct >= 70 ? C.positive : pct >= 40 ? C.accent : C.negative
  return (
    <div className="gauge-wrap" aria-label={pct == null ? 'Agreement unavailable' : `Agreement ${pct} percent`}>
      <svg viewBox="0 0 92 92">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.surfaceSunken} strokeWidth="8" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={tone} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={`${filled.toFixed(2)} ${circ.toFixed(2)}`} transform={`rotate(-90 ${cx} ${cy})`} />
      </svg>
      <div className="gauge-center">
        <div className="gauge-value" style={{ color: tone }}>{pct == null ? '—' : `${pct}`}</div>
        <div className="gauge-unit">{pct == null ? 'n/a' : 'percent'}</div>
      </div>
    </div>
  )
}

/** Same gauge-wrap/gauge-center shell and fill mechanism as AgreementGauge
 *  (deliberately, so it reads as a sibling headline metric rather than a
 *  one-off widget). Net sentiment (-1 positive..+1, well: -1..1) is
 *  normalized onto the same 0-100 ring, with the tone and label reframed
 *  around a neutral midpoint instead of a zero floor. */
export function SentimentGauge({ net, total }: { net: number | null; total: number }) {
  const pct = net == null || total === 0 ? null : Math.round(((Math.max(-1, Math.min(1, net)) + 1) / 2) * 100)
  const r = 38, cx = 46, cy = 46
  const circ = 2 * Math.PI * r
  const filled = pct == null ? 0 : (pct / 100) * circ
  const tone = pct == null ? C.borderStrong : pct >= 58 ? C.positive : pct <= 42 ? C.negative : C.accent
  const readout = net == null || total === 0 ? '—' : `${net > 0 ? '+' : ''}${Math.round(net * 100)}`
  return (
    <div className="gauge-wrap" aria-label={pct == null ? 'Sentiment unavailable' : `Net sentiment ${pct} percent positive`}>
      <svg viewBox="0 0 92 92">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.surfaceSunken} strokeWidth="8" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={tone} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={`${filled.toFixed(2)} ${circ.toFixed(2)}`} transform={`rotate(-90 ${cx} ${cy})`} />
      </svg>
      <div className="gauge-center">
        <div className="gauge-value" style={{ color: tone }}>{readout}</div>
        <div className="gauge-unit">{pct == null ? 'n/a' : 'net tone'}</div>
      </div>
    </div>
  )
}

export function SignalWheel({ supportive, neutral, opposing, total }: { supportive: number; neutral: number; opposing: number; total: number }) {
  const safeTotal = Math.max(0, total)
  const slices = [
    { key: 'supportive', value: supportive, color: C.positive, label: 'Supportive' },
    { key: 'neutral', value: neutral, color: '#8A8474', label: 'Neutral' },
    { key: 'opposing', value: opposing, color: C.negative, label: 'Opposing' },
  ]
  let cursor = 0
  const arcs = slices.map(slice => {
    const portion = safeTotal ? (slice.value / safeTotal) * 360 : 0
    const start = cursor
    const end = cursor + portion
    cursor = end
    return { ...slice, start, end, pct: safeTotal ? Math.round((slice.value / safeTotal) * 100) : 0 }
  })

  return (
    <div className="signal-wheel-wrap">
      <div className="signal-wheel">
        <svg viewBox="0 0 128 128">
          <circle cx="64" cy="64" r="54" fill="none" stroke={C.surfaceSunken} strokeWidth="14" />
          {safeTotal > 0 && arcs.map(slice => slice.value > 0 && (
            <path key={slice.key} d={donutSlice(64, 64, 61, 43, slice.start, slice.end)} fill={slice.color} />
          ))}
        </svg>
        <div className="signal-wheel-center">
          <div className="signal-wheel-total">{safeTotal}</div>
          <div className="signal-wheel-label">signals</div>
        </div>
      </div>
      <div className="signal-wheel-legend">
        {arcs.map(slice => (
          <div key={slice.key}>
            <div className="signal-legend-row">
              <span className="signal-dot" style={{ background: slice.color }} />
              <span>{slice.label}</span>
              <strong>{slice.value}</strong>
            </div>
            <div className="signal-legend-track"><div className="signal-legend-fill" style={{ width: `${slice.pct}%`, background: slice.color }} /></div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function AgreementChart({ agreements }: { agreements: AgreementRecord[] }) {
  const uid = useId().replace(/[:]/g, '')
  const w = 720, h = 236
  const pad = { l: 40, r: 18, t: 28, b: 36 }
  const innerW = w - pad.l - pad.r
  const innerH = h - pad.t - pad.b
  const n = Math.max(agreements.length, 1)
  const colW = innerW / n
  const barW = Math.min(46, Math.max(18, colW * 0.42))
  const points = agreements.map((item, i) => {
    const score = item.score == null ? null : Math.max(0, Math.min(1, item.score))
    return {
      round: item.round,
      score,
      x: pad.l + colW * i + colW / 2,
      y: score == null ? null : pad.t + innerH * (1 - score),
      h: score == null ? 0 : innerH * score,
    }
  })
  const valid = points.filter((p): p is typeof p & { y: number; score: number } => p.y != null && p.score != null)
  const lineD = valid.map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
  const areaD = valid.length
    ? `M ${valid[0].x.toFixed(1)} ${(pad.t + innerH).toFixed(1)} ${valid.map(p => `L ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')} L ${valid[valid.length - 1].x.toFixed(1)} ${(pad.t + innerH).toFixed(1)} Z`
    : ''
  const ticks = [0, 0.25, 0.5, 0.75, 1]

  return (
    <div className="chart-frame">
      <svg className="chart-svg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Agreement by discussion round">
        <defs>
          <linearGradient id={`${uid}-area`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.sage} stopOpacity="0.28" />
            <stop offset="100%" stopColor={C.sage} stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id={`${uid}-bar`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.accent} />
            <stop offset="100%" stopColor="#C9844A" />
          </linearGradient>
        </defs>
        {ticks.map(t => {
          const y = pad.t + innerH * (1 - t)
          return (
            <g key={t}>
              <line x1={pad.l} x2={w - pad.r} y1={y} y2={y} stroke={C.border} strokeWidth="1" />
              <text x={pad.l - 8} y={y + 3} textAnchor="end" fontSize="10" fontFamily={C.mono} fill={C.textFaint}>{Math.round(t * 100)}</text>
            </g>
          )
        })}
        {areaD && <path d={areaD} fill={`url(#${uid}-area)`} />}
        {points.map((p, i) => (
          <rect key={p.round} className="chart-bar" x={p.x - barW / 2} y={pad.t + innerH - p.h} width={barW} height={p.h}
            fill={`url(#${uid}-bar)`} opacity="0.88" style={{ animationDelay: `${i * 70}ms` }} />
        ))}
        {lineD && <path d={lineD} fill="none" stroke={C.slate} strokeWidth="2.2" />}
        {valid.map(p => (
          <g key={`pt-${p.round}`}>
            <circle cx={p.x} cy={p.y} r="5.5" fill="#FAF8F3" stroke={C.slate} strokeWidth="2" />
            <text x={p.x} y={p.y - 10} textAnchor="middle" fontSize="10" fontFamily={C.mono} fontWeight="700" fill={C.text}>{Math.round(p.score * 100)}%</text>
          </g>
        ))}
        {points.map(p => (
          <text key={`lbl-${p.round}`} x={p.x} y={h - 12} textAnchor="middle" fontSize="10" fontFamily={C.mono} fill={C.textMuted}>R{p.round}</text>
        ))}
      </svg>
      <div className="chart-caption"><span>Round sequence</span><span>Pairwise alignment · 0–100%</span></div>
    </div>
  )
}

export function InfluenceChart({ influence, participants }: { influence: { agent_id: string; score?: number | null }[]; participants: string[] }) {
  const max = Math.max(0.01, ...influence.map(item => item.score || 0))
  return (
    <div className="influence-list">
      {influence.map(item => {
        const color = analyticsColorFor(item.agent_id, participants)
        const persona = personaFor(item.agent_id)
        const score = item.score ?? 0
        return (
          <div className="influence-row" key={item.agent_id}>
            <span className="influence-avatar" style={{ color, border: `1.5px solid ${color}`, background: `${color}18` }}>{persona?.code ?? initials(prettyAgent(item.agent_id))}</span>
            <div>
              <div className="influence-name">{displayName(item.agent_id)}</div>
              <div className="influence-track">
                <div className="influence-fill" style={{ width: `${Math.max(4, (score / max) * 100)}%`, background: `linear-gradient(90deg, ${color}, ${color}99)` }} />
              </div>
            </div>
            <span className="influence-score">{item.score == null ? '—' : score.toFixed(2)}</span>
          </div>
        )
      })}
    </div>
  )
}

function stanceKind(stance: number | null | undefined, candidate?: string | null) {
  if (!candidate && (stance == null)) return 'unresolved'
  if (stance == null) return 'neutral'
  if (stance > 0) return 'aligned'
  if (stance < 0) return 'alternative'
  return 'neutral'
}

export function TrajectoryBoard({ trajectories, participants }: { trajectories: Record<string, StanceRecord[]>; participants: string[] }) {
  const agents = Object.keys(trajectories)
  const rounds = [...new Set(Object.values(trajectories).flatMap(records => records.map(r => r.round)))].sort((a, b) => a - b)
  if (!agents.length || !rounds.length) return <div className="empty">Trajectory data will appear after the next completed round.</div>
  const columns = `minmax(168px,0.9fr) repeat(${rounds.length}, minmax(120px,1fr))`
  return (
    <div className="traj-board">
      <div className="traj-grid">
        <div className="traj-head" style={{ gridTemplateColumns: columns }}>
          <div className="traj-corner">Agent</div>
          {rounds.map(round => <div className="traj-round" key={round}>Round {round}</div>)}
        </div>
        {agents.map(agent => {
          const color = analyticsColorFor(agent, participants)
          const persona = personaFor(agent)
          const byRound = Object.fromEntries((trajectories[agent] || []).map(record => [record.round, record]))
          return (
            <div className="traj-row" key={agent} style={{ gridTemplateColumns: columns }}>
              <div className="traj-agent">
                <span className="influence-avatar" style={{ color, border: `1.5px solid ${color}`, background: `${color}18` }}>{persona?.code ?? initials(prettyAgent(agent))}</span>
                <div style={{ minWidth: 0 }}>
                  <div className="traj-agent-name">{displayName(agent)}</div>
                  <div className="traj-agent-role">{persona?.role ?? 'Participant'}</div>
                </div>
              </div>
              {rounds.map(round => {
                const record = byRound[round]
                const kind = record ? stanceKind(record.stance, record.candidate) : 'unresolved'
                return (
                  <div className={`traj-cell ${kind}`} key={`${agent}-${round}`}>
                    <span className="traj-candidate">{record?.candidate || 'No candidate'}</span>
                    {record?.candidate_changed ? <span className="traj-flag">Changed</span> : null}
                  </div>
                )
              })}
            </div>
          )
        })}
      </div>
      <div className="traj-legend">
        <span><i className="traj-swatch" style={{ background: 'rgba(92,122,84,.18)', borderColor: 'rgba(92,122,84,.32)' }} />Aligned</span>
        <span><i className="traj-swatch" style={{ background: 'rgba(64,84,90,.10)', borderColor: 'rgba(64,84,90,.22)' }} />Neutral</span>
        <span><i className="traj-swatch" style={{ background: 'rgba(168,78,69,.12)', borderColor: 'rgba(168,78,69,.30)' }} />Alternative</span>
        <span><i className="traj-swatch" style={{ background: C.surfaceSunken }} />Unresolved</span>
        <span><i className="traj-swatch" style={{ background: C.accent, borderColor: C.accent }} />Candidate changed</span>
      </div>
    </div>
  )
}

