import { Activity, BarChart3, GitBranch, Network, TrendingUp } from 'lucide-react'
import type { AgreementRecord, AnalyticsArtifact, DiscussionSnapshot, StanceRecord } from '../api'
import { C } from '../theme'
import { last } from '../utils'
import { AgreementChart, AgreementGauge, InfluenceChart, SentimentGauge, SignalWheel, TrajectoryBoard } from './Charts'
import { InteractionGraph } from './InteractionGraph'

export function AnalyticsView({ snapshot, analytics }: { snapshot: DiscussionSnapshot; analytics: AnalyticsArtifact | null }) {
  const final = analytics?.final_analysis
  const latest = last(analytics?.round_analyses || [])
  const agreements: AgreementRecord[] = final?.agreement || analytics?.round_analyses.map(item => item.agreement).filter((item): item is AgreementRecord => item != null) || []
  const latestAgreement = last(agreements)?.score
  const influence = Object.values(final?.influence || {}).sort((a, b) => (b.score || 0) - (a.score || 0))
  const sentiment = final?.sentiment || latest?.sentiment || []
  const evaluations = sentiment.flatMap(r => r.evaluations || [])
  const positive = evaluations.filter(i => i.label === 'positive' || (i.score || 0) > .15).length
  const negative = evaluations.filter(i => i.label === 'negative' || (i.score || 0) < -.15).length
  const neutral = Math.max(0, evaluations.length - positive - negative)
  const netSentiment = evaluations.length ? (positive - negative) / evaluations.length : null
  const trajectories: Record<string, StanceRecord[]> = final?.opinion_change || (latest ? Object.fromEntries(Object.entries(latest.stances).map(([key, value]) => [key, [value]])) : {})
  const switches = Object.values(trajectories).flat().filter(i => i.candidate_changed)
  const participants = snapshot.graph?.agents?.length ? snapshot.graph.agents : snapshot.participants

  if (!analytics) return <div className="panel"><div className="empty"><BarChart3 className="pulse" size={30} /><strong>Analytics begin after round one</strong>This page will update as the discussion produces structured signals.</div></div>

  return (
    <div className="analytics-grid fade-up">
      <div className="metric-card quarter">
        <div className="eyebrow">Latest agreement</div>
        <div className="metric-hero">
          <AgreementGauge score={latestAgreement} />
          <div className="metric-hero-copy">
            <div className="metric-value" style={{ marginTop: 0 }}>{latestAgreement == null ? '—' : `${Math.round(latestAgreement * 100)}%`}</div>
            <div className="metric-label">Pairwise candidate alignment across the seated panel.</div>
          </div>
        </div>
      </div>
      <div className="metric-card quarter">
        <div className="eyebrow">Position changes</div>
        <div className="metric-value">{switches.length}</div>
        <div className="metric-label">Committed candidate switches recorded in the trajectory.</div>
        {switches.length > 0 && (
          <div className="change-pills">
            {switches.slice(0, 4).map((item, index) => (
              <span className="change-pill" key={`${item.round}-${index}`}><i />R{item.round}{item.candidate ? ` · ${item.candidate}` : ''}</span>
            ))}
          </div>
        )}
      </div>
      <div className="metric-card quarter">
        <div className="eyebrow">Sentiment</div>
        <div className="metric-hero">
          <SentimentGauge net={netSentiment} total={evaluations.length} />
          <div className="metric-hero-copy">
            <div className="metric-value" style={{ marginTop: 0 }}>{netSentiment == null ? '—' : netSentiment > .15 ? 'Positive' : netSentiment < -.15 ? 'Negative' : 'Mixed'}</div>
            <div className="metric-label">Net tone across {evaluations.length} evaluated signal{evaluations.length === 1 ? '' : 's'}.</div>
          </div>
        </div>
      </div>
      <div className="metric-card quarter analytics-signal-card">
        <div className="eyebrow">Evaluated signals</div>
        <SignalWheel supportive={positive} neutral={neutral} opposing={negative} total={evaluations.length} />
      </div>
      <div className="metric-card wide">
        <div className="panel-title"><TrendingUp size={16} />Agreement by round</div>
        {agreements.length ? <AgreementChart agreements={agreements} /> : <div className="empty">No agreement scores yet.</div>}
      </div>
      <div className="metric-card">
        <div className="panel-title"><Activity size={16} />Influence</div>
        {influence.length
          ? <InfluenceChart influence={influence} participants={participants} />
          : <div className="empty" style={{ padding: '34px 8px' }}>Influence is calculated when the discussion ends.</div>}
      </div>
      <div className="metric-card full">
        <div className="panel-title"><Network size={16} />Interaction Graph</div>
        <InteractionGraph snapshot={snapshot} />
      </div>
      <div className="metric-card full">
        <div className="panel-title"><GitBranch size={16} />Opinion trajectory</div>
        <TrajectoryBoard trajectories={trajectories} participants={participants} />
      </div>
      {final?.issues?.length ? <div className="metric-card full"><div className="eyebrow">Data notes</div><ul className="plain-list">{final.issues.map(i => <li key={i}>{i}</li>)}</ul></div> : null}
      <div className="metric-card full" style={{ color: C.textMuted, fontSize: 12 }}>Analytics status: <strong style={{ color: C.text }}>{analytics.status}</strong> · Discussion status: <strong style={{ color: C.text }}>{snapshot.status}</strong></div>
    </div>
  )
}
