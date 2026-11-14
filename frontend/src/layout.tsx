import type { FormEvent } from 'react'
import { Activity, AlertCircle, ArrowRight, BarChart3, Clock3, FileText, GitBranch, MessageSquare, PanelLeftClose, PanelLeftOpen, RotateCcw, ShieldCheck, Users } from 'lucide-react'
import type { AnalyticsArtifact, DiscussionReplay as ReplayTimeline, DiscussionSnapshot, ParticipantSelectionMode } from './api'
import { AnalyticsView } from './analytics/AnalyticsView'
import { ConclusionView } from './ConclusionView'
import { DossierPreparation, preparationPercent } from './DossierPreparation'
import { DiscussionReplay } from './DiscussionReplay'
import { PersonaColumn, PersonaTopologyPreview } from './personas'
import { C } from './theme'
import { topicText, type View } from './utils'

/** Wordless brand mark: a faceted gem — many flat facets catching light
 *  differently, resolving into one crystal — a visual echo of the app's
 *  own idea (many agent perspectives converging on one decision).
 *  Rendered with per-facet gradients, a specular glint and a soft cast
 *  shadow for real dimensionality. Sized to sit inside the .brand-mark
 *  frame. */
function BrandMark() {
  const uid = 'bm'
  return (
    <svg width="24" height="24" viewBox="0 0 24 24">
      <defs>
        <linearGradient id={`${uid}-table`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#F0B79B" />
          <stop offset="100%" stopColor="#DE8C68" />
        </linearGradient>
        <linearGradient id={`${uid}-cl`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#E2916A" />
          <stop offset="100%" stopColor={C.accent} />
        </linearGradient>
        <linearGradient id={`${uid}-cr`} x1="1" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={C.accent} />
          <stop offset="100%" stopColor={C.accentStrong} />
        </linearGradient>
        <linearGradient id={`${uid}-pl`} x1="0" y1="0" x2="0.6" y2="1">
          <stop offset="0%" stopColor={C.accent} />
          <stop offset="100%" stopColor={C.accentStrong} />
        </linearGradient>
        <linearGradient id={`${uid}-pr`} x1="1" y1="0" x2="0.3" y2="1">
          <stop offset="0%" stopColor={C.accentStrong} />
          <stop offset="100%" stopColor="#5E2A1B" />
        </linearGradient>
        <filter id={`${uid}-shadow`} x="-60%" y="-60%" width="220%" height="220%">
          <feDropShadow dx="0" dy="1" stdDeviation="1" floodColor="#201F18" floodOpacity="0.35" />
        </filter>
      </defs>

      <g filter={`url(#${uid}-shadow)`} strokeLinejoin="round">
        {/* crown facets (behind table, either side) */}
        <polygon points="9,4 3,9 7,7.5" fill={`url(#${uid}-cl)`} stroke={C.accentStrong} strokeWidth="0.4" />
        <polygon points="15,4 17,7.5 21,9" fill={`url(#${uid}-cr)`} stroke={C.accentStrong} strokeWidth="0.4" />
        {/* pavilion facets */}
        <polygon points="3,9 12,21 12,9" fill={`url(#${uid}-pl)`} stroke={C.accentStrong} strokeWidth="0.4" />
        <polygon points="12,9 12,21 21,9" fill={`url(#${uid}-pr)`} stroke={C.accentStrong} strokeWidth="0.4" />
        {/* table (top facet, drawn last so it sits proud) */}
        <polygon points="9,4 15,4 17,7.5 7,7.5" fill={`url(#${uid}-table)`} stroke={C.accentStrong} strokeWidth="0.4" />
        {/* specular glint */}
        <path d="M9.3,5 L11.4,4.6" stroke="#FFFFFF" strokeWidth="0.9" strokeLinecap="round" opacity="0.85" />
        <path d="M4.6,8.5 L6.4,7.6" stroke="#FFFFFF" strokeWidth="0.7" strokeLinecap="round" opacity="0.55" />
      </g>
    </svg>
  )
}

export function Header({ snapshot, historyOpen, onHistory }: { snapshot: DiscussionSnapshot | null; historyOpen: boolean; onHistory: () => void }) {
  const running = snapshot && !['completed', 'failed'].includes(snapshot.status)
  const liveLabel = snapshot?.status === 'queued' ? 'Waiting for worker' : snapshot?.status === 'preparing' ? 'Preparing evidence' : 'Live discussion'
  return <header className="app-header"><div className="header-start"><button type="button" className="history-toggle" onClick={onHistory} aria-label={historyOpen ? 'Close discussion history' : 'Open discussion history'} aria-expanded={historyOpen}>{historyOpen ? <PanelLeftClose size={19} /> : <PanelLeftOpen size={19} />}</button><div className="brand"><span className="brand-mark"><BrandMark /></span><span>Decision Engine</span></div></div><div className="eyebrow" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>{running ? <><span className="status-dot pulse" />{liveLabel}</> : 'Multi-agent intelligence'}</div></header>
}

interface SetupProps { topic: string; objective: string; constraints: string; rounds: number; participantSelectionMode: ParticipantSelectionMode; participantCount: number; selectedParticipantIds: string[]; objectiveFile: File | null; starting: boolean; error: string | null; onTopic: (v: string) => void; onObjective: (v: string) => void; onConstraints: (v: string) => void; onRounds: (v: number) => void; onParticipantSelectionMode: (v: ParticipantSelectionMode) => void; onParticipantCount: (v: number) => void; onToggleParticipant: (id: string) => void; onFile: (file: File | null) => void; onFileSubmit: () => void; onSubmit: (e: FormEvent) => void }
export function Setup(p: SetupProps) {
  const participantSelectionValid = p.participantSelectionMode === 'random' || p.selectedParticipantIds.length >= 3
  return <section className="setup fade-up"><div className="hero"><div className="feature-row"><div className="feature"><GitBranch size={18} />Live agent topology</div><div className="feature"><BarChart3 size={18} />Round-by-round analytics</div><div className="feature"><ShieldCheck size={18} />Evidence-aware conclusion</div></div><div className="hero-copy"><div><div className="eyebrow">Collaborative decision intelligence</div><h1>Put the question<br />before the <em>room.</em></h1><p className="lede">Specialized agents research, challenge, and refine their positions. Watch the network deliberate, inspect the analytics, then receive the final conclusion.</p></div><PersonaTopologyPreview mode={p.participantSelectionMode} participantCount={p.participantSelectionMode === 'custom' ? p.selectedParticipantIds.length : p.participantCount} /></div>
    <div className="setup-columns">
    <form className="form-panel" onSubmit={p.onSubmit}><div className="eyebrow">New discussion</div><h2 className="form-title">Frame the decision</h2>{p.error && <div className="error-box"><AlertCircle size={17} />{p.error}</div>}
      <div className="upload-primary"><div className="upload-heading"><FileText size={20} /><div><div className="upload-title">Upload an objective file <span className="eyebrow" style={{color:C.accent,marginLeft:6}}>Recommended</span></div><div className="upload-note">For the best results, use a structured UTF-8 `.txt` or `.md` file containing the objective, constraints, discussion topics, evidence requirements, and final goal.</div></div></div><input className="file-input" type="file" accept=".txt,.md,text/plain,text/markdown" onChange={e => p.onFile(e.target.files?.[0] || null)} disabled={p.starting} />{p.objectiveFile && <div className="file-name">Selected: {p.objectiveFile.name}</div>}<button className="start-button" type="button" onClick={p.onFileSubmit} disabled={p.starting || !p.objectiveFile || !participantSelectionValid}><span>{p.starting ? 'Preparing the room…' : 'Start with uploaded file'}</span>{p.starting ? <Activity className="pulse" size={18} /> : <ArrowRight size={18} />}</button></div>
      <div className="section-divider">Or enter it manually</div>
      <div className="manual-fields"><div className="field"><label htmlFor="topic">Discussion topic <span className="field-hint">3–160 characters</span></label><input id="topic" className="input" value={p.topic} placeholder="e.g. Choosing the best architecture" onChange={e => p.onTopic(e.target.value)} minLength={3} maxLength={160} required /></div>
      <div className="field"><label htmlFor="objective">Objective <span className="field-hint">What must the agents decide?</span></label><textarea id="objective" className="input" value={p.objective} placeholder="e.g. Evaluate the available approaches and reach a defensible recommendation." onChange={e => p.onObjective(e.target.value)} minLength={10} maxLength={4000} required /></div>
      <div className="field"><label htmlFor="constraints">Constraints <span className="field-hint">One per line</span></label><textarea id="constraints" className="input" value={p.constraints} placeholder={'e.g. Prefer feasible options\nMake risks and uncertainty explicit'} onChange={e => p.onConstraints(e.target.value)} maxLength={4000} /></div>
      <div className="field-row"><div className="field field-rounds"><label htmlFor="rounds">Discussion rounds <span className="field-hint">3–6 rounds</span></label><select id="rounds" className="input" value={p.rounds} onChange={e => p.onRounds(Number(e.target.value))}>{[3,4,5,6].map(v => <option key={v} value={v}>{v} rounds</option>)}</select></div>
      <button className="start-button" type="submit" disabled={p.starting || !participantSelectionValid || p.topic.trim().length < 3 || p.objective.trim().length < 10}><span>{p.starting ? 'Preparing the room…' : 'Start with manual details'}</span>{p.starting ? <Activity className="pulse" size={18} /> : <ArrowRight size={18} />}</button></div></div>
    </form>
    <PersonaColumn mode={p.participantSelectionMode} participantCount={p.participantCount} selectedIds={p.selectedParticipantIds} onMode={p.onParticipantSelectionMode} onParticipantCount={p.onParticipantCount} onToggle={p.onToggleParticipant} />
    </div></div></section>
}

interface WorkspaceProps { snapshot: DiscussionSnapshot | null; topic: string | null; analytics: AnalyticsArtifact | null; replay: ReplayTimeline | null; view: View; selectedAgent: string | null; error: string | null; onView: (v: View) => void; onAgent: (a: string) => void; onReset: () => void }

export function RunWorkspace(p: WorkspaceProps) {
  if (!p.snapshot) return <div className="panel fade-up"><div className="empty"><Activity className="pulse" size={30} /><strong>Preparing the agent room</strong>The first persisted snapshot is on its way.</div></div>
  const s = p.snapshot, complete = s.status === 'completed'
  const showingPreparation = ['queued', 'preparing'].includes(s.status) || (s.status === 'failed' && s.progress.completed_rounds === 0 && Boolean(s.preparation))
  const progress = showingPreparation ? preparationPercent(s.preparation) : s.progress.total_rounds ? s.progress.completed_rounds / s.progress.total_rounds * 100 : 0
  return <section className="fade-up"><div className="run-header"><div><div className="eyebrow">Discussion room · {s.discussion_id.slice(-12)}</div><h1>{topicText(s, p.topic)}</h1><div className="run-sub">{s.participants.length} agents · {s.progress.total_rounds} rounds · live snapshots from the discussion backend</div></div><button className="reset-button" onClick={p.onReset}><RotateCcw size={15} />New discussion</button></div>
    {p.error && <div className="error-box"><AlertCircle size={17} />{p.error}</div>}{s.warnings?.map((warning, index) => <div className="warning-box" key={`${warning.code}-${index}`}><AlertCircle size={17} /><span>{warning.message}</span></div>)}<div className="status-row"><span className="status-pill"><span className={`status-dot ${!complete && s.status !== 'failed' ? 'pulse' : ''}`} />{s.status}</span><span className="status-pill"><Clock3 size={13} />{showingPreparation ? 'Phase 1 of 2 · Evidence' : `Round ${s.progress.current_round} / ${s.progress.total_rounds}`}</span><span className="status-pill"><Users size={13} />{s.participants.length} participants</span></div><div className="progress-track"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
    {showingPreparation ? <DossierPreparation preparation={s.preparation} failed={s.status === 'failed'} /> : <><nav className="tabs"><button className={`tab ${p.view === 'discussion' ? 'active' : ''}`} onClick={() => p.onView('discussion')}><MessageSquare size={16} />Discussion</button><button className={`tab ${p.view === 'analytics' ? 'active' : ''}`} onClick={() => p.onView('analytics')}><BarChart3 size={16} />Analytics</button><button className={`tab ${p.view === 'conclusion' ? 'active' : ''}`} onClick={() => p.onView('conclusion')} disabled={!complete}><FileText size={16} />Conclusion {!complete && '· pending'}</button></nav>
    {p.view === 'discussion' && <DiscussionView snapshot={s} replay={p.replay} selectedAgent={p.selectedAgent} onAgent={p.onAgent} />}{p.view === 'analytics' && <AnalyticsView snapshot={s} analytics={p.analytics} />}{p.view === 'conclusion' && <ConclusionView snapshot={s} />}</>}
  </section>
}

function DiscussionView({ snapshot, replay, selectedAgent, onAgent }: { snapshot: DiscussionSnapshot; replay: ReplayTimeline | null; selectedAgent: string | null; onAgent: (a: string) => void }) {
  return <DiscussionReplay snapshot={snapshot} replay={replay} selectedAgent={selectedAgent} onAgent={onAgent} />
}

/* ============================================================================
   AGENT GRAPH (Discussion tab)
   Agents sit on a ring as profile cards, joined by a single one-way loop of
   arrows (clockwise). The message currently being replayed travels along that
   loop from its sender to each recipient: a line draws, a packet hands off
   from card to card, and every recipient pulses as the packet arrives.
   Everything is driven by the existing snapshot (participants + messages) —
   no new API calls or backend fields are needed.
   ========================================================================== */
