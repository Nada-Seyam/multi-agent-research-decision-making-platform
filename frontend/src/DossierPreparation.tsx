import { Activity, AlertCircle, Check, Circle, Database, MessageSquare, Search, Sparkles, Users } from 'lucide-react'
import type { PreparationProgress, PreparationStage } from './api'

const STAGES: PreparationStage[] = ['room', 'planning', 'retrieving', 'consolidating', 'ready']

export function preparationPercent(preparation?: PreparationProgress | null) {
  if (!preparation || preparation.stage === 'queued') return 4
  if (preparation.stage === 'room') return 12
  if (preparation.stage === 'planning') return 25
  if (preparation.stage === 'retrieving') {
    const queryProgress = preparation.total_units > 0 ? preparation.completed_units / preparation.total_units : 0
    return 35 + queryProgress * 38
  }
  if (preparation.stage === 'consolidating') return 86
  return 98
}

export function DossierPreparation({ preparation, failed }: { preparation?: PreparationProgress | null; failed: boolean }) {
  const current = preparation?.stage || 'queued'
  const currentIndex = STAGES.indexOf(current)
  const steps = [
    { stage: 'room' as const, title: 'Agent room created', detail: 'Participants and communication graph are ready.', icon: Users },
    { stage: 'planning' as const, title: 'Research plan', detail: 'Creating focused questions for the shared evidence search.', icon: Sparkles },
    { stage: 'retrieving' as const, title: 'Search and validate sources', detail: 'Retrieving evidence and checking whether it answers each query.', icon: Search },
    { stage: 'consolidating' as const, title: 'Build shared dossier', detail: 'Organizing accepted evidence into one neutral briefing.', icon: Database },
    { stage: 'ready' as const, title: 'Begin discussion', detail: 'Handing the fixed dossier to every participating agent.', icon: MessageSquare },
  ]
  const queries = preparation?.queries || []
  const completeQueries = queries.filter(query => query.status !== 'pending').length

  return <section className={`prep-panel panel ${failed ? 'failed' : ''}`} aria-live="polite">
    <div className="prep-heading"><div><div className="eyebrow">Phase 1 of 2 · Evidence preparation</div><h2>{failed ? 'Dossier preparation stopped' : preparation?.message || 'Waiting for a discussion worker'}</h2></div>{!failed && <Activity className="pulse" size={23} />}</div>
    <div className="prep-body"><ol className="prep-timeline">{steps.map((step, index) => {
      const isComplete = currentIndex > index || current === 'ready' && step.stage === 'ready'
      const isActive = !isComplete && currentIndex === index
      const Icon = step.icon
      return <li className={`${isComplete ? 'complete' : ''} ${isActive ? 'active' : ''}`} key={step.stage}>
        <span className="prep-marker">{isComplete ? <Check size={15} /> : isActive ? <Activity className="pulse" size={15} /> : <Circle size={11} />}</span>
        <span className="prep-step-icon"><Icon size={17} /></span><span><strong>{step.title}</strong><small>{step.detail}</small></span>
      </li>
    })}</ol>
    <div className="prep-activity"><div className="prep-activity-head"><div><div className="eyebrow">Research activity</div><strong>{queries.length ? `${completeQueries} of ${queries.length} queries validated` : 'Queries appear after the research plan is ready'}</strong></div>{typeof preparation?.accepted_evidence === 'number' && <span className="prep-evidence-count">{preparation.accepted_evidence} accepted excerpts</span>}</div>
      {queries.length ? <div className="prep-query-list">{queries.map((query, index) => <div className={`prep-query ${query.status}`} key={`${index}-${query.title}`}><span className="prep-query-state">{query.status === 'pending' ? <Activity className="pulse" size={13} /> : query.status === 'failed' ? <AlertCircle size={13} /> : <Check size={13} />}</span><div><strong>{query.title}</strong><small>{query.status === 'pending' ? 'Waiting to be searched' : `${query.status}${query.source ? ` · ${query.source}` : ''}${typeof query.accepted_evidence === 'number' ? ` · ${query.accepted_evidence} excerpts` : ''}`}</small></div></div>)}</div>
      : <div className="prep-query-empty">The system is preparing a bounded set of research questions.</div>}
    </div></div>
  </section>
}
