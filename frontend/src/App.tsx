import { type FormEvent, useEffect, useRef, useState } from 'react'
import { type AnalyticsArtifact, ApiError, type DiscussionHistoryItem, type DiscussionReplay, type DiscussionSnapshot, type ParticipantSelectionInput, type ParticipantSelectionMode, getAnalysis, getDiscussion, getDiscussionReplay, listDiscussions, startDiscussion, startDiscussionFromFile } from './api'
import { DiscussionHistory } from './DiscussionHistory'
import { Header, RunWorkspace, Setup } from './layout'
import { PERSONAS } from './personas'
import { GLOBAL_STYLE } from './styles'
import { extractTopic, type View } from './utils'

function App() {
  const [topic, setTopic] = useState('')
  const [objective, setObjective] = useState('')
  const [constraints, setConstraints] = useState('')
  const [rounds, setRounds] = useState(4)
  const [participantSelectionMode, setParticipantSelectionMode] = useState<ParticipantSelectionMode>('random')
  const [participantCount, setParticipantCount] = useState(4)
  const [selectedParticipantIds, setSelectedParticipantIds] = useState<string[]>(() => PERSONAS.slice(0, 4).map(persona => persona.id))
  const [objectiveFile, setObjectiveFile] = useState<File | null>(null)
  const [discussionId, setDiscussionId] = useState<string | null>(null)
  const [startedTopic, setStartedTopic] = useState<string | null>(null)
  const [snapshot, setSnapshot] = useState<DiscussionSnapshot | null>(null)
  const [analytics, setAnalytics] = useState<AnalyticsArtifact | null>(null)
  const [replay, setReplay] = useState<DiscussionReplay | null>(null)
  const [history, setHistory] = useState<DiscussionHistoryItem[]>([])
  const [historyTotal, setHistoryTotal] = useState(0)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [view, setView] = useState<View>('discussion')
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const conclusionShown = useRef(false)

  function participantSelection(): ParticipantSelectionInput {
    return {
      participant_selection_mode: participantSelectionMode,
      num_participants: participantSelectionMode === 'custom' ? selectedParticipantIds.length : participantCount,
      selected_participant_ids: participantSelectionMode === 'custom' ? selectedParticipantIds : [],
    }
  }

  function toggleParticipant(personaId: string) {
    setSelectedParticipantIds(current => current.includes(personaId)
      ? current.filter(id => id !== personaId)
      : [...current, personaId])
  }

  async function loadHistory(reset = false) {
    if (historyLoading) return
    setHistoryLoading(true)
    setHistoryError(null)
    const offset = reset ? 0 : history.length
    try {
      const page = await listDiscussions(10, offset)
      setHistory(current => reset ? page.items : [...current, ...page.items.filter(item => !current.some(existing => existing.discussion_id === item.discussion_id))])
      setHistoryTotal(page.total)
    } catch (cause) {
      setHistoryError(cause instanceof Error ? cause.message : 'Could not load saved discussions.')
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => { void loadHistory(true) }, [])

  useEffect(() => {
    if (!discussionId) return
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      const [discussionResult, analysisResult] = await Promise.allSettled([getDiscussion(discussionId), getAnalysis(discussionId)])
      if (cancelled) return
      let nextStatus: DiscussionSnapshot['status'] | null = null
      if (discussionResult.status === 'fulfilled') {
        const next = discussionResult.value
        nextStatus = next.status
        setSnapshot(next)
        setSelectedAgent(current => current || next.participants[0] || null)
        if (next.status === 'completed' && !conclusionShown.current) { conclusionShown.current = true; setView('conclusion') }
        if (next.status === 'failed') setError(next.failure?.message || 'The discussion failed before completing.')
      } else {
        setError(discussionResult.reason instanceof Error ? discussionResult.reason.message : 'Could not load the discussion.')
      }
      if (analysisResult.status === 'fulfilled') setAnalytics(analysisResult.value)
      else if (!(analysisResult.reason instanceof ApiError && analysisResult.reason.status === 404)) console.warn(analysisResult.reason)
      const analyticsReady = analysisResult.status === 'fulfilled' && ['completed', 'failed'].includes(analysisResult.value.status)
      const shouldContinue = discussionResult.status === 'rejected' || (nextStatus !== 'failed' && (nextStatus !== 'completed' || !analyticsReady))
      if (shouldContinue) timer = window.setTimeout(poll, discussionResult.status === 'rejected' ? 2500 : 1500)
    }
    void poll()
    return () => { cancelled = true; if (timer) window.clearTimeout(timer) }
  }, [discussionId])

  useEffect(() => {
    if (!discussionId || !snapshot || !['completed', 'failed'].includes(snapshot.status)) return
    let cancelled = false
    getDiscussionReplay(discussionId)
      .then(value => { if (!cancelled) setReplay(value) })
      .catch(cause => { if (!cancelled) console.warn(cause) })
    return () => { cancelled = true }
  }, [discussionId, snapshot?.status])

  async function handleStart(event: FormEvent) {
    event.preventDefault()
    const cleanTopic = topic.trim(), cleanObjective = objective.trim(), cleanConstraints = constraints.trim()
    if (cleanTopic.length < 3 || cleanObjective.length < 10) {
      setError('Enter a discussion topic and an objective before starting.')
      return
    }
    setStarting(true); setError(null); setSnapshot(null); setAnalytics(null); setReplay(null); setView('discussion'); conclusionShown.current = false
    setStartedTopic(cleanTopic)
    try { setDiscussionId((await startDiscussion({ topic: cleanTopic, objective: cleanObjective, constraints: cleanConstraints, rounds, ...participantSelection() })).discussion_id) }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not start the discussion.') }
    finally { setStarting(false) }
  }
  async function handleFileStart() {
    if (!objectiveFile) { setError('Choose a .txt or .md objective file first.'); return }
    setStarting(true); setError(null); setSnapshot(null); setAnalytics(null); setReplay(null); setView('discussion'); conclusionShown.current = false
    setStartedTopic(await objectiveFile.text().then(extractTopic).catch(() => null))
    try { setDiscussionId((await startDiscussionFromFile(objectiveFile, rounds, participantSelection())).discussion_id) }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not upload the objective file.') }
    finally { setStarting(false) }
  }
  function openHistory(item: DiscussionHistoryItem) {
    setDiscussionId(item.discussion_id)
    setStartedTopic(item.title)
    setSnapshot(null)
    setAnalytics(null)
    setReplay(null)
    setSelectedAgent(null)
    setError(null)
    setView('discussion')
    setHistoryOpen(false)
    // Saved runs open in replay mode instead of jumping directly to conclusion.
    conclusionShown.current = true
  }
  function reset() { setDiscussionId(null); setSnapshot(null); setAnalytics(null); setReplay(null); setError(null); setView('discussion'); conclusionShown.current = false; void loadHistory(true) }

  return <div className="app-shell"><style>{GLOBAL_STYLE}</style><Header snapshot={snapshot} historyOpen={historyOpen} onHistory={() => setHistoryOpen(value => !value)} />
    <DiscussionHistory open={historyOpen} items={history} total={historyTotal} loading={historyLoading} error={historyError} onOpen={openHistory} onClose={() => setHistoryOpen(false)} onLoadMore={() => void loadHistory(false)} />
    <div className={`content-shell ${historyOpen ? 'history-open' : ''}`}><main className={`main ${!discussionId ? 'main-setup' : ''}`}>
    {!discussionId ? <Setup topic={topic} objective={objective} constraints={constraints} rounds={rounds} participantSelectionMode={participantSelectionMode} participantCount={participantCount} selectedParticipantIds={selectedParticipantIds} objectiveFile={objectiveFile} starting={starting} error={error} onTopic={setTopic} onObjective={setObjective} onConstraints={setConstraints} onRounds={setRounds} onParticipantSelectionMode={setParticipantSelectionMode} onParticipantCount={setParticipantCount} onToggleParticipant={toggleParticipant} onFile={setObjectiveFile} onFileSubmit={handleFileStart} onSubmit={handleStart} />
      : <RunWorkspace snapshot={snapshot} topic={startedTopic} analytics={analytics} replay={replay} view={view} selectedAgent={selectedAgent} error={error} onView={setView} onAgent={setSelectedAgent} onReset={reset} />}
  </main></div></div>
}

export default App
