import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react'
import type { DiscussionReplay as ReplayTimeline, DiscussionReplayFrame, DiscussionSnapshot } from './api'
import { AgentGraph } from './AgentGraph'

function frameLabel(frame: DiscussionReplayFrame, last: boolean) {
  if (frame.event_type === 'discussion.started') return 'Room prepared'
  if (frame.event_type === 'discussion.completed') return 'Final decision'
  if (frame.event_type === 'discussion.failed') return 'Run failed'
  if (frame.event_type === 'round.completed') return `Round ${frame.completed_round}`
  return last ? 'Latest checkpoint' : `Checkpoint ${frame.sequence}`
}

function fallbackFrames(snapshot: DiscussionSnapshot): DiscussionReplayFrame[] {
  const frames: DiscussionReplayFrame[] = [{
    sequence: 0,
    event_type: 'discussion.started',
    occurred_at: snapshot.started_at,
    status: 'preparing',
    completed_round: 0,
  }]
  snapshot.rounds.forEach((_, index) => frames.push({
    sequence: index + 1,
    event_type: 'round.completed',
    occurred_at: snapshot.updated_at,
    status: 'running',
    completed_round: index + 1,
  }))
  if (snapshot.status === 'completed' || snapshot.status === 'failed') frames.push({
    sequence: frames.length,
    event_type: `discussion.${snapshot.status}`,
    occurred_at: snapshot.completed_at || snapshot.updated_at,
    status: snapshot.status,
    completed_round: snapshot.rounds.length,
  })
  return frames
}

interface DiscussionReplayProps {
  snapshot: DiscussionSnapshot
  replay: ReplayTimeline | null
  selectedAgent: string | null
  onAgent: (agent: string) => void
}

export function DiscussionReplay({ snapshot, replay, selectedAgent, onAgent }: DiscussionReplayProps) {
  const canReplay = snapshot.status === 'completed' || snapshot.status === 'failed'
  const frames = useMemo(
    () => replay?.frames.length ? replay.frames : fallbackFrames(snapshot),
    [replay, snapshot.discussion_id, snapshot.rounds.length, snapshot.status, snapshot.updated_at],
  )
  const [cursor, setCursor] = useState(Math.max(0, frames.length - 1))
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const previousCount = useRef(frames.length)

  useEffect(() => {
    setCursor(Math.max(0, frames.length - 1))
    setPlaying(false)
    previousCount.current = frames.length
  }, [snapshot.discussion_id])

  useEffect(() => {
    setCursor(current => current >= previousCount.current - 1
      ? Math.max(0, frames.length - 1)
      : Math.min(current, Math.max(0, frames.length - 1)))
    previousCount.current = frames.length
  }, [frames.length])

  useEffect(() => {
    if (!playing) return
    if (cursor >= frames.length - 1) {
      setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setCursor(current => current + 1), 1800 / speed)
    return () => window.clearTimeout(timer)
  }, [playing, cursor, frames.length, speed])

  const frame = frames[Math.min(cursor, frames.length - 1)]
  const visibleRound = Math.min(frame?.completed_round ?? snapshot.rounds.length, snapshot.rounds.length)
  const replaySnapshot = useMemo<DiscussionSnapshot>(() => ({
    ...snapshot,
    status: frame?.status || snapshot.status,
    rounds: snapshot.rounds.slice(0, visibleRound),
    progress: {
      ...snapshot.progress,
      current_round: visibleRound,
      completed_rounds: visibleRound,
    },
  }), [snapshot, frame?.status, visibleRound])

  function togglePlay() {
    if (playing) {
      setPlaying(false)
      return
    }
    if (cursor >= frames.length - 1) setCursor(0)
    setPlaying(true)
  }

  return <div className="discussion-grid">
    {canReplay && <div className="replay-panel">
      <div className="replay-summary">
        <div><span className="eyebrow">Discussion replay</span><strong>{frameLabel(frame, cursor === frames.length - 1)}</strong></div>
        <span className="replay-counter">{cursor + 1} / {frames.length}</span>
      </div>
      <div className="replay-controls">
        <button type="button" title="Replay from the beginning" onClick={() => { setCursor(0); setPlaying(false) }}><RotateCcw size={15} /></button>
        <button type="button" title="Previous checkpoint" onClick={() => { setCursor(value => Math.max(0, value - 1)); setPlaying(false) }} disabled={cursor === 0}><ChevronLeft size={17} /></button>
        <button type="button" className="replay-play" onClick={togglePlay}>{playing ? <Pause size={16} /> : <Play size={16} />}{playing ? 'Pause' : cursor === frames.length - 1 ? 'Replay' : 'Play'}</button>
        <button type="button" title="Next checkpoint" onClick={() => { setCursor(value => Math.min(frames.length - 1, value + 1)); setPlaying(false) }} disabled={cursor === frames.length - 1}><ChevronRight size={17} /></button>
        <label className="replay-speed">Speed<select value={speed} onChange={event => setSpeed(Number(event.target.value))}><option value={0.75}>0.75×</option><option value={1}>1×</option><option value={1.5}>1.5×</option><option value={2}>2×</option></select></label>
      </div>
      <input className="replay-range" type="range" min={0} max={Math.max(0, frames.length - 1)} value={cursor} onChange={event => { setCursor(Number(event.target.value)); setPlaying(false) }} aria-label="Replay checkpoint" />
      {replay?.warnings.map((warning, index) => <div className="replay-warning" key={index}><AlertCircle size={13} />{warning}</div>)}
    </div>}
    <AgentGraph snapshot={replaySnapshot} selected={selectedAgent} onSelect={onAgent} />
  </div>
}
