export type DiscussionStatus = 'queued' | 'preparing' | 'running' | 'completed' | 'failed'
export type ParticipantSelectionMode = 'random' | 'custom'

export interface ParticipantSelectionInput {
  participant_selection_mode: ParticipantSelectionMode
  num_participants: number
  selected_participant_ids: string[]
}

export interface DiscussionStartInput extends ParticipantSelectionInput {
  topic: string
  objective: string
  constraints: string
  rounds: number
}

export interface DiscussionStartResponse {
  discussion_id: string
  status: DiscussionStatus
  discussion_url: string
  analysis_url: string
}

export interface DiscussionMessage {
  round_number: number
  sequence: number
  sender: string
  recipients: string[]
  content: string
}

export interface Opinion {
  persona_id: string
  round_number: number
  opinion?: string | null
  position?: string | null
  architecture?: string | null
  confidence?: number
  changed_mind?: boolean
  change_reason?: string | null
  decisive_criterion?: string | null
  ranked_candidate?: string | null
}

export interface DiscussionRound {
  round: number
  opinions: Opinion[]
  messages: DiscussionMessage[]
}

export type PreparationStage = 'queued' | 'room' | 'planning' | 'retrieving' | 'consolidating' | 'ready'

export interface PreparationQuery {
  title: string
  status: 'pending' | 'accepted' | 'partial' | 'insufficient' | 'failed' | string
  source?: string
  accepted_evidence?: number
  rejected_evidence?: number
}

export interface PreparationProgress {
  stage: PreparationStage
  message: string
  completed_units: number
  total_units: number
  queries: PreparationQuery[]
  accepted_evidence?: number
  rejected_evidence?: number
}

export interface DecisionSummary {
  preferred_candidate?: string | null
  consensus_status?: string
  support_personas?: string[]
  dissenting_positions?: Record<string, string>
  strongest_alternative?: string | null
  aggregate_utilities?: Record<string, number>
  unresolved_uncertainties?: string[]
  switch_conditions?: Record<string, string>
}

export interface DiscussionSnapshot {
  schema_version: string
  discussion_id: string
  status: DiscussionStatus
  progress: {
    current_round: number
    completed_rounds: number
    total_rounds: number
  }
  preparation?: PreparationProgress | null
  started_at?: string | null
  updated_at?: string | null
  completed_at?: string | null
  failure?: { stage?: string; message?: string; error_type?: string } | null
  warnings?: Array<{ stage: string; code: string; message: string; details?: string[] }>
  objective: { goal?: string; final_goal?: string } | string
  participants: string[]
  graph: { agents: string[]; edges: string[][]; directed?: boolean }
  configuration: Record<string, unknown>
  rounds: DiscussionRound[]
  opinions: Record<string, { rounds: Opinion[] }>
  decision_summary?: DecisionSummary | null
  conclusion?: string | null
}

export interface DiscussionHistoryItem {
  discussion_id: string
  title: string
  status: DiscussionStatus | 'unknown'
  participants: string[]
  completed_rounds: number
  total_rounds: number
  started_at?: string | null
  updated_at?: string | null
  completed_at?: string | null
  preferred_candidate?: string | null
  consensus_status?: string | null
  conclusion_preview?: string | null
}

export interface DiscussionHistoryPage {
  items: DiscussionHistoryItem[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export interface DiscussionReplayFrame {
  sequence: number
  event_type: 'discussion.started' | 'round.completed' | 'discussion.completed' | 'discussion.failed' | string
  occurred_at?: string | null
  status?: DiscussionStatus | null
  completed_round: number
  snapshot_sha256?: string | null
}

export interface DiscussionReplay {
  discussion_id: string
  frames: DiscussionReplayFrame[]
  warnings: string[]
}

export interface StanceRecord {
  agent_id: string
  round: number
  candidate?: string | null
  stance?: number | null
  change?: number | null
  candidate_changed?: boolean | null
}

export interface AgreementRecord {
  round: number
  score?: number | null
  disagreement_score?: number | null
  num_agents_considered: number
}

export interface InfluenceRecord {
  agent_id: string
  score?: number | null
  num_data_points: number
  reliability: string
  num_recipients: number
}

export interface SentimentEvaluation {
  recipient: string
  target_candidate?: string | null
  score?: number | null
  label?: string | null
}

export interface SentimentRecord {
  message_id: string
  round: number
  sender?: string | null
  evaluations: SentimentEvaluation[]
}

export interface FinalAnalysis {
  opinion_change: Record<string, StanceRecord[]>
  agreement: AgreementRecord[]
  influence: Record<string, InfluenceRecord>
  sentiment: SentimentRecord[]
  issues: string[]
}

export interface RoundAnalysis {
  round: number
  stances: Record<string, StanceRecord>
  agreement?: AgreementRecord | null
  sentiment: SentimentRecord[]
}

export interface AnalyticsArtifact {
  discussion_id: string
  status: string
  round_analyses: RoundAnalysis[]
  final_analysis?: FinalAnalysis | null
  failure?: { message?: string } | null
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const payload = await response.json() as {
        detail?: string | { message?: string }
      }
      if (typeof payload.detail === 'string') {
        message = payload.detail
      } else if (payload.detail?.message) {
        message = payload.detail.message
      }
    } catch {
      // Keep the HTTP fallback when the server did not return JSON.
    }
    throw new ApiError(message, response.status)
  }
  return response.json() as Promise<T>
}

export function startDiscussion(input: DiscussionStartInput): Promise<DiscussionStartResponse> {
  return request('/api/v1/discussions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export function startDiscussionFromFile(file: File, rounds: number, selection: ParticipantSelectionInput): Promise<DiscussionStartResponse> {
  if (!Number.isInteger(rounds) || rounds < 3 || rounds > 6) {
    return Promise.reject(new Error('Discussion rounds must be a whole number from 3 to 6.'))
  }
  const body = new FormData()
  body.append('objective_file', file)
  body.append('rounds', String(rounds))
  body.append('participant_selection_mode', selection.participant_selection_mode)
  body.append('num_participants', String(selection.num_participants))
  body.append('selected_participant_ids', JSON.stringify(selection.selected_participant_ids))
  return request('/api/v1/discussions/upload', {
    method: 'POST',
    body,
  })
}

export function getDiscussion(id: string): Promise<DiscussionSnapshot> {
  return request(`/api/v1/discussions/${encodeURIComponent(id)}`)
}

export function listDiscussions(limit = 20, offset = 0): Promise<DiscussionHistoryPage> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  return request(`/api/v1/discussions?${params}`)
}

export function getDiscussionReplay(id: string): Promise<DiscussionReplay> {
  return request(`/api/v1/discussions/${encodeURIComponent(id)}/replay`)
}

export function getAnalysis(id: string): Promise<AnalyticsArtifact> {
  return request(`/api/v1/discussions/${encodeURIComponent(id)}/analysis`)
}
