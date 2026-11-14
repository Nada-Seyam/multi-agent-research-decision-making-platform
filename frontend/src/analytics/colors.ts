import { personaFor } from '../personas'
import { agentColor } from '../utils'

export function analyticsColorFor(agent: string, participants: string[]) {
  return personaFor(agent)?.marker ?? agentColor(agent, participants)
}
