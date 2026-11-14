export const C = {
  bg: '#F4F1EA', surface: '#F4F1EA', surfaceSunken: '#ECE7DB', border: '#DAD3C3', borderStrong: '#C2B9A3',
  text: '#202522', textMuted: '#68706A', textFaint: '#94978E', accent: '#B85C44', accentSoft: 'rgba(184,92,68,0.09)',
  accentStrong: '#9A4A35', sage: '#71856B', sageSoft: 'rgba(113,133,107,0.10)', positive: '#5C7A54',
  negative: '#A84E45', negativeSoft: 'rgba(168,78,69,0.10)', slate: '#40545A',
  gold: '#A47C2E', goldSoft: 'rgba(164,124,46,0.12)',
  sans: '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
  serif: '"DM Serif Display", Georgia, "Times New Roman", serif',
}

export const AGENT_COLORS = ['#B85C44', '#71856B', '#40545A', '#B38A3E', '#A84E45', '#6B667C']

/** Agent-graph motion timing (ms): time for the packet to cross one arrow, delay before it leaves, and pause after the last hop before the next message. */
export const FLOW = { hop: 1000, delay: 150, hold: 700 }
