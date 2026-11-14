import { useMemo, useState } from 'react'
import { Activity, AlertCircle, Clock3, History, PanelLeftClose, Search, Users } from 'lucide-react'
import type { DiscussionHistoryItem } from './api'

interface DiscussionHistoryProps {
  open: boolean
  items: DiscussionHistoryItem[]
  total: number
  loading: boolean
  error: string | null
  onOpen: (item: DiscussionHistoryItem) => void
  onClose: () => void
  onLoadMore: () => void
}

function formatDate(value?: string | null) {
  if (!value) return 'Date unavailable'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'Date unavailable'
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() === new Date().getFullYear() ? undefined : 'numeric',
  }).format(date)
}

function readableCandidate(value?: string | null) {
  return value ? value.replace(/_/g, ' ') : null
}

export function DiscussionHistory({ open, items, total, loading, error, onOpen, onClose, onLoadMore }: DiscussionHistoryProps) {
  const [query, setQuery] = useState('')
  const visibleItems = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return items
    return items.filter(item => [item.title, item.preferred_candidate, item.consensus_status]
      .some(value => value?.toLowerCase().replace(/_/g, ' ').includes(normalized)))
  }, [items, query])

  return <>
    {open && <button type="button" className="history-backdrop" onClick={onClose} aria-label="Close discussion history" />}
    <aside className={`history-drawer ${open ? 'open' : ''}`} aria-hidden={!open} aria-label="Discussion history">
      <div className="history-drawer-head">
        <div><span className="eyebrow">Saved discussions</span><strong><History size={17} />History</strong></div>
        <button type="button" className="history-close" onClick={onClose} aria-label="Close discussion history"><PanelLeftClose size={18} /></button>
      </div>

      <label className="history-search"><Search size={15} /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search history" tabIndex={open ? 0 : -1} /></label>

      <div className="history-drawer-meta"><span>{query ? `${visibleItems.length} matching` : `${total} discussions`}</span><span>Newest first</span></div>
      <div className="history-scroll">
        {error && <div className="history-drawer-message error"><AlertCircle size={17} />{error}</div>}
        {!error && loading && items.length === 0 && <div className="history-drawer-message"><Activity className="pulse" size={18} />Loading history…</div>}
        {!error && !loading && items.length === 0 && <div className="history-drawer-message"><History size={18} />No saved discussions yet.</div>}
        {!error && items.length > 0 && visibleItems.length === 0 && <div className="history-drawer-message"><Search size={18} />No discussions match “{query}”.</div>}

        <div className="history-drawer-list">
          {visibleItems.map(item => (
            <button type="button" className="history-drawer-card" key={item.discussion_id} onClick={() => onOpen(item)} tabIndex={open ? 0 : -1}>
              <span className="history-drawer-title">{item.title}</span>
              <span className="history-drawer-details">
                <span className={`history-status ${item.status}`}>{item.status}</span>
                <span><Clock3 size={11} />{formatDate(item.completed_at || item.updated_at || item.started_at)}</span>
              </span>
              <span className="history-drawer-details">
                <span><Users size={11} />{item.participants.length}</span>
                <span>{item.completed_rounds}/{item.total_rounds} rounds</span>
                {item.consensus_status && <span>{item.consensus_status.replace(/_/g, ' ')}</span>}
              </span>
              {item.preferred_candidate && <span className="history-drawer-decision">{readableCandidate(item.preferred_candidate)}</span>}
            </button>
          ))}
        </div>

        {!query && items.length < total && <button type="button" className="history-more" onClick={onLoadMore} disabled={loading || !open} tabIndex={open ? 0 : -1}>
          {loading ? <><Activity className="pulse" size={14} />Loading…</> : 'Load more'}
        </button>}
      </div>
    </aside>
  </>
}
