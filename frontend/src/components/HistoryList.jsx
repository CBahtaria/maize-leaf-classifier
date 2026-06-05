import React from 'react';

function formatRelativeTime(isoStr) {
  const diffMs = Date.now() - new Date(isoStr).getTime();
  const diffMinutes = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMinutes < 1) return 'Just now';
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  return new Date(isoStr).toLocaleDateString();
}

export default function HistoryList({ history, onClear }) {
  if (history.length === 0) {
    return (
      <div className="empty-state">
        <span className="empty-state__icon">🍃</span>
        <p className="empty-state__text">No scans yet</p>
        <p className="empty-state__sub">Capture a leaf to begin your diagnosis history.</p>
      </div>
    );
  }

  return (
    <div>
      <div className="history-header">
        <span className="history-title">Scan history ({history.length})</span>
        <button className="history-clear" onClick={onClear} aria-label="Clear History">
          Clear all
        </button>
      </div>

      <div className="history-table">
        {history.map((entry) => (
          <div className="history-row" key={entry.timestamp}>
            <span className={`badge badge--${entry.label === 'Healthy' ? 'healthy' : 'diseased'}`}>
              {entry.label}
            </span>
            <span className="history-row__confidence">
              {Math.round(entry.confidence * 100)}%
            </span>
            <span className="history-row__time">{formatRelativeTime(entry.timestamp)}</span>
            {entry.isOffline && (
              <span className="badge badge--offline">Offline</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
