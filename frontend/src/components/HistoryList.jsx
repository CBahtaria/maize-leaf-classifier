import React from 'react';

function formatRelativeTime(isoStr) {
  const now = Date.now();
  const then = new Date(isoStr).getTime();
  const diffMs = now - then;
  const diffMinutes = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMinutes < 1) return 'Just now';
  if (diffMinutes < 60) return `${diffMinutes} minute${diffMinutes === 1 ? '' : 's'} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
  if (diffDays === 1) return 'Yesterday';

  return new Date(isoStr).toLocaleDateString();
}

export default function HistoryList({ history, onClear }) {
  if (history.length === 0) {
    return (
      <p className="empty-state">No scans yet. Capture a leaf to begin.</p>
    );
  }

  return (
    <div className="history-list">
      <button
        className="btn btn-secondary"
        style={{ marginBottom: '16px', minHeight: '48px' }}
        onClick={onClear}
        aria-label="Clear History"
      >
        Clear History
      </button>

      {history.map((entry) => (
        <div className="history-item" key={entry.timestamp}>
          <span className={`badge badge--${entry.label.toLowerCase()}`}>
            {entry.label}
          </span>
          <span>{Math.round(entry.confidence * 100)}% confidence</span>
          <span className="history-time">{formatRelativeTime(entry.timestamp)}</span>
          {entry.isOffline && (
            <span className="badge badge--offline">Offline</span>
          )}
        </div>
      ))}
    </div>
  );
}
