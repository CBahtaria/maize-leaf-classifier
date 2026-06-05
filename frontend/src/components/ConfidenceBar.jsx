import React from 'react';

export default function ConfidenceBar({ value, label }) {
  const isHealthy = label === 'Healthy';
  const fillClass = isHealthy ? 'confidence-fill confidence-fill--healthy' : 'confidence-fill confidence-fill--diseased';
  const pct = Math.round(value * 100);

  return (
    <div className="confidence-section">
      <label>
        Confidence
        <span>{pct}%</span>
      </label>
      <div
        className="confidence-track"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label} confidence`}
      >
        <div className={fillClass} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
