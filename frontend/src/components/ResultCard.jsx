import React from 'react';
import ConfidenceBar from './ConfidenceBar';

export default function ResultCard({ result, onScanAnother }) {
  const isHealthy = result.label === 'Healthy';
  const cardClass = isHealthy ? 'result-card result-card--healthy' : 'result-card result-card--diseased';

  return (
    <div className={cardClass}>
      <p className="result-label">{result.label.toUpperCase()}</p>

      <p>Confidence: {Math.round(result.confidence * 100)}%</p>
      <ConfidenceBar value={result.confidence} label={result.label} />

      <p>Processed in {result.processing_time_ms}ms</p>

      {result.isOffline && (
        <span className="badge badge--offline">Offline result</span>
      )}

      <button
        className="btn btn-secondary"
        style={{ marginTop: '16px', minHeight: '48px' }}
        onClick={onScanAnother}
        aria-label="Scan Another"
      >
        Scan Another
      </button>
    </div>
  );
}
