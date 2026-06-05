import React from 'react';
import ConfidenceBar from './ConfidenceBar';

const ADVICE = {
  Healthy: 'No signs of disease detected. Continue regular monitoring and maintain good field hygiene to keep your crop healthy.',
  Diseased: 'Disease indicators found. Inspect nearby plants and consult an agricultural extension officer if multiple leaves show similar symptoms.',
  Unknown: 'Result is inconclusive. Try photographing a single clear leaf in good lighting.',
};

export default function ResultCard({ result, onScanAnother }) {
  const isHealthy = result.label === 'Healthy';
  const isUnknown = result.label === 'Unknown';
  const modifier = isHealthy ? 'healthy' : isUnknown ? 'diseased' : 'diseased';
  const icon = isHealthy ? '✅' : isUnknown ? '❓' : '⚠️';

  return (
    <div className={`diagnosis-card diagnosis-card--${modifier}`}>
      <div className="diagnosis-header">
        <span className="diagnosis-icon">{icon}</span>
        <div className="diagnosis-title">
          <div className="diagnosis-label">{result.label.toUpperCase()}</div>
          <div className="diagnosis-subtitle">Maize leaf diagnosis</div>
        </div>
        {result.isOffline && (
          <span className="badge badge--offline">Offline</span>
        )}
      </div>

      <div className="diagnosis-body">
        <div className="diagnosis-metrics">
          <div className="metric-box">
            <div className="metric-box__label">Confidence</div>
            <div className={`metric-box__value metric-box__value--${modifier}`}>
              {Math.round(result.confidence * 100)}%
            </div>
          </div>
          <div className="metric-box">
            <div className="metric-box__label">Processing</div>
            <div className="metric-box__value">{result.processing_time_ms}ms</div>
          </div>
        </div>

        <ConfidenceBar value={result.confidence} label={result.label} />

        <p className={`diagnosis-advice diagnosis-advice--${modifier}`}>
          {ADVICE[result.label] || ADVICE.Unknown}
        </p>
      </div>

      <div className="diagnosis-actions">
        <button
          className="btn btn-outline"
          onClick={onScanAnother}
          aria-label="Scan Another"
        >
          📷 Scan Another Leaf
        </button>
      </div>
    </div>
  );
}
