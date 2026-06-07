import React from 'react';
import ConfidenceBar from './ConfidenceBar';

const ADVICE = {
  Healthy: 'No disease indicators detected. Continue monitoring every 7–10 days and maintain good field hygiene to protect your crop.',
  Diseased: 'Disease indicators found. Isolate affected plants, check surrounding leaves, and consult an agricultural extension officer if multiple plants show similar symptoms.',
  Unknown: 'Result inconclusive — try again with a single leaf, good lighting, and the leaf filling the frame.',
};

function confidenceGrade(conf) {
  if (conf >= 0.90) return { label: 'Very high', color: '' };
  if (conf >= 0.75) return { label: 'High', color: '' };
  if (conf >= 0.60) return { label: 'Moderate', color: 'warn' };
  return { label: 'Low — retake photo', color: 'low' };
}

export default function ResultCard({ result, onScanAnother }) {
  const isHealthy = result.label === 'Healthy';
  const isUnknown = result.label === 'Unknown';
  const modifier = isHealthy ? 'healthy' : 'diseased';
  const icon = isHealthy ? '✅' : isUnknown ? '❓' : '⚠️';
  const confPct = Math.round(result.confidence * 100);
  const grade = confidenceGrade(result.confidence);

  return (
    <div className={`diagnosis-card diagnosis-card--${modifier}`}>
      <div className="diagnosis-header">
        <span className="diagnosis-icon">{icon}</span>
        <div className="diagnosis-title">
          <div className="diagnosis-label">{result.label.toUpperCase()}</div>
          <div className="diagnosis-subtitle">Maize leaf diagnosis</div>
        </div>
        {result.isOffline && <span className="badge badge--offline">Offline</span>}
      </div>

      <div className="diagnosis-body">
        <div className="diagnosis-metrics">
          <div className="metric-box">
            <div className="metric-box__label">{result.label} Confidence</div>
            <div className={`metric-box__value metric-box__value--${modifier}`}>
              {confPct}%
            </div>
            {grade.color && (
              <div className={`metric-box__grade metric-box__grade--${grade.color}`}>
                {grade.label}
              </div>
            )}
          </div>
          <div className="metric-box">
            <div className="metric-box__label">Processing</div>
            <div className="metric-box__value">{result.processing_time_ms}ms</div>
            {result.model_version && (
              <div className="metric-box__grade">{result.model_version}</div>
            )}
          </div>
        </div>

        <ConfidenceBar value={result.confidence} label={result.label} />

        <p className={`diagnosis-advice diagnosis-advice--${modifier}`}>
          {ADVICE[result.label] ?? ADVICE.Unknown}
        </p>

        {confPct < 60 && (
          <p className="diagnosis-retake">
            Confidence is low. For a more reliable result, retake the photo with the leaf filling the frame in good lighting.
          </p>
        )}
      </div>

      <div className="diagnosis-actions">
        <button className="btn btn-outline" onClick={onScanAnother} aria-label="Scan Another">
          📷 Scan Another Leaf
        </button>
      </div>
    </div>
  );
}
