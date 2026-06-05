import React from 'react';

export default function ConfidenceBar({ value, label }) {
  const fillClass =
    label === 'Healthy'
      ? 'confidence-bar__fill--healthy'
      : 'confidence-bar__fill--diseased';

  return (
    <>
      <div className="confidence-bar">
        <div
          className={`confidence-bar__fill ${fillClass}`}
          style={{ width: `${value * 100}%` }}
          role="progressbar"
          aria-valuenow={Math.round(value * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label} confidence`}
        />
      </div>
      <span>{Math.round(value * 100)}%</span>
    </>
  );
}
