import React, { useState } from 'react';
import CameraCapture from './components/CameraCapture';
import ResultCard from './components/ResultCard';
import HistoryList from './components/HistoryList';
import OfflineBanner from './components/OfflineBanner';

export default function App() {
  const [view, setView] = useState('scan'); // 'scan' | 'result' | 'history'
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('maizescan_history') || '[]');
    } catch {
      return [];
    }
  });

  function handleResult(result) {
    const entry = { ...result, timestamp: new Date().toISOString() };
    const updated = [entry, ...history].slice(0, 20);
    setHistory(updated);
    try { localStorage.setItem('maizescan_history', JSON.stringify(updated)); } catch {}
    setPrediction(entry);
    setView('result');
  }

  function handleScanAnother() {
    setPrediction(null);
    setView('scan');
  }

  return (
    <div className="app">
      <OfflineBanner />

      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__icon">🌽</span>
          <div>
            <div className="app-header__title">MaizeScan</div>
            <div className="app-header__subtitle">Crop Disease Detection</div>
          </div>
        </div>
        <nav className="app-nav">
          <button
            className={`nav-btn${view === 'scan' || view === 'result' ? ' nav-btn--active' : ''}`}
            onClick={() => { setPrediction(null); setView('scan'); }}
            aria-label="Scan"
          >Scan</button>
          <button
            className={`nav-btn${view === 'history' ? ' nav-btn--active' : ''}`}
            onClick={() => setView('history')}
            aria-label="History"
          >History</button>
        </nav>
      </header>

      <main className="app-main">
        {view === 'scan' && <CameraCapture onResult={handleResult} />}
        {view === 'result' && prediction && (
          <ResultCard result={prediction} onScanAnother={handleScanAnother} />
        )}
        {view === 'history' && (
          <HistoryList history={history} onClear={() => {
            setHistory([]);
            localStorage.removeItem('maizescan_history');
          }} />
        )}
      </main>

      <footer className="app-footer">
        <div className="app-footer__fao">
          <span>🌾</span>
          <span>Supporting smallholder farmers in Sub-Saharan Africa</span>
        </div>
        <div className="app-footer__divider" />
        <span>Binary maize leaf disease classifier · Offline-capable PWA</span>
      </footer>
    </div>
  );
}
