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
    const updated = [entry, ...history].slice(0, 20); // keep last 20
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
        <span className="app-logo">🌽</span>
        <h1 className="app-title">MaizeScan</h1>
        <nav className="app-nav">
          <button
            className={`nav-btn${view === 'scan' || view === 'result' ? ' active' : ''}`}
            onClick={() => { setPrediction(null); setView('scan'); }}
            aria-label="Scan"
          >Scan</button>
          <button
            className={`nav-btn${view === 'history' ? ' active' : ''}`}
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
        <p>Binary maize leaf disease classifier for SSA farmers</p>
      </footer>
    </div>
  );
}
