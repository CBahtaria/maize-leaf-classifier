import React, { useState, useEffect } from 'react';
import CameraCapture from './components/CameraCapture';
import ResultCard from './components/ResultCard';
import HistoryList from './components/HistoryList';
import OfflineBanner from './components/OfflineBanner';
import { getHealth } from './lib/api';

export default function App() {
  const [view, setView] = useState('scan'); // 'scan' | 'result' | 'history'
  const [prediction, setPrediction] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking'); // 'checking' | 'online' | 'offline'
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('maizescan_history') || '[]');
    } catch {
      return [];
    }
  });

  useEffect(() => {
    getHealth()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'));
  }, []);

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

  const API_LABEL = {
    checking: 'Checking API…',
    online:   'API Online',
    offline:  'API Offline',
  };

  return (
    <div className="app">
      <OfflineBanner />

      <header className="app-header">
        <div className="app-header__brand">
          <div className="app-header__logo">🌽</div>
          <div>
            <div className="app-header__title">MaizeScan</div>
            <div className="app-header__subtitle">Crop Disease Detection</div>
          </div>
        </div>
        <div className="app-header__right">
          <span
            className={`api-dot api-dot--${apiStatus}`}
            title={API_LABEL[apiStatus]}
            aria-label={API_LABEL[apiStatus]}
          />
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
        </div>
      </header>

      {view === 'scan' && (
        <section className="hero">
          <div className="hero__inner">
            <p className="hero__eyebrow">AI-Powered Crop Health</p>
            <h1 className="hero__title">Detect Maize<br />Disease Instantly</h1>
            <p className="hero__sub">Free, offline-capable diagnosis for smallholder farmers</p>

            <div className="hero__stats">
              <div className="stat-item">
                <span className="stat-value">InceptionV3</span>
                <span className="stat-label">Model</span>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <span className="stat-value">2 Datasets</span>
                <span className="stat-label">Training Data</span>
              </div>
              <div className="stat-divider" />
              <div className="stat-item">
                <span className="stat-value">Free</span>
                <span className="stat-label">Access</span>
              </div>
            </div>

            <div className="hero__previews">
              <div className="preview-card preview-card--a">
                <span className="preview-card__icon">✅</span>
                <div>
                  <div className="preview-card__label">Healthy</div>
                  <div className="preview-card__conf">97% confidence</div>
                </div>
              </div>
              <div className="preview-card preview-card--b">
                <span className="preview-card__icon">⚠️</span>
                <div>
                  <div className="preview-card__label">Diseased</div>
                  <div className="preview-card__conf">84% confidence</div>
                </div>
              </div>
              <div className="preview-card preview-card--c">
                <div className="preview-mini-spinner" />
                <div>
                  <div className="preview-card__label">Scanning</div>
                  <div className="preview-card__conf">Processing…</div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {apiStatus === 'offline' && view === 'scan' && (
        <div className="api-banner">
          <span>🔌</span>
          <span>
            API server not connected — scans will fail until a backend is
            running. Set <code>VITE_API_URL</code> in Vercel environment
            variables and redeploy.
          </span>
        </div>
      )}

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
        <div className="app-footer__inner">
          <div className="app-footer__grid">
            <div>
              <p className="app-footer__col-title">MaizeScan</p>
              <p className="app-footer__col-text">
                Binary maize leaf disease classifier for Sub-Saharan African smallholder farmers.
                Works offline after first load.
              </p>
            </div>
            <div>
              <p className="app-footer__col-title">Technology</p>
              <span className="app-footer__col-item">InceptionV3 · TensorFlow Lite</span>
              <span className="app-footer__col-item">Offline-capable PWA</span>
              <span className="app-footer__col-item">Dual-dataset training</span>
            </div>
          </div>
          <div className="app-footer__bottom">
            <span className="app-footer__copy">© 2025 MaizeScan · Open Source</span>
            <span className="app-footer__badge">PWA · Offline Ready</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
