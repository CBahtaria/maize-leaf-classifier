import React, { useRef, useState, useEffect } from 'react';
import { predictImage } from '../lib/api';

// Must match model/config.py IMG_SIZE; InceptionV3 was trained at this resolution
const CANVAS_SIZE = 224;

export default function CameraCapture({ onResult }) {
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorType, setErrorType] = useState(null); // 'offline' | 'api_down' | 'server_error'
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const prevPreviewRef = useRef(null);
  const lastBlobRef = useRef(null);

  useEffect(() => () => {
    if (prevPreviewRef.current) URL.revokeObjectURL(prevPreviewRef.current);
  }, []);

  function handleFileChange(e) {
    if (prevPreviewRef.current) URL.revokeObjectURL(prevPreviewRef.current);
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorType(null);

    const objectUrl = URL.createObjectURL(file);
    prevPreviewRef.current = objectUrl;
    setPreview(objectUrl);

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      // Center-crop to square to preserve aspect ratio, then scale to CANVAS_SIZE
      const size = Math.min(img.naturalWidth, img.naturalHeight);
      const sx = (img.naturalWidth - size) / 2;
      const sy = (img.naturalHeight - size) / 2;
      ctx.drawImage(img, sx, sy, size, size, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
      canvas.toBlob(blob => {
        if (blob) { lastBlobRef.current = blob; runInference(blob); }
      }, 'image/jpeg', 0.92);
    };
    img.onerror = () => setErrorType('server_error');
    img.src = objectUrl;
  }

  async function runInference(blob) {
    setLoading(true);
    setErrorType(null);
    if (!navigator.onLine) { setLoading(false); setErrorType('offline'); return; }
    try {
      const result = await predictImage(blob);
      onResult({ ...result, isOffline: false });
    } catch (err) {
      const msg = String(err?.message ?? '');
      if (!navigator.onLine || /failed to fetch|networkerror|load failed/i.test(msg)) {
        setErrorType('api_down');
      } else {
        setErrorType('server_error');
      }
    } finally {
      setLoading(false);
    }
  }

  const ERRORS = {
    offline: {
      icon: '📡',
      title: 'No internet connection',
      body: 'Reconnect and tap Retry.',
    },
    api_down: {
      icon: '🔌',
      title: 'API server not reachable',
      body: 'The inference backend is offline. Deploy your API and set VITE_API_URL, then redeploy the frontend.',
    },
    server_error: {
      icon: '⚠️',
      title: 'Scan failed',
      body: 'Server returned an error. Try a clearer close-up photo of a single leaf and retry.',
    },
  };
  const currentError = ERRORS[errorType];

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <canvas ref={canvasRef} width={CANVAS_SIZE} height={CANVAS_SIZE} style={{ display: 'none' }} />

      <div
        className={`upload-zone${preview ? ' upload-zone--filled' : ''}`}
        onClick={() => !loading && fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Tap to select a leaf image"
        onKeyDown={e => e.key === 'Enter' && !loading && fileInputRef.current?.click()}
      >
        {preview
          ? <img src={preview} alt="Leaf preview" className="upload-preview" />
          : <div className="upload-placeholder">
              <span className="upload-icon">🍃</span>
              <p className="upload-text">Tap to photograph a leaf</p>
              <p className="upload-hint">JPEG · PNG · WebP · max 10 MB</p>
            </div>
        }
      </div>

      {currentError && (
        <div className="scan-error">
          <span className="scan-error__icon">{currentError.icon}</span>
          <div>
            <p className="scan-error__title">{currentError.title}</p>
            <p className="scan-error__body">{currentError.body}</p>
          </div>
        </div>
      )}

      {loading
        ? <div className="spinner-wrap">
            <div className="spinner" role="status" aria-label="Processing" />
            <span>Analysing leaf…</span>
          </div>
        : <div className="btn-stack">
            <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()}>
              {preview ? '🔄 Scan Again' : '📷 Scan Leaf'}
            </button>
            {currentError && lastBlobRef.current && (
              <button className="btn btn-outline" onClick={() => runInference(lastBlobRef.current)}>
                ↩ Retry
              </button>
            )}
          </div>
      }

      {!preview && (
        <div className="scan-tips">
          <p className="scan-tips__title">Tips for accurate results</p>
          <ul className="scan-tips__list">
            <li>One leaf per scan — avoid leaf clusters</li>
            <li>Good lighting — no shadows or flash glare</li>
            <li>Fill the frame; keep the camera steady</li>
            <li>Upper leaf surface works best</li>
          </ul>
        </div>
      )}
    </div>
  );
}
