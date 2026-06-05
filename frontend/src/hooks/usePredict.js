import { useState, useRef } from 'react';
import { predictImage } from '../lib/api.js';

export function usePredict() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isOffline, setIsOffline] = useState(false);
  const predictingRef = useRef(false);

  async function predict(blob) {
    if (predictingRef.current) return;
    predictingRef.current = true;
    setLoading(true);
    setError(null);
    try {
      if (!navigator.onLine) throw new Error('offline');
      const data = await predictImage(blob);
      setIsOffline(false);
      setResult({ ...data, isOffline: false });
    } catch {
      // Offline fallback: try TF.js
      try {
        const { predictOffline } = await import('../lib/tfjs-inference.js');
        // predictOffline needs a canvas — pass blob as ImageBitmap
        const bitmap = await createImageBitmap(blob);
        let offlineCanvas;
        if (typeof OffscreenCanvas !== 'undefined') {
          offlineCanvas = new OffscreenCanvas(224, 224);
        } else {
          offlineCanvas = document.createElement('canvas');
          offlineCanvas.width = 224;
          offlineCanvas.height = 224;
        }
        const ctx = offlineCanvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0, 224, 224);
        const offlineResult = await predictOffline(offlineCanvas);
        setIsOffline(true);
        setResult({ ...offlineResult, isOffline: true });
      } catch (fallbackErr) {
        setError('Could not process image. Please try again.');
        setResult({
          label: 'Unknown',
          confidence: 0.5,
          processing_time_ms: 0,
          isOffline: true,
        });
      }
    } finally {
      predictingRef.current = false;
      setLoading(false);
    }
  }

  return { predict, result, loading, error, isOffline };
}
