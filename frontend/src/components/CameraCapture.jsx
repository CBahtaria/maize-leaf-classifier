import React, { useRef, useState, useEffect } from 'react';

export default function CameraCapture({ onResult }) {
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const prevPreviewRef = useRef(null);

  useEffect(() => {
    return () => {
      if (prevPreviewRef.current) URL.revokeObjectURL(prevPreviewRef.current);
    };
  }, []);

  function handleFileChange(e) {
    if (prevPreviewRef.current) URL.revokeObjectURL(prevPreviewRef.current);
    const file = e.target.files[0];
    if (!file) return;

    // Show thumbnail preview
    const objectUrl = URL.createObjectURL(file);
    prevPreviewRef.current = objectUrl;
    setPreview(objectUrl);

    // Draw image to canvas at 224×224 for resize
    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 224, 224);

      canvas.toBlob(async (blob) => {
        if (!blob) {
          onResult({ label: 'Unknown', confidence: 0.5, processing_time_ms: 0, isOffline: true });
          setLoading(false);
          return;
        }
        await submitBlob(blob);
      }, 'image/jpeg', 0.85);
    };
    img.src = objectUrl;
  }

  async function submitBlob(blob) {
    setLoading(true);

    // Offline / no connectivity path
    if (!navigator.onLine) {
      setLoading(false);
      onResult({ label: 'Unknown', confidence: 0.5, processing_time_ms: 0, isOffline: true });
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', blob, 'leaf.jpg');

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      onResult({
        label: data.label,
        confidence: data.confidence,
        processing_time_ms: data.processing_time_ms,
        isOffline: false,
      });
    } catch {
      // Network error fallback
      onResult({ label: 'Unknown', confidence: 0.5, processing_time_ms: 0, isOffline: true });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: 'none' }}
        aria-label="Select leaf image"
        onChange={handleFileChange}
      />

      {/* Hidden canvas used for resizing to 224×224 */}
      <canvas ref={canvasRef} width={224} height={224} style={{ display: 'none' }} />

      {preview && (
        <img
          src={preview}
          alt="Selected leaf preview"
          className="camera-preview"
        />
      )}

      {loading && <div className="spinner" role="status" aria-label="Processing image" />}

      {!loading && (
        <button
          className="btn btn-primary"
          style={{ minHeight: '48px' }}
          aria-label="Scan Leaf"
          onClick={() => fileInputRef.current && fileInputRef.current.click()}
        >
          Scan Leaf
        </button>
      )}
    </div>
  );
}
