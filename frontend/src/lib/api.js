const BASE_URL = import.meta.env.VITE_API_URL || '';

export async function predictImage(blob) {
  const fd = new FormData();
  fd.append('file', blob, 'leaf.jpg');
  const resp = await fetch(`${BASE_URL}/predict`, { method: 'POST', body: fd });
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  const data = await resp.json();
  return {
    label: data.label,
    confidence: data.confidence,
    processing_time_ms: data.processing_time_ms,
    model_version: data.model_version,
  };
}

export async function getHealth() {
  const resp = await fetch(`${BASE_URL}/health`);
  if (!resp.ok) throw new Error(`Health check failed ${resp.status}`);
  return resp.json();
}
