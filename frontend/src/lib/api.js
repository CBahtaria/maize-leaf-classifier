const BASE_URL = import.meta.env.VITE_API_URL || '';

export async function predictImage(blob) {
  const fd = new FormData();
  fd.append('file', blob, 'leaf.jpg');
  const resp = await fetch(`${BASE_URL}/predict`, { method: 'POST', body: fd });
  if (!resp.ok) throw new Error(`API ${resp.status}`);
  const data = await resp.json();

  const label = data.label;
  const rawConf = data.confidence; // P(Diseased): 0 = Healthy, 1 = Diseased
  // Normalise so confidence always represents certainty for the predicted class
  const confidence = label === 'Healthy' ? 1 - rawConf : rawConf;

  return {
    label,
    confidence,
    raw_confidence: rawConf,
    processing_time_ms: data.processing_time_ms,
    model_version: data.model_version,
  };
}

export async function getHealth() {
  const resp = await fetch(`${BASE_URL}/health`, {
    signal: AbortSignal.timeout(5000),
  });
  if (!resp.ok) throw new Error(`Health ${resp.status}`);
  const ct = resp.headers.get('content-type') || '';
  if (!ct.includes('application/json')) throw new Error('Not JSON — SPA or wrong URL');
  return resp.json();
}
