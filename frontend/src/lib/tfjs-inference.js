let _model = null;

export async function predictOffline(canvas) {
  // Lazy-load TF.js — NOT in main bundle (dynamic import keeps bundle small)
  const tf = await import('@tensorflow/tfjs');

  if (!_model) {
    // tf.loadGraphModel caches to IndexedDB automatically with the idb:// prefix
    try {
      _model = await tf.loadGraphModel('/models/tfjs/model.json', {
        onProgress: (fraction) =>
          console.log(`TF.js model loading: ${Math.round(fraction * 100)}%`),
      });
    } catch {
      throw new Error(
        'TF.js model not available — ensure /models/tfjs/model.json is deployed'
      );
    }
  }

  // Get pixel data from canvas.
  // OffscreenCanvas does not support tf.browser.fromPixels directly — use ImageData.
  let tensor;
  if (canvas instanceof OffscreenCanvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, 224, 224);
    tensor = tf.browser.fromPixels({
      data: imageData.data,
      width: 224,
      height: 224,
    });
  } else {
    tensor = tf.browser.fromPixels(canvas);
  }

  // CRITICAL: Do NOT divide by 255 or normalize —
  // model.preprocess_input (FIX-1) is embedded in the SavedModel graph.
  const batched = tensor.expandDims(0).cast('float32');
  const prediction = _model.predict(batched);
  const prob = (await prediction.data())[0];

  // Cleanup tensors to prevent memory leaks
  tensor.dispose();
  batched.dispose();
  prediction.dispose();

  return {
    label: prob >= 0.5 ? 'Diseased' : 'Healthy',
    confidence: prob >= 0.5 ? prob : 1 - prob,
    processing_time_ms: 0,
    isOffline: true,
  };
}
