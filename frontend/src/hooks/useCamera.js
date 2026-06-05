import { useRef, useState } from 'react';

export function useCamera() {
  const canvasRef = useRef(null);
  const [preview, setPreview] = useState(null);

  function handleFileSelect(file) {
    // Set preview URL — caller is responsible for revoking
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);

    // Draw file to canvas at 224×224, then return a Promise<Blob>
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) {
          reject(new Error('Canvas ref not attached'));
          return;
        }
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 224, 224);
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error('canvas.toBlob returned null'));
              return;
            }
            resolve(blob);
          },
          'image/jpeg',
          0.85
        );
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = objectUrl;
    });
  }

  function reset() {
    setPreview(null);
  }

  return { canvasRef, preview, handleFileSelect, reset };
}
