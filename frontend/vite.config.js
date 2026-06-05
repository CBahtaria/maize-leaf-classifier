import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icons/icon-192.png', 'icons/icon-512.png'],
      manifest: {
        name: 'Maize Disease Checker',
        short_name: 'MaizeScan',
        description: 'Binary maize leaf disease classifier for SSA farmers',
        theme_color: '#2d6a4f',
        background_color: '#f0f4f0',
        display: 'standalone',
        start_url: '/',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [
          {
            urlPattern: /^https?.*\/predict/,
            handler: 'NetworkFirst',
            options: { cacheName: 'api-cache', expiration: { maxEntries: 10 } },
          },
        ],
      },
    }),
  ],
  server: {
    host: true,  // bind to 0.0.0.0 — required inside Docker and useful for mobile testing
    port: 5173,
    proxy: {
      // API_HOST is set to the Docker service name ("api") inside docker-compose.dev.yml.
      // Falls back to "localhost" for plain local development.
      '/predict': { target: `http://${process.env.API_HOST ?? 'localhost'}:8000`, changeOrigin: true },
      '/health':  { target: `http://${process.env.API_HOST ?? 'localhost'}:8000`, changeOrigin: true },
      '/model':   { target: `http://${process.env.API_HOST ?? 'localhost'}:8000`, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
});
