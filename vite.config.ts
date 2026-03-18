import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import compression from 'vite-plugin-compression'

export default defineConfig({
  base: "/Fridge-Ai-Master/",
  plugins: [
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used  do not remove them
    react(),
    tailwindcss(),
    compression(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['apple-icon.png', 'vite.svg'],
      manifest: {
        name: 'AI Fridge for LLM - Culinary Intelligence',
        short_name: 'FridgeLLM',
        description: 'AI-powered fridge management optimized for LLM integration and optimized vision performance.',
        theme_color: '#00ff88',
        background_color: '#0f2e24',
        display: 'standalone',
        orientation: 'portrait',
        icons: [
          {
            src: 'apple-icon.png',
            sizes: '1024x1024',
            type: 'image/png'
          },
          {
            src: 'vite.svg',
            sizes: '192x192 512x512',
            type: 'image/svg+xml'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,onnx,wasm}'],
        cleanupOutdatedCaches: true,
        maximumFileSizeToCacheInBytes: 100 * 1024 * 1024, // 100MB for model and wasm
      }
    }),
  ],
  server: {
    allowedHosts: true,
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
  },
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },

  build: {
    minify: 'esbuild',
    target: 'esnext',
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router'],
          'ui-vendor': ['lucide-react', 'motion'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,
    cssCodeSplit: true,
    sourcemap: false,
  },
  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
})
