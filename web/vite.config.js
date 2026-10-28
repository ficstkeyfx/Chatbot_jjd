import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Khi chạy `npm run dev`, mọi yêu cầu /api được chuyển tới server FastAPI (cổng 8000).
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false } },
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 600 },
})
