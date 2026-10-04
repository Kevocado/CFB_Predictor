/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8003",
      // `/facts/{game_id}` is the ONE route this API serves at the root rather
      // than under `/api`, because the explainer service is its caller and it
      // addresses the API root. That makes it invisible to this dev server,
      // which forwards `/api` only — so the fixture modal could not read the
      // facts block in development while working in production, where FastAPI
      // serves both the SPA and the route from one origin. Forwarded here so the
      // two environments agree. NOT mounted under `/api` on the backend: the
      // explainer's path is a contract with another service.
      "/facts": "http://localhost:8003",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    // Session/game times render in the viewer's zone; pin one so tests are stable.
    env: { TZ: "America/Chicago" },
  },
})
