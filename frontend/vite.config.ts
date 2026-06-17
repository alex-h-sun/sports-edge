import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev`, proxy API calls to the local Starlette server on :8000.
// In production the same server serves this build, so requests are same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
