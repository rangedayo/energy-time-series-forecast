import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite 개발 서버는 5173 포트(백엔드 CORS_ORIGINS 허용 대상).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
});
