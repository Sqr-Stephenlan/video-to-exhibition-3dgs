import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  // Vitest reads this Vite config field at runtime; the Vite type does not include it.
  // @ts-expect-error Vitest-only configuration extension
  test: {
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
  },
});
