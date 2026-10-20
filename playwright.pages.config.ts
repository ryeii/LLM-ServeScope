import { defineConfig } from "@playwright/test";
export default defineConfig({
  outputDir: "test-results/pages",
  testDir: "./tests/pages",
  timeout: 90000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4182/LLM-ServeScope/",
    viewport: { width: 1512, height: 1050 },
    headless: true,
    channel: process.env.SERVESCOPE_BROWSER_CHANNEL || "chromium",
  },
  webServer: {
    command:
      "npx vite preview --outDir dist-pages --base /LLM-ServeScope/ --host 127.0.0.1 --port 4182 --strictPort",
    url: "http://127.0.0.1:4182/LLM-ServeScope/",
    reuseExistingServer: false,
  },
  reporter: "list",
});
