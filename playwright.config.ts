import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/browser",
  timeout: 90000,
  workers: 1,
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:8788",
    viewport: { width: 1512, height: 1050 },
    headless: true,
    channel: "chromium",
  },
  reporter: "list",
});
