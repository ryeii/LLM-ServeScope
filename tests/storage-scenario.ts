import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
async function exported(page: Page) {
  const d = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  return JSON.parse(await readFile((await (await d).path())!, "utf8"));
}
export async function storageScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(
    page.getByLabel("Operator timing model", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  await page
    .getByRole("button", { name: "Load cache demo", exact: true })
    .click();
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.locator(".storage-request-table tbody tr")).toHaveCount(8);
  await expect(
    page.locator('[data-formula-key="storage.request.4.reuse"]'),
  ).toHaveText("1,023");
  const manual = await exported(page);
  expect(manual.storage.stats.reused_tokens).toBe(4092);
  expect(manual.requests[4].cache.computed_tokens).toBe(1);
  const suffix = manual.timeline.find(
    (e: any) => e.phase === "prefill" && e.request_ids.includes(4),
  );
  expect(manual.execution_shapes[suffix.shape_id].tokens).toBe(1);
  expect(manual.execution_shapes[suffix.shape_id].context).toBe(1024);
  await page.locator('[data-formula-key="storage.request.4.reuse"]').hover();
  const card = (d: number) => page.locator(`[data-cascade-depth="${d}"]`);
  await card(0)
    .getByRole("button", {
      name: "Explain Follow this request’s complete path",
      exact: true,
    })
    .focus();
  await expect(card(1)).toContainText("Prefix restoration");
  await card(1)
    .getByRole("button", { name: "Explain Prefix restoration", exact: true })
    .focus();
  await card(2)
    .getByRole("button", { name: /^Explain ssd read$/ })
    .first()
    .focus();
  await expect(card(3)).toContainText("FIFO transfer path");
  await card(3)
    .getByRole("button", { name: "Explain I/O service function", exact: true })
    .focus();
  await expect(card(4)).toContainText("Linear size coefficient");
  await card(4)
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path:
      "docs/screenshots/" +
      (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
      "storage-tiers.png",
    fullPage: true,
  });
  await page.getByText("KV storage tiers", { exact: true }).click();
  await page
    .getByLabel("SSD service model", { exact: true })
    .selectOption("local-mooncake-path");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByRole("columnheader", { name: "Latency MAPE", exact: true }),
  ).toBeVisible();
  const fitted = await exported(page);
  const local = JSON.parse(
    await readFile("examples/storage-tier-result.json", "utf8"),
  );
  expect(fitted.metrics.throughput_tps).toBeCloseTo(
    local.metrics.throughput_tps,
    9,
  );
  expect(fitted.storage.stats.ssd_service_ms).toBeCloseTo(
    local.storage.stats.ssd_service_ms,
    9,
  );
  expect(fitted.storage.fit.payload_sha256).toBe(
    local.storage.fit.payload_sha256,
  );
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.locator('[data-formula-key="metrics.throughput_tps"]').hover();
  await expect(card(0).locator("pre")).toContainText("Σ output_tokens");
  await card(0)
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  await page.getByRole("button", { name: "Box graph", exact: true }).click();
  const ioIndex = fitted.timeline.findIndex(
    (e: any) => e.kind === "storage_io",
  );
  await page.locator(`[data-formula-key="timeline.${ioIndex}"]`).hover();
  await expect(card(0)).toContainText("service_ms");
  await card(0)
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  const rows = [
    { timestamp: 5000, input_length: 1024, output_length: 1, hash_ids: [1, 2] },
    {
      timestamp: 15000,
      input_length: 1536,
      output_length: 3,
      hash_ids: [1, 2, 3],
    },
  ];
  await page.getByLabel("Import request JSONL", { exact: true }).setInputFiles({
    name: "window.jsonl",
    mimeType: "application/x-ndjson",
    buffer: Buffer.from(rows.map((r) => JSON.stringify(r)).join("\n")),
  });
  await expect(page.getByLabel("Input length", { exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.locator(".storage-request-table tbody tr")).toHaveCount(2);
  const replay = await exported(page);
  expect(replay.workload_totals.output_tokens).toBe(4);
  expect(replay.requests[1].arrival_ms).toBe(10000);
  expect(replay.requests[1].cache.reused_tokens).toBe(1024);
  expect(replay.storage.workload_source.source_file_sha256).toHaveLength(64);
  // Load a real published window, preserving its original lengths instead of clipping them.
  await page.getByLabel("Trace window requests", { exact: true }).fill("1");
  await page
    .getByRole("button", { name: "Load trace window", exact: true })
    .click();
  await expect(page.getByLabel("Input length", { exact: true })).toHaveValue(
    "6758",
  );
  await expect(page.getByLabel("Output length", { exact: true })).toHaveValue(
    "500",
  );
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.locator(".storage-request-table tbody tr")).toHaveCount(1);
  const original = await exported(page);
  expect(original.workload_totals.output_tokens).toBe(500);
  expect(original.storage.workload_source.source_file_sha256).toBe(
    "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df",
  );
  expect(original.storage.hit_forecast.outside_request_range).toBe(true);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path:
      "docs/screenshots/" +
      (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
      "storage-mobile.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(391);
  expect(errors).toEqual([]);
}
