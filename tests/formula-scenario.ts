import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

async function exported(page: Page) {
  const ready = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  return JSON.parse(await readFile((await (await ready).path())!, "utf8"));
}
export async function formulaScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  const throughput = page.locator(
    '[data-formula-key="metrics.throughput_tps"]',
  );
  await expect(throughput).toBeVisible({ timeout: 60000 });
  const initial = await exported(page);
  await throughput.hover();
  let tip = page.locator('[data-cascade-depth="0"]');
  await expect(tip).toBeVisible();
  await expect(tip.locator("pre")).toContainText("N_req × L_out / T_s");
  await expect(tip).toContainText("Completed requests");
  const outputRow = () =>
    tip
      .getByText("Output tokens per request", { exact: true })
      .locator('xpath=ancestor::div[contains(@class,"formula-term-row")]')
      .locator("dd");
  await expect(outputRow()).toHaveText(
    String(initial.config.output_tokens) + "tokens",
  );
  const box = await tip.boundingBox();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.keyboard.press("Escape");
  await expect(tip).toHaveCount(0);
  await page.getByLabel("Output length", { exact: true }).fill("8");
  await throughput.focus();
  await expect(tip).toBeVisible();
  // Pending edits must never contaminate a completed-run explanation.
  await expect(outputRow()).toHaveText(
    String(initial.config.output_tokens) + "tokens",
  );
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  await throughput.hover();
  await expect(tip).toBeVisible();
  await expect(outputRow()).toHaveText("8tokens");
  await page.screenshot({
    path:
      "docs/screenshots/" +
      (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
      "formula-throughput.png",
    fullPage: false,
  });
  await page.keyboard.press("Escape");
  const run = await exported(page);
  expect(run.metrics.throughput_tps).toBeCloseTo(
    (run.config.requests * run.config.output_tokens) /
      (run.metrics.elapsed_ms / 1000),
    10,
  );
  await page
    .getByRole("button", { name: "Layer / chunk", exact: true })
    .click();
  const row = page
    .getByRole("row")
    .filter({ hasText: "QKV projection" })
    .first();
  const service = row.locator('[data-formula-key$=".duration_ms"]');
  await service.focus();
  await expect(tip).toBeVisible();
  await expect(tip.locator("pre")).toContainText("T_base = max");
  await expect(tip).toContainText("Fitted intercept");
  const op = run.layers.find(
    (o: any) => o.name === "QKV projection" && o.phase === "prefill",
  );
  const intercept = tip
    .getByText("Fitted intercept", { exact: true })
    .locator('xpath=ancestor::div[contains(@class,"formula-term-row")]')
    .locator("dd");
  await expect(intercept).toContainText(
    op.fit_parameters.coefficients[0].toLocaleString("en-US", {
      maximumSignificantDigits: 9,
    }),
  );
  await page.screenshot({
    path:
      "docs/screenshots/" +
      (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
      "formula-operator.png",
    fullPage: false,
  });
  await page.keyboard.press("Escape");
  await row.locator('[data-formula-key$=".memory_ms"]').hover();
  await expect(tip).toContainText("BW_TBps");
  await expect(tip).toContainText("not a decomposition");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Model graph", exact: true }).click();
  await page
    .getByRole("button", { name: "Inspect Attention", exact: true })
    .click();
  await page
    .locator('[data-formula-key^="graph."][data-formula-key$=".duration_ms"]')
    .hover();
  await expect(tip).toContainText("Service time");
  await page.keyboard.press("Escape");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await throughput.click();
  await expect(tip).toBeVisible();
  const mobile = await tip.boundingBox();
  expect(mobile!.x).toBeGreaterThanOrEqual(0);
  expect(mobile!.x + mobile!.width).toBeLessThanOrEqual(390);
  expect(mobile!.y).toBeGreaterThanOrEqual(0);
  expect(mobile!.y + mobile!.height).toBeLessThanOrEqual(844);
  await page.keyboard.press("Escape");
  expect(errors).toEqual([]);
}
