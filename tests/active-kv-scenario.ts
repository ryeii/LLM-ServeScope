import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
async function exported(page: Page) {
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  return JSON.parse(await readFile((await (await pending).path())!, "utf8"));
}
export async function activeKVScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  await page
    .getByLabel("Import scenario", { exact: true })
    .setInputFiles("examples/active-kv-scenario.json");
  await expect(page.getByRole("alert")).toContainText("Imported");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  const run = await exported(page);
  const reference = JSON.parse(
    await readFile("examples/active-kv-result.json", "utf8"),
  );
  expect(run.metrics).toEqual(reference.metrics);
  expect(run.active_kv.stats).toEqual(reference.active_kv.stats);
  expect(run.active_kv.model_source_sha256).toBe(
    reference.active_kv.model_source_sha256,
  );
  expect(run.storage.stats.ssd_write_bytes).toBeGreaterThan(0);
  expect(run.active_kv.stats.ssd_read_bytes).toBeGreaterThan(0);
  const reads = run.timeline.filter(
    (e: any) => e.cause === "active_read" && e.tier === "ssd",
  );
  expect(reads.length).toBeGreaterThan(0);
  for (const e of reads)
    for (const id of e.request_ids)
      expect(e.start_ms).toBeGreaterThanOrEqual(
        run.requests[id].first_token_ms,
      );
  const lanes: any = {};
  for (const e of run.timeline.filter((e: any) => e.kind === "storage_io"))
    (lanes[e.resource] ||= []).push(e);
  expect(new Set(lanes["kv_ssd:0"].map((e: any) => e.scope))).toEqual(
    new Set(["active", "prefix"]),
  );
  for (const lane of Object.values(lanes) as any[])
    for (let i = 1; i < lane.length; i++)
      expect(lane[i].start_ms + 1e-8).toBeGreaterThanOrEqual(
        lane[i - 1].end_ms,
      );
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  await expect(page.locator(".active-request-table tbody tr")).toHaveCount(8);
  await expect(page.locator(".storage-request-table tbody tr")).toHaveCount(8);
  await expect(
    page.getByText(/Unvalidated active-offload composition/),
  ).toBeVisible();
  const card = (d: number) => page.locator(`[data-cascade-depth="${d}"]`);
  await page
    .locator('[data-formula-key="active.attention_read_bytes"]')
    .hover();
  await card(0)
    .getByRole("button", { name: /^Explain active_read/ })
    .first()
    .focus();
  await expect(card(1)).toContainText("FIFO");
  await card(1)
    .getByRole("button", { name: "Explain I/O service function", exact: true })
    .focus();
  await expect(card(2)).toContainText("Linear size coefficient");
  await card(2)
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  await page.getByRole("button", { name: "Box graph", exact: true }).click();
  const stage = run.timeline.findIndex(
    (e: any) =>
      e.phase === "decode" &&
      e.kind !== "active_staging" &&
      e.active_io_event_ids?.length,
  );
  await page.locator(`[data-formula-key="timeline.${stage}"]`).hover();
  await expect(card(0)).toContainText("HBM-reference GPU operators");
  await expect(card(0)).toContainText("active_read");
  await card(0)
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  for (const name of ["Active decode KV", "Retained prefixes", "KV I/O paths"])
    await page
      .getByRole("complementary")
      .getByText(name, { exact: true })
      .click();
  const active = page.getByLabel("Active decode KV placement", { exact: true });
  const prefix = page.getByLabel("Retained prefix cache layout", {
    exact: true,
  });
  await expect(active.locator("option")).toHaveCount(5);
  await expect(active).toHaveValue("ssd");
  await expect(prefix).toHaveValue("ssd-only");
  await prefix.selectOption("hbm-only");
  await expect(active).toHaveValue("ssd");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  const mixed = await exported(page);
  expect(mixed.config.prefill.batch_size).toBe(2);
  expect(mixed.config.decode.batch_size).toBe(2);
  expect(mixed.storage.stats.ssd_write_bytes || 0).toBe(0);
  expect(mixed.active_kv.stats.ssd_read_bytes).toBeGreaterThan(0);
  await active.selectOption("hbm-dram");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  const hybrid = await exported(page);
  expect(hybrid.active_kv.peak_resident_bytes_per_gpu).toBeLessThanOrEqual(
    0.05e9,
  );
  expect(hybrid.active_kv.stats.attention_read_bytes).toBeGreaterThan(0);
  expect(hybrid.storage.placement).toBe("hbm-only");
  if (!process.env.SERVESCOPE_SKIP_SCREENSHOTS)
    await page.screenshot({
      path:
        "docs/screenshots/" +
        (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
        "active-kv.png",
      fullPage: true,
    });
  await active.selectOption("hbm");
  await expect(prefix).toHaveValue("hbm-only");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  const resident = await exported(page);
  expect(resident.active_kv).toBeUndefined();
  expect(resident.storage.placement).toBe("hbm-only");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Inputs", exact: true }).click();
  await active.selectOption("dram");
  await page.getByRole("button", { name: "Inputs", exact: true }).click();
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Active decode KV", exact: true }),
  ).toBeVisible();
  expect((await exported(page)).active_kv.placement).toBe("dram");
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(391);
  if (!process.env.SERVESCOPE_SKIP_SCREENSHOTS)
    await page.screenshot({
      path:
        "docs/screenshots/" +
        (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
        "active-kv-mobile.png",
      fullPage: true,
    });
  expect(errors).toEqual([]);
}
