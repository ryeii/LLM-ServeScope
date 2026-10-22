import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

async function exported(page: Page) {
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  return JSON.parse(await readFile((await (await pending).path())!, "utf8"));
}

export async function gdsScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  const initial = await exported(page);
  expect(initial.schema_version).toBe("servescope.behavioral.v6");
  expect(initial.provenance.kv_transfer_model_sha256).toHaveLength(64);
  expect(initial.config.active_kv.placement).toBe("hbm");
  expect(initial.storage.placement).toBe("dynamo");
  expect(initial.storage.publication_mode).toBe("async");
  expect(initial.storage.ssd_transfer_mode).toBe("gds");
  await expect(
    page.getByRole("heading", {
      name: "Executed resource service",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText("Nominal GPU resource work · analytical diagnostic", {
      exact: true,
    }),
  ).toBeVisible();
  const sidebar = page.getByRole("complementary");
  for (const title of ["Active decode KV", "Retained prefixes", "KV I/O paths"])
    await sidebar.getByText(title, { exact: true }).click();
  await expect(
    page.getByLabel("Active decode KV placement", { exact: true }),
  ).toHaveValue("hbm");
  await expect(
    page.getByLabel("SSD transfer path", { exact: true }),
  ).toHaveValue("gds");
  await expect(
    page.getByLabel("Prefix publication", { exact: true }),
  ).toHaveValue("async");

  await page
    .getByLabel("Import scenario", { exact: true })
    .setInputFiles("examples/gds-scenario.json");
  await expect(page.getByRole("alert")).toContainText("Imported");
  const references = JSON.parse(
    await readFile("examples/gds-comparison.json", "utf8"),
  );
  const run = async () => {
    await page
      .getByRole("button", { name: "Run scenario", exact: true })
      .click();
    await expect(
      page.getByText("Results match scenario", { exact: true }),
    ).toBeVisible();
    return exported(page);
  };
  const direct = await run();
  expect(direct.metrics).toEqual(references.results[0].metrics);
  expect(direct.storage.stats).toEqual(references.results[0].storage_stats);
  expect(direct.timeline.some((e: any) => e.tier === "pcie")).toBe(false);
  expect(direct.timeline.filter((e: any) => e.gds).length).toBeGreaterThan(0);
  await page.locator('[data-formula-key="service.SSD ↔ GPU · GDS"]').hover();
  await page
    .locator('[data-cascade-depth="0"]')
    .getByRole("button", { name: /^Explain Event/ })
    .first()
    .focus();
  await expect(page.locator('[data-cascade-depth="1"]')).toContainText(
    "max(storage_ms, link_ms)",
  );
  await page
    .locator('[data-cascade-depth="1"]')
    .getByRole("button", { name: "Explain SSD storage service", exact: true })
    .focus();
  await expect(page.locator('[data-cascade-depth="2"]')).toContainText(
    "Linear size coefficient",
  );
  await page
    .locator('[data-cascade-depth="2"]')
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  await page
    .getByLabel("SSD transfer path", { exact: true })
    .selectOption("host-staged");
  const staged = await run();
  expect(staged.metrics).toEqual(references.results[1].metrics);
  expect(staged.storage.stats).toEqual(references.results[1].storage_stats);
  expect(staged.timeline.some((e: any) => e.tier === "pcie")).toBe(true);
  await page
    .getByLabel("SSD transfer path", { exact: true })
    .selectOption("gds");
  await expect(
    page.getByLabel("SSD service model", { exact: true }),
  ).toHaveValue("manual");
  await expect(
    page.getByLabel("SSD service model", { exact: true }).locator("option"),
  ).toHaveCount(1);

  await page
    .getByRole("button", { name: "Apply Dynamo-style setup", exact: true })
    .click();
  const aligned = await run();
  expect(aligned.config.active_kv.placement).toBe("hbm");
  expect(aligned.storage.placement).toBe("dynamo");
  expect(aligned.storage.publication_mode).toBe("async");
  expect(aligned.config.input_tokens).toBe(direct.config.input_tokens);
  expect(aligned.timeline.some((e: any) => e.cause === "active_read")).toBe(
    false,
  );
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  await expect(page.getByText(/GDS direct path · configured/)).toBeVisible();
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  if (!process.env.SERVESCOPE_SKIP_SCREENSHOTS) {
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({
      path:
        "docs/screenshots/" +
        (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
        "gds.png",
      fullPage: true,
    });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", {
      name: "Executed resource service",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
}
