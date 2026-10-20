import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

export async function calibrationScenario(page: Page) {
  await page.goto("./");
  await expect(
    page.getByLabel("Operator timing model", { exact: true }),
  ).toHaveValue("calibrated", { timeout: 60000 });
  await page
    .getByLabel("Import scenario", { exact: true })
    .setInputFiles("examples/calibration-reference.json");
  await expect(page.getByRole("alert")).toContainText("Imported.");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByLabel("Timing calibration", { exact: true }),
  ).toContainText("Fitted GPU function");
  const exported = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  const result = JSON.parse(
    await readFile((await (await exported).path())!, "utf8"),
  );
  const local = JSON.parse(
    await readFile("examples/calibration-reference-result.json", "utf8"),
  );
  for (const key of ["throughput_tps", "ttft_mean_ms", "tpot_mean_ms"])
    expect(result.metrics[key]).toBeCloseTo(local.metrics[key], 8);
  const qkv = result.phases.decode.operations.find(
    (o: any) => o.name === "QKV projection",
  );
  expect(qkv.duration_ms).toBeCloseTo(
    local.phases.decode.operations.find((o: any) => o.name === "QKV projection")
      .duration_ms,
    12,
  );
  expect(qkv.fit_features_ns).toHaveLength(3);
  expect(qkv.timing_source).toBe("fitted-profiled-configuration");
  expect(result.calibration.dataset.data_sha256).toHaveLength(64);
  await page
    .getByRole("button", { name: "Layer / chunk", exact: true })
    .click();
  await expect(
    page.getByRole("columnheader", { name: "Timing source", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page
    .getByLabel("Operator timing model", { exact: true })
    .selectOption("roofline");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByLabel("Timing calibration", { exact: true }),
  ).toContainText("Original roofline");
  const legacyExport = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  const legacy = JSON.parse(
    await readFile((await (await legacyExport).path())!, "utf8"),
  );
  expect(legacy.metrics.throughput_tps).toBeCloseTo(281.0797355442539, 8);
  expect(legacy.metrics.throughput_tps).toBeLessThan(
    result.metrics.throughput_tps,
  );
}
