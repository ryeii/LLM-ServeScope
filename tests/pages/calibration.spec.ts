import { test } from "@playwright/test";
import { calibrationScenario } from "../calibration-scenario";

test("fitted GPU function matches local Python and the original roofline remains available", async ({
  page,
}) => {
  await calibrationScenario(page);
});

test("FP16 Vidur cohort matches local Python and exposes missing-operator extrapolation", async ({
  page,
}) => {
  const { expect } = await import("@playwright/test");
  const { readFile } = await import("node:fs/promises");
  await page.goto("./");
  await expect(
    page.getByLabel("Operator timing model", { exact: true }),
  ).toHaveValue("calibrated", { timeout: 60000 });
  await page
    .getByLabel("Import scenario", { exact: true })
    .setInputFiles("examples/vidur-reference.json");
  await expect(page.getByRole("alert")).toContainText("Imported.");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  const result = JSON.parse(
    await readFile((await (await download).path())!, "utf8"),
  );
  const local = JSON.parse(
    await readFile("examples/vidur-reference-result.json", "utf8"),
  );
  for (const key of ["throughput_tps", "ttft_mean_ms", "tpot_mean_ms"])
    expect(result.metrics[key]).toBeCloseTo(local.metrics[key], 8);
  const qkv = result.phases.decode.operations.find(
    (o: any) => o.name === "QKV projection",
  );
  expect(qkv.fit_cohort).toBe("vidur");
  expect(qkv.timing_source).toBe("fitted-profiled-configuration");
  expect(
    result.calibration.operator_sources.some((o: any) =>
      o.fallback_reason?.includes("absent"),
    ),
  ).toBeTruthy();
  await expect(
    page.getByRole("link", { name: "Accuracy", exact: true }),
  ).toBeVisible();
});
