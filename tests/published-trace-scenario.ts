import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

export async function publishedTraceScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  const config = JSON.parse(
    await readFile("examples/mooncake-conversation-scenario.json", "utf8"),
  );
  const reference = JSON.parse(
    await readFile(
      "examples/mooncake-conversation-result-summary.json",
      "utf8",
    ),
  );
  await page
    .getByLabel("Import scenario", { exact: true })
    .setInputFiles("examples/mooncake-conversation-scenario.json");
  await expect(page.getByRole("alert")).toContainText("Imported");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  const run = JSON.parse(
    await readFile((await (await pending).path())!, "utf8"),
  );
  expect(run.storage.timing_evidence).toBe(
    "configured-assumptions-not-request-trace-measurements",
  );
  expect(run.storage.fit).toBeUndefined();
  expect(run.storage.workload_source.source_file_sha256).toBe(
    config.storage.source_file_sha256,
  );
  expect(run.config.storage.workload).toEqual(config.storage.workload);
  for (const [key, value] of Object.entries(reference.metrics))
    expect(run.metrics[key], key).toBeCloseTo(value as number, 8);
  expect(run.workload_totals).toEqual(reference.workload_totals);
  for (let i = 0; i < run.requests.length; i++) {
    const original = config.storage.workload[i];
    expect(run.requests[i].arrival_ms).toBe(
      original.timestamp - config.storage.workload[0].timestamp,
    );
    expect(run.requests[i].input_tokens).toBe(original.input_length);
    expect(run.requests[i].output_tokens).toBe(original.output_length);
    expect(run.requests[i].token_times_ms.length).toBe(original.output_length);
  }
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  await expect(page.locator(".storage-request-table tbody tr")).toHaveCount(4);
  await expect(
    page.getByRole("heading", {
      name: "Trace evidence and timing assumptions",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator("footer")).toContainText(
    "Published request traces: Mooncake",
  );
  expect(errors).toEqual([]);
}
