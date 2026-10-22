import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

export async function prefixReuseScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  const sidebar = page.getByRole("complementary");
  for (const label of ["Active decode KV", "Retained prefixes", "KV I/O paths"])
    await expect(sidebar.getByText(label, { exact: true })).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "KV storage", exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".resource-service")).not.toContainText(
    /SSD|GDS|Host ↔ GPU/,
  );
  await expect(
    page.getByLabel("Prefix reuse assumption", { exact: true }),
  ).toHaveValue("none");
  const references = JSON.parse(
    await readFile("examples/prefix-reuse-comparison.json", "utf8"),
  );
  const exported = async () => {
    const pending = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export run", exact: true }).click();
    return JSON.parse(await readFile((await (await pending).path())!, "utf8"));
  };
  const baseline = await exported();
  for (const source of ["conversation", "toolagent", "synthetic"]) {
    await page
      .getByLabel("Prefix reuse assumption", { exact: true })
      .selectOption(source);
    await page
      .getByRole("button", { name: "Run scenario", exact: true })
      .click();
    await expect(
      page.getByText("Results match scenario", { exact: true }),
    ).toBeVisible();
    const run = await exported();
    expect(run.prefix_reuse).toEqual(references[source].prefix_reuse);
    for (const [key, value] of Object.entries(references[source].metrics))
      expect(run.metrics[key], source + ":" + key).toBeCloseTo(
        value as number,
        8,
      );
    expect(run.memory).toEqual(baseline.memory);
    expect(run.storage).toBeUndefined();
    expect(run.active_kv).toBeUndefined();
    expect(run.config.storage).toBeUndefined();
    expect(run.config.active_kv).toBeUndefined();
    await expect(
      page.getByRole("region", { name: "Prefix reuse assumption in this run" }),
    ).toContainText("ideal HBM availability assumption");
    await expect(
      page.locator('[data-formula-key="prefix.reused_tokens"]'),
    ).toContainText(
      run.prefix_reuse.reused_tokens_per_request.toLocaleString(),
    );
  }
  await page.locator('[data-formula-key="prefix.reused_tokens"]').hover();
  await expect(page.locator('[data-cascade-depth="0"]')).toContainText(
    "floor(L_in × f_trace)",
  );
  await page
    .locator('[data-cascade-depth="0"]')
    .getByRole("button", {
      name: "Explain Token-weighted earlier-arrival reuse potential",
      exact: true,
    })
    .focus();
  await expect(page.locator('[data-cascade-depth="1"]')).toContainText(
    "SHA-256",
  );
  await page
    .locator('[data-cascade-depth="1"]')
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  const invalid = { ...baseline.config, storage: { enabled: true } };
  await page
    .getByLabel("Import scenario", { exact: true })
    .setInputFiles({
      name: "retired.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(invalid)),
    });
  await expect(page.getByRole("alert")).toContainText("retired KV placement");
  await expect(page.locator("footer")).toContainText(
    "Prefix-reuse workload evidence: Mooncake",
  );
  await expect(page.locator("footer")).not.toContainText("Dynamo");
  if (!process.env.SERVESCOPE_SKIP_SCREENSHOTS)
    await page.screenshot({
      path:
        "docs/screenshots/" +
        (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
        "prefix-reuse.png",
      fullPage: true,
    });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
}
