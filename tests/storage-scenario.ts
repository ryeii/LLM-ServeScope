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
  await page
    .getByRole("complementary")
    .getByText("KV I/O paths", { exact: true })
    .click();
  await page
    .getByLabel("SSD service model", { exact: true })
    .selectOption("local-mooncake-path");
  await page
    .getByRole("complementary")
    .getByText("Retained prefixes", { exact: true })
    .click();
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
  await page
    .getByLabel("Retained prefix cache layout", { exact: true })
    .selectOption("disabled");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  const off = await exported(page);
  expect(off.workload_totals.output_tokens).toBe(4);
  expect(off.requests[1].arrival_ms).toBe(10000);
  expect(off.requests[1].input_tokens).toBe(1536);
  expect(off.storage.retention_enabled).toBe(false);
  expect(off.storage.stats.reused_tokens || 0).toBe(0);
  await expect(page.getByLabel("Input length", { exact: true })).toBeDisabled();
  await page
    .getByLabel("Retained prefix cache layout", { exact: true })
    .selectOption("ssd-only");

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

export async function storageLayoutsScenario(page: Page) {
  await page.goto("./");
  await expect(
    page.getByLabel("Operator timing model", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  await page
    .getByRole("button", { name: "Load cache demo", exact: true })
    .click();
  await page
    .getByRole("complementary")
    .getByText("Retained prefixes", { exact: true })
    .click();
  const layout = page.getByLabel("Retained prefix cache layout", {
    exact: true,
  });
  await expect(layout).toHaveValue("ssd-only");
  await expect(layout.locator("option")).toHaveCount(13);
  for (const tier of ["hbm", "dram", "ssd"]) {
    await layout.selectOption(tier + "-only");
    await expect(
      page.getByLabel("Retained HBM / GPU", { exact: true }),
    ).toHaveCount(tier === "hbm" ? 1 : 0);
    await expect(
      page.getByLabel("SSD service model", { exact: true }),
    ).toHaveCount(1);
    await page
      .getByRole("button", { name: "Run scenario", exact: true })
      .click();
    await expect(
      page.getByText("Results match scenario", { exact: true }),
    ).toBeVisible();
    const run = await exported(page);
    expect(run.storage.placement).toBe(tier + "-only");
    for (const other of ["hbm", "dram", "ssd"].filter((t) => t !== tier))
      expect(run.storage.capacity_blocks[other]).toBe(0);
    expect(run.requests[4].cache.hit_tiers).toEqual({ [tier]: 2 });
    const paths = [
      ...new Set(
        run.timeline
          .filter((e: any) => e.kind === "storage_io")
          .map((e: any) => e.tier),
      ),
    ].sort();
    expect(paths).toEqual(
      tier === "hbm" ? ["hbm"] : tier === "dram" ? ["pcie"] : ["pcie", "ssd"],
    );
  }
  for (const id of ["hbm-ssd", "dram-hbm", "hbm-dram-ssd"]) {
    await layout.selectOption(id);
    const primary =
      id === "dram-hbm" ? "Host DRAM cache" : "Retained HBM / GPU";
    await page.getByLabel(primary, { exact: true }).fill("0.067108864");
    if (id === "dram-hbm")
      await page.getByLabel("Retained HBM / GPU", { exact: true }).fill("1");
    if (id === "hbm-dram-ssd")
      await page
        .getByLabel("Host DRAM cache", { exact: true })
        .fill("0.067108864");
    await page
      .getByRole("button", { name: "Run scenario", exact: true })
      .click();
    await expect(
      page.getByRole("heading", {
        name: "Eviction spill transfers",
        exact: true,
      }),
    ).toBeVisible();
    await expect(
      page.getByText("Results match scenario", { exact: true }),
    ).toBeVisible();
    const run = await exported(page);
    expect(run.storage.stats.spill_blocks).toBeGreaterThan(0);
    expect(run.timeline.some((e: any) => e.cause === "spill")).toBe(true);
    if (id === "dram-hbm") {
      expect(run.storage.stats.dram_to_hbm_spill_blocks).toBeGreaterThan(0);
      expect(run.timeline.some((e: any) => e.tier === "ssd")).toBe(false);
    }
    if (id === "hbm-dram-ssd") {
      expect(run.storage.stats.hbm_to_dram_spill_blocks).toBeGreaterThan(0);
      expect(run.storage.stats.dram_to_ssd_spill_blocks).toBeGreaterThan(0);
      const python = JSON.parse(
        await readFile("examples/storage-spill-result.json", "utf8"),
      );
      expect(run.metrics.elapsed_ms).toBeCloseTo(python.metrics.elapsed_ms, 9);
      expect(run.storage.stats.ssd_service_ms).toBeCloseTo(
        python.storage.stats.ssd_service_ms,
        9,
      );
      expect(run.storage.spill_transitions).toEqual(
        python.storage.spill_transitions,
      );
      await page.locator('[data-formula-key="storage.request.3.ttft"]').hover();
      await page
        .locator('[data-cascade-depth="0"]')
        .getByRole("button", {
          name: "Explain First token release",
          exact: true,
        })
        .focus();
      await page
        .locator('[data-cascade-depth="1"]')
        .getByRole("button", {
          name: "Explain Cache publication and eviction spill",
          exact: true,
        })
        .focus();
      await expect(page.locator('[data-cascade-depth="2"]')).toContainText(
        "ssd write",
      );
      await page
        .locator('[data-cascade-depth="2"]')
        .getByRole("button", { name: "Close explanations", exact: true })
        .click();
    }
  }
  await layout.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path:
      "docs/screenshots/" +
      (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
      "storage-layouts.png",
    fullPage: true,
  });
  await layout.selectOption("none");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  const baseline = await exported(page);
  expect(baseline.storage.stats.reused_tokens || 0).toBe(0);
  expect(baseline.timeline.some((e: any) => e.kind === "storage_io")).toBe(
    false,
  );
  expect(baseline.config.prefill.batch_size).toBe(1);
  await layout.selectOption("disabled");
  await expect(
    page.getByLabel("SSD service model", { exact: true }),
  ).toHaveCount(1);
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  expect((await exported(page)).storage).toBeUndefined();
}

export async function prefixBaselineScenario(page: Page) {
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  const baseline = await exported(page);
  const withoutStorage = (config: any) => {
    const { storage, ...rest } = config;
    return rest;
  };
  expect(baseline.config.prefill.batch_size).toBe(4);
  expect(baseline.config.decode.batch_size).toBe(4);
  expect(baseline.metrics.throughput_tps).toBeCloseTo(186.63494412484803, 9);
  await page.getByRole("button", { name: "KV storage", exact: true }).click();
  await page
    .getByRole("complementary")
    .getByText("Retained prefixes", { exact: true })
    .click();
  const layout = page.getByLabel("Retained prefix cache layout", {
    exact: true,
  });
  await expect(page.getByRole("complementary")).toContainText(
    "Active decode KV",
  );
  const references = JSON.parse(
    await readFile("examples/prefix-layout-comparison.json", "utf8"),
  );
  for (const id of ["none", "hbm-only", "ssd-only", "disabled"]) {
    await layout.selectOption(id);
    await page
      .getByRole("button", { name: "Run scenario", exact: true })
      .click();
    await expect(
      page.getByText("Results match scenario", { exact: true }),
    ).toBeVisible();
    const run = await exported(page);
    expect(withoutStorage(run.config)).toEqual(withoutStorage(baseline.config));
    if (id === "none" || id === "disabled")
      expect(run.metrics).toEqual(baseline.metrics);
    else
      expect(run.metrics.throughput_tps).toBeCloseTo(
        references.layouts[id].metrics.throughput_tps,
        9,
      );
    if (id !== "disabled") {
      expect(run.storage.active_kv_location).toBe("hbm");
      expect(run.storage.active_kv_offload).toBe(false);
      await expect(
        page.getByRole("heading", {
          name: "Active decode KV · HBM only",
          exact: true,
        }),
      ).toBeVisible();
      for (const e of run.timeline.filter(
        (e: any) => e.kind === "storage_io",
      )) {
        const request = run.requests.find(
          (q: any) => q.id === e.request_ids[0],
        );
        expect(e.end_ms).toBeLessThanOrEqual(request.first_token_ms + 1e-9);
      }
    }
  }
}
