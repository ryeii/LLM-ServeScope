import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

async function exported(page: any) {
  const ready = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  return JSON.parse(await readFile((await (await ready).path())!, "utf8"));
}

test("heterogeneous MoE servers preserve separate hardware, batching, engine and memory policies", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByLabel("Model architecture", { exact: true }).locator("option"),
  ).toHaveCount(14);
  await expect(
    page.getByLabel("Hardware preset", { exact: true }).locator("option"),
  ).toHaveCount(10);
  await page
    .getByLabel("Model architecture", { exact: true })
    .selectOption("qwen30");
  await page.getByLabel("Input length", { exact: true }).fill("256");
  await page.getByLabel("Output length", { exact: true }).fill("8");
  await page
    .getByLabel("KV cache precision", { exact: true })
    .selectOption("fp8");
  await page
    .getByLabel("Hardware preset", { exact: true })
    .selectOption("h200");
  await page.getByLabel("EP", { exact: true }).fill("2");
  await page.getByLabel("Prefill microbatch", { exact: true }).fill("6");
  await page
    .getByLabel("Server engine policy", { exact: true })
    .selectOption("trtllm");
  await page.getByLabel("Runtime reservation", { exact: true }).fill("3");
  await page.getByRole("button", { name: "Decode pool", exact: true }).click();
  await page
    .getByLabel("Hardware preset", { exact: true })
    .selectOption("a100");
  await page.getByLabel("EP", { exact: true }).fill("2");
  await page.getByLabel("Decode batch cap", { exact: true }).fill("2");
  await page.getByLabel("Active batches / replica", { exact: true }).fill("1");
  await page.getByLabel("Memory allocation limit", { exact: true }).fill("0.9");
  await page.getByLabel("Runtime reservation", { exact: true }).fill("2");
  await page
    .getByLabel("Server engine policy", { exact: true })
    .selectOption("eager");
  await page.getByLabel("KV transfer channels", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  const r = await exported(page);
  expect(r.config.prefill.hardware_id).toBe("h200");
  expect(r.config.decode.hardware_id).toBe("a100");
  expect(r.config.prefill.engine).toBe("trtllm");
  expect(r.config.decode.engine).toBe("eager");
  expect(r.config.prefill.batch_size).toBe(6);
  expect(r.config.decode.batch_size).toBe(2);
  expect(r.config.decode.max_concurrent_batches).toBe(1);
  expect(r.config.kv_precision).toBe("fp8");
  expect(r.memory.decode.capacity_GB).toBe(70);
  expect(r.requests).toHaveLength(16);
  expect(
    r.timeline
      .filter((e: any) => e.phase === "decode")
      .every((e: any) => e.batch_size <= 2),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Model graph", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Inspect Expert router", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Inspect QK normalization", exact: true }),
  ).toBeVisible();
  await page.getByText("Model & workload", { exact: true }).click();
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.screenshot({
    path: "docs/screenshots/server-options.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Prefill pool", exact: true }).click();
  await expect(page.getByLabel("Hardware preset", { exact: true })).toHaveValue(
    "h200",
  );
  await expect(
    page.getByLabel("Prefill microbatch", { exact: true }),
  ).toHaveValue("6");
  await expect(
    page.getByLabel("Runtime reservation", { exact: true }),
  ).toHaveValue("3");
});

test("custom architecture editor changes computed geometry and survives JSON reimport", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByLabel("Model architecture", { exact: true })
    .selectOption("qwen4");
  await page
    .getByRole("button", { name: "Customize architecture", exact: true })
    .click();
  await page
    .getByLabel("Model name", { exact: true })
    .fill("Custom MHA decoder");
  for (const [label, value] of [
    ["Transformer layers", "8"],
    ["Hidden width", "1024"],
    ["FFN / expert width", "2048"],
    ["Attention heads", "8"],
    ["KV heads", "8"],
    ["Vocabulary size", "4096"],
  ]) {
    await page.getByLabel(label, { exact: true }).fill(value);
  }
  await page.getByLabel("QK normalization", { exact: true }).uncheck();
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  const r = await exported(page);
  expect(r.model.name).toBe("Custom MHA decoder");
  expect(r.model.layers).toBe(8);
  expect(r.model.hidden).toBe(1024);
  expect(r.model.kv_heads).toBe(r.model.heads);
  expect(r.model.tie_embeddings).toBe(true);
  expect(r.model.source_revision).toBe("user");
  expect(r.metrics.throughput_tps).toBeGreaterThan(0);
  await page.getByLabel("Import scenario", { exact: true }).setInputFiles({
    name: "custom.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(r.config)),
  });
  await expect(page.getByLabel("Model name", { exact: true })).toHaveValue(
    "Custom MHA decoder",
  );
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  const replay = await exported(page);
  expect(replay.provenance.config_sha256).toBe(r.provenance.config_sha256);
  expect(replay.metrics).toEqual(r.metrics);
  await page
    .getByRole("heading", { name: "Scenario inputs", exact: true })
    .scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "docs/screenshots/custom-model.png",
    fullPage: true,
  });
});
