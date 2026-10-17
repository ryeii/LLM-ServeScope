import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("six outputs, scenario fingerprint, reproducible exports and capacity errors", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: "docs/screenshots/overview.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Pin baseline", exact: true }).click();
  await page.getByLabel("Output length", { exact: true }).fill("32");
  await expect(page.getByText("Inputs changed · run to update")).toBeVisible();
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  await expect(page.getByText("vs pinned").first()).toBeVisible();
  for (const [name, file, check] of [
    ["Roofline", "roofline", "Compute or bandwidth?"],
    ["Layer / chunk", "layers", "Layer and chunk detail"],
    ["Box graph", "boxes", "Behavior over time"],
    ["Model graph", "computation-graph", "From architecture to operations"],
    ["Configuration", "configuration", "A reproducible scenario"],
  ]) {
    await page.getByRole("button", { name, exact: true }).click();
    await expect(
      page.getByRole("heading", { name: check, exact: true }),
    ).toBeVisible();
    if (name === "Model graph") {
      await page
        .getByRole("button", { name: "Inspect Attention", exact: true })
        .click();
      await expect(page.locator(".node-detail h3")).toHaveText("Attention");
    }
    await page.screenshot({
      path: `docs/screenshots/${file}.png`,
      fullPage: true,
    });
  }
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  const d = await download;
  const result = JSON.parse(await readFile((await d.path())!, "utf8"));
  expect(result.metrics.throughput_tps).toBeGreaterThan(0);
  expect(result.config.output_tokens).toBe(32);
  expect(result.provenance.kind).toBe("behavioral_extension");
  await page.getByLabel("Import scenario", { exact: true }).setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from("{}"),
  });
  await expect(page.getByRole("alert")).toContainText(
    "Choose a supported model",
  );
  await expect(page.locator(".app")).toBeVisible();
  await page.getByText("Memory + tile shape", { exact: true }).click();
  await page.getByLabel("Main memory", { exact: true }).fill("1");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("above 1 GB");
  expect(errors).toEqual([]);
});

test("native simulator runs from UI and returns actual Chakra graph and request CSV", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "LLMServingSim 2.0", exact: true })
    .click();
  await expect(
    page.getByText("Native backend ready", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Run native simulator", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "LLMServingSim results", exact: true }),
  ).toBeVisible({ timeout: 85000 });
  await expect(
    page.getByRole("img", {
      name: "Actual native Chakra computation DAG",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText("294 nodes · 293 dependency edges.", { exact: false }),
  ).toBeVisible();
  const requestLink = page.getByRole("link", {
    name: "Native request CSV",
    exact: true,
  });
  const response = await page.request.get(
    (await requestLink.getAttribute("href"))!,
  );
  expect(response.ok()).toBeTruthy();
  expect(await response.text()).toContain("TTFT");
  await page.screenshot({
    path: "docs/screenshots/native.png",
    fullPage: true,
  });
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export JSON", exact: true }).click();
  const d = await download;
  const r = JSON.parse(await readFile((await d.path())!, "utf8"));
  expect(r.engine).toContain("KAIST");
  expect(r.graphs[0].nodes.length).toBe(294);
  expect(r.metrics.mean_ttft_ms).toBeGreaterThan(0);
  await page
    .getByLabel("Native tensor parallel", { exact: true })
    .selectOption("2");
  await page.getByLabel("Native link bandwidth", { exact: true }).fill("1");
  await page
    .getByRole("button", { name: "Run native simulator", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "LLMServingSim results", exact: true }),
  ).toBeHidden();
  await expect(
    page.getByRole("heading", { name: "LLMServingSim results", exact: true }),
  ).toBeVisible({ timeout: 85000 });
  await expect(
    page.getByText(/^360 nodes · 359 dependency edges\./),
  ).toBeVisible();
  const secondDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export JSON", exact: true }).click();
  const sd = await secondDownload;
  const slow = JSON.parse(await readFile((await sd.path())!, "utf8"));
  expect(slow.config.tp).toBe(2);
  expect(slow.config.link_bandwidth_gibps).toBe(1);
  expect(slow.metrics.simulated_duration_ms).toBeGreaterThan(250);
  expect(
    slow.graphs[0].nodes.some((n: any) => n.type === "COMM_COLL_NODE"),
  ).toBeTruthy();
  await page.screenshot({
    path: "docs/screenshots/native-network.png",
    fullPage: true,
  });
});

test("mobile layout stays inside viewport and scenario inputs work", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Inputs", exact: true }).click();
  await page
    .getByLabel("Model architecture", { exact: true })
    .selectOption("llama8");
  await page
    .getByLabel("Separate prefill and decode", { exact: true })
    .uncheck();
  await page.getByRole("button", { name: "Inputs", exact: true }).click();
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  await page.screenshot({
    path: "docs/screenshots/mobile.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Model graph", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Transformer computation graph",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "docs/screenshots/mobile-graph.png",
    fullPage: true,
  });
});
