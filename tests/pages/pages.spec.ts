import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("static Pages runs the same Python SimPy model without an API or external asset service", async ({
  page,
  baseURL,
}) => {
  const errors: string[] = [];
  const apiRequests: string[] = [];
  const remoteRequests: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    const u = new URL(request.url());
    if (u.pathname.includes("/api/")) apiRequests.push(request.url());
    if (u.origin !== new URL(baseURL!).origin)
      remoteRequests.push(request.url());
  });
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  await expect(page).toHaveTitle(/LLM-ServeScope/);
  await expect(
    page.getByText("Browser simulation", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Behavioral simulation, not mechanistic simulation.", {
      exact: true,
    }),
  ).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export run", exact: true }).click();
  const result = JSON.parse(
    await readFile((await (await download).path())!, "utf8"),
  );
  const local = JSON.parse(
    await readFile("examples/default-result.json", "utf8"),
  );
  for (const [key, value] of Object.entries(local.metrics)) {
    expect(result.metrics[key], key).toBeCloseTo(value as number, 8);
  }
  expect(result.provenance.config_sha256).toBe(local.provenance.config_sha256);
  expect(result.provenance.execution_runtime).toContain("Pyodide");
  expect(result.requests).toHaveLength(16);
  expect(result.timeline).toHaveLength(local.timeline.length);
  await expect(page.locator(".brand")).toHaveAttribute(
    "href",
    "/LLM-ServeScope/",
  );
  // The README screenshot is captured from the live deployment after publishing,
  // not overwritten by a localhost regression test.
  const logo = page.locator(".brand-mark img");
  await expect(logo).toBeVisible();
  expect(
    await logo.evaluate(
      (image: HTMLImageElement) => image.complete && image.naturalWidth > 0,
    ),
  ).toBeTruthy();
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute(
    "href",
    "/LLM-ServeScope/brand/servescope-serving-telescope.png",
  );
  expect(apiRequests).toEqual([]);
  expect(remoteRequests).toEqual([]);
  expect(errors).toEqual([]);
});

test("Pages supports scenario changes and six output families and explains the local native engine", async ({
  page,
}) => {
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  await page.getByLabel("Output length", { exact: true }).fill("8");
  await page
    .getByLabel("Model architecture", { exact: true })
    .selectOption("qwen4");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
  for (const [view, title] of [
    ["Roofline", "Performance roofline"],
    ["Layer / chunk", "Layer and chunk detail"],
    ["Box graph", "Execution timeline"],
    ["Model graph", "Computation graph"],
    ["Configuration", "Run configuration"],
  ]) {
    await page.getByRole("button", { name: view, exact: true }).click();
    await expect(
      page.getByRole("heading", { name: title, exact: true }),
    ).toBeVisible();
  }
  await page
    .getByRole("button", { name: "LLMServingSim 2.0", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "LLMServingSim 2.0 · local reference engine",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByText(/Jaehong Cho, Hyunmin Choi, Guseul Heo and Jongse Park/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run native simulator", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.getByLabel("Import scenario", { exact: true }).setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from("{}"),
  });
  await expect(page.getByRole("alert")).toContainText(
    "Choose a supported model",
  );
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByText("Results match scenario")).toBeVisible();
});
