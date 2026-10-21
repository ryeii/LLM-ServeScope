import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
const views = [
  "Overview",
  "Roofline",
  "Layer / chunk",
  "Box graph",
  "KV storage",
  "Model graph",
  "Configuration",
];
async function exported(page: Page, name = "Export run") {
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name, exact: true }).click();
  return JSON.parse(await readFile((await (await pending).path())!, "utf8"));
}
export async function pinnedScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  const results = page.locator(".behavioral-results");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  const original = await exported(page);
  const originalFingerprint = original.provenance.config_sha256;
  await page.getByRole("button", { name: "Pin baseline", exact: true }).click();
  const pinned = page.getByRole("region", { name: "Pinned run", exact: true });
  const newer = page.getByRole("region", { name: "New run", exact: true });
  await expect(pinned).toHaveAttribute("aria-current", "true");
  await expect(newer).toHaveCount(0);
  const baselineViews: Record<string, string> = {};
  for (const name of views) {
    await page.getByRole("button", { name, exact: true }).click();
    baselineViews[name] = await results.innerText();
  }
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.getByLabel("Output length", { exact: true }).fill("32");
  await expect(newer).toBeVisible();
  await expect(
    newer.locator('[data-input-change="output_tokens"]'),
  ).toContainText("Output length");
  await expect(
    newer.locator('[data-input-change="output_tokens"]'),
  ).toContainText("64");
  await expect(
    newer.locator('[data-input-change="output_tokens"]'),
  ).toContainText("32 tokens");
  await expect(pinned).toHaveAttribute("aria-current", "true");
  await expect(results).toHaveAttribute(
    "data-result-fingerprint",
    originalFingerprint,
  );
  await page.getByRole("button", { name: "Decode pool", exact: true }).click();
  await page.getByText("Compute throughput", { exact: true }).click();
  await page.getByLabel("Compute clock scale", { exact: true }).fill("1.25");
  await expect(
    newer.locator('[data-input-change="decode.hardware.clock_scale"]'),
  ).toContainText("Decode · Compute clock scale");
  await expect(newer.locator("[data-input-change]")).toHaveCount(2);
  const originalExport = await exported(page, "Export pinned run");
  expect(originalExport).toEqual(original);
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(newer).toContainText("Results match scenario");
  const current = await exported(page);
  expect(current.config.output_tokens).toBe(32);
  expect(current.config.decode.hardware.clock_scale).toBe(1.25);
  expect(current.provenance.config_sha256).not.toBe(originalFingerprint);
  const fingerprint = current.provenance.config_sha256;
  await expect(newer).toHaveAttribute("aria-current", "true");
  const currentViews: Record<string, string> = {};
  for (const name of views) {
    await page.getByRole("button", { name, exact: true }).click();
    currentViews[name] = await results.innerText();
  }
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.locator('[data-formula-key="metrics.throughput_tps"]').hover();
  await expect(page.locator('[data-cascade-depth="0"]')).toBeVisible();
  await page.keyboard.down("p");
  await expect(results).toHaveAttribute(
    "data-result-fingerprint",
    originalFingerprint,
  );
  await expect(pinned).toHaveAttribute("aria-current", "true");
  await expect(page.locator('[data-cascade-depth="0"]')).toHaveCount(0);
  await page.keyboard.down("p"); // auto-repeat keeps preview active instead of toggling it.
  for (const name of views) {
    await page.getByRole("button", { name, exact: true }).click();
    await expect(results).toHaveAttribute(
      "data-result-fingerprint",
      originalFingerprint,
    );
    expect(
      await results.innerText(),
      `${name} must show the original snapshot`,
    ).toBe(baselineViews[name]);
  }
  await page.keyboard.up("p");
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await expect(newer).toHaveAttribute("aria-current", "true");
  expect(await results.innerText()).toBe(currentViews.Configuration);
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.getByLabel("Output length", { exact: true }).focus();
  await page.keyboard.down("p");
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await page.keyboard.up("p");
  await page
    .getByRole("heading", { name: "Serving analysis", exact: true })
    .click();
  await page.evaluate(() =>
    window.dispatchEvent(
      new KeyboardEvent("keydown", { key: "p", ctrlKey: true, bubbles: true }),
    ),
  );
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await page.keyboard.down("P");
  await expect(results).toHaveAttribute(
    "data-result-fingerprint",
    originalFingerprint,
  );
  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await page.keyboard.up("P");
  const preview = page.getByRole("button", {
    name: "Preview pinned results",
    exact: true,
  });
  // Keyboard preview survives focus leaving the pointer/keyboard preview control.
  await preview.focus();
  await page.keyboard.down("p");
  await page
    .getByRole("button", { name: "Configuration", exact: true })
    .click();
  await expect(results).toHaveAttribute(
    "data-result-fingerprint",
    originalFingerprint,
  );
  await page.keyboard.up("p");
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await preview.hover();
  await page.mouse.down();
  await expect(results).toHaveAttribute(
    "data-result-fingerprint",
    originalFingerprint,
  );
  await page.mouse.move(10, 10);
  await page.mouse.up();
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await page.evaluate(() => window.scrollTo(0, 0));
  if (!process.env.SERVESCOPE_SKIP_SCREENSHOTS)
    await page.screenshot({
      path: `docs/screenshots/${page.url().includes("/LLM-ServeScope/") ? "pages-" : ""}pinned-comparison.png`,
    });
  await page.getByLabel("Output length", { exact: true }).fill("0");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await page
    .getByRole("heading", { name: "Serving analysis", exact: true })
    .click();
  await page.keyboard.down("p");
  await expect(results).toHaveAttribute(
    "data-result-fingerprint",
    originalFingerprint,
  );
  await page
    .getByRole("button", { name: "Unpin baseline", exact: true })
    .click();
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await expect(page.locator(".run-comparison")).toHaveCount(0);
  await page.keyboard.up("p");
  await page
    .getByRole("button", { name: "Dismiss error", exact: true })
    .click();
  await page.getByLabel("Output length", { exact: true }).fill("32");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Pin baseline", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Inputs", exact: true }).click();
  await page.getByLabel("Output length", { exact: true }).fill("16");
  await page.getByRole("button", { name: "Inputs", exact: true }).click();
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(newer).toContainText("Results match scenario");
  const mobileFingerprint = await results.getAttribute(
    "data-result-fingerprint",
  );
  await preview.hover();
  await page.mouse.down();
  await expect(results).toHaveAttribute("data-result-fingerprint", fingerprint);
  await page.mouse.up();
  await expect(results).toHaveAttribute(
    "data-result-fingerprint",
    mobileFingerprint!,
  );
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(391);
  await page.evaluate(() => window.scrollTo(0, 0));
  if (!process.env.SERVESCOPE_SKIP_SCREENSHOTS)
    await page.screenshot({
      path: `docs/screenshots/${page.url().includes("/LLM-ServeScope/") ? "pages-" : ""}pinned-comparison-mobile.png`,
    });
  expect(errors).toEqual([]);
}
