import { expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

export async function inputHelpScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(
    page.getByText("Generation throughput", { exact: true }),
  ).toBeVisible({ timeout: 60000 });
  const sidebar = page.getByRole("complementary"),
    help = page.locator('.scenario-input-help[role="tooltip"]');
  const saved = async () => {
    const pending = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Save scenario", exact: true })
      .click();
    return JSON.parse(await readFile((await (await pending).path())!, "utf8"));
  };
  const baseline = await saved();
  for (const summary of await sidebar
    .locator("details.config-section > summary")
    .all())
    if (
      !(await summary.evaluate(
        (e) => (e.parentElement as HTMLDetailsElement).open,
      ))
    )
      await summary.click();
  // Enumerate actual controls, so new or renamed inputs cannot silently lack help.
  const check = async (controls: ReturnType<Page["locator"]>) => {
    for (const control of await controls.all()) {
      await control.hover();
      await expect(help).toHaveAttribute(
        "data-input-help-label",
        (await control.getAttribute("aria-label"))!,
      );
      await expect(help.locator("h3")).toBeVisible();
      await expect(help.locator("p")).toHaveCount(1);
      expect(
        (await help.locator("p").innerText()).split(/\s+/).length,
      ).toBeLessThanOrEqual(30);
      await expect(
        help.locator("h4,.input-help-value,.input-help-hint"),
      ).toHaveCount(0);
      await expect(help).not.toContainText("A configurable scenario input.");
      await expect(control).toHaveAttribute(
        "aria-describedby",
        (await help.getAttribute("id"))!,
      );
      await page.keyboard.press("Escape");
      await expect(help).toHaveCount(0);
    }
  };
  await check(sidebar.locator('input:not([type="file"]),select'));
  expect(await saved()).toEqual(baseline);
  const clock = page.getByLabel("Compute clock scale", { exact: true });
  await clock.scrollIntoViewIfNeeded();
  await clock.focus();
  await expect(help).toContainText("Prefill · Compute clock scale");
  await expect(help.locator("p")).toHaveText(
    "A multiplier on tensor/vector compute peaks for the selected pool; 2× represents twice the modeled compute rate.",
  );
  await clock.fill("");
  await expect(clock).toHaveValue("");
  await clock.fill("2");
  await expect(clock).toHaveValue("2");
  await page.keyboard.press("Escape");
  await expect(clock).toHaveValue("2");
  await page.getByRole("button", { name: "Decode pool", exact: true }).click();
  await clock.hover();
  await expect(help).toContainText("Decode · Compute clock scale");
  await expect(clock).toHaveValue("1");
  await page.keyboard.press("Escape");
  await page
    .getByLabel("Prefix reuse assumption", { exact: true })
    .selectOption("conversation");
  await page.getByLabel("Prefix reuse assumption", { exact: true }).hover();
  await expect(help).toContainText("KV is assumed already ready in HBM");
  await page.keyboard.press("Escape");
  await page
    .getByRole("button", { name: "Customize architecture", exact: true })
    .click();
  await check(sidebar.locator(".model-geometry input"));
  await page
    .getByRole("button", { name: "Reset scenario", exact: true })
    .click();
  await page.getByRole("button", { name: "Prefill pool", exact: true }).click();
  await clock.hover();
  await expect(help).toContainText("Prefill · Compute clock scale");
  if (!process.env.SERVESCOPE_SKIP_SCREENSHOTS)
    await page.screenshot({
      path:
        "docs/screenshots/" +
        (page.url().includes("/LLM-ServeScope/") ? "pages-" : "") +
        "input-help.png",
    });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Inputs", exact: true }).click();
  const input = page.getByLabel("Input length", { exact: true });
  await input.hover();
  await expect(help).toContainText("The full prompt length");
  const box = (await help.boundingBox())!,
    field = (await input.boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(390);
  expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.y + box.height).toBeLessThanOrEqual(844);
  expect(box.y + box.height <= field.y || box.y >= field.y + field.height).toBe(
    true,
  );
  await input.fill("1024");
  await expect(input).toHaveValue("1024");
  await page.keyboard.press("Escape");
  if (!page.url().includes("/LLM-ServeScope/")) {
    await page.setViewportSize({ width: 1512, height: 1050 });
    await page
      .getByRole("button", { name: "LLMServingSim 2.0", exact: true })
      .click();
    await page.getByLabel("Native memory capacity", { exact: true }).hover();
    await expect(help).toContainText("binary GiB");
  }
  expect(errors).toEqual([]);
}
