import { expect, type Page } from "@playwright/test";

export async function cascadeScenario(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  const throughput = page.locator(
    '[data-formula-key="metrics.throughput_tps"]',
  );
  await expect(throughput).toBeVisible({ timeout: 60000 });
  // This regression follows the published no-retention worked example.
  await page
    .getByLabel("Import scenario", { exact: true })
    .setInputFiles("examples/default-scenario.json");
  await expect(page.getByRole("alert")).toContainText("Imported");
  await page.getByRole("button", { name: "Run scenario", exact: true }).click();
  await expect(
    page.getByText("Results match scenario", { exact: true }),
  ).toBeVisible();
  await throughput.hover();
  const card = (depth: number) =>
    page.locator(`[data-cascade-depth="${depth}"]`);
  const follow = async (
    depth: number,
    label: string | RegExp,
    hover = false,
  ) => {
    const term = card(depth).getByRole("button", { name: label, exact: true });
    if (hover) await term.hover();
    else await term.focus();
    await expect(card(depth + 1)).toBeVisible();
    const box = (await card(depth + 1).boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(page.viewportSize()!.width);
    expect(box.y + box.height).toBeLessThanOrEqual(page.viewportSize()!.height);
    return card(depth + 1);
  };
  await follow(0, "Explain Simulation horizon", true);
  await expect(card(1)).toContainText("Concurrent batches overlap");
  await follow(1, /^Explain Last completion/);
  await expect(card(2).locator("pre")).toContainText(
    "formation_wait + prefill_span + handoff_span + decode_span",
  );
  await follow(2, "Explain Decode service and waits");
  await expect(card(3)).toContainText("Admission may delay");
  await follow(3, "Explain Final interval in this phase");
  await follow(4, "Explain Interval service cost");
  await expect(card(5).locator("pre")).toContainText(
    "layers_per_stage × Σ operator_ms",
  );
  await follow(5, "Explain One transformer layer");
  await expect(card(6)).toContainText("Actual executed shape");
  await follow(6, "Explain QKV projection");
  await expect(card(7)).toContainText("Fitted intercept");
  await follow(7, "Explain Nominal compute feature");
  await follow(8, "Explain Arithmetic work for fitted function");
  await expect(card(9).locator("pre")).toContainText("F_fit");
  await follow(9, "Explain Hidden width");
  await expect(card(10)).toContainText("Model architecture");
  await expect(card(10)).toContainText("Source reached");
  await page.screenshot({ path: "/tmp/servescope-cascade-desktop.png" });
  await page.keyboard.press("Escape");
  await expect(card(10)).toHaveCount(0);
  await expect(card(9)).toBeVisible();
  await card(9).getByRole("button", { name: "Result", exact: true }).click();
  await expect(page.getByRole("tooltip")).toHaveCount(1);
  await follow(0, "Explain Completed requests");
  await expect(card(1)).toContainText("Scenario input");
  await card(1)
    .getByRole("button", { name: "Close explanations", exact: true })
    .click();
  await expect(page.getByRole("tooltip")).toHaveCount(0);

  // Touch/small screens show one level at a time, with explicit Back and breadcrumbs.
  await page.setViewportSize({ width: 390, height: 844 });
  await throughput.click();
  await card(0)
    .getByRole("button", { name: "Explain Simulation horizon", exact: true })
    .click();
  await expect(card(0)).toBeHidden();
  await expect(card(1)).toBeVisible();
  await card(1)
    .getByRole("button", { name: /^Explain Last completion/ })
    .click();
  await expect(card(2)).toBeVisible();
  await page.screenshot({ path: "/tmp/servescope-cascade-mobile.png" });
  await card(2).getByRole("button", { name: "← Back", exact: true }).click();
  await expect(card(1)).toBeVisible();
  const box = (await card(1).boundingBox())!;
  expect(box.x + box.width).toBeLessThanOrEqual(390);
  expect(box.y + box.height).toBeLessThanOrEqual(844);
  await page.mouse.click(2, 2);
  await expect(page.getByRole("tooltip")).toHaveCount(0);
  expect(errors).toEqual([]);
}
