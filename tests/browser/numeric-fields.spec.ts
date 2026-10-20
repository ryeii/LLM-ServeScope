import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("numeric fields allow replacement, preserve decimals and zero, and follow pool changes", async ({
  page,
}) => {
  await page.goto("/");
  const inputLength = page.getByLabel("Input length", { exact: true });
  await expect(inputLength).toHaveValue("2048");
  await inputLength.fill("");
  await expect(inputLength).toHaveValue("");
  await inputLength.pressSequentially("1");
  await expect(inputLength).toHaveValue("1");
  await inputLength.press("Tab");
  await expect(inputLength).toHaveValue("1");

  await page.getByRole("button", { name: "Decode pool", exact: true }).click();
  await page.getByText("Compute throughput", { exact: true }).click();
  const clock = page.getByLabel("Compute clock scale", { exact: true });
  await clock.fill("");
  await expect(clock).toHaveValue("");
  await clock.pressSequentially("1");
  await clock.press("Tab");
  await expect(clock).toHaveValue("1");

  const efficiency = page.getByLabel("Compute efficiency", { exact: true });
  await efficiency.fill("");
  await expect(efficiency).toHaveValue("");
  await efficiency.pressSequentially("0.5");
  await efficiency.press("Tab");
  await expect(efficiency).toHaveValue("0.5");
  // Abandoning an empty edit retains the last actual number.
  await efficiency.fill("");
  await expect(efficiency).toHaveValue("");
  await efficiency.press("Tab");
  await expect(efficiency).toHaveValue("0.5");

  const reservation = page.getByLabel("Runtime reservation", { exact: true });
  await reservation.fill("");
  await expect(reservation).toHaveValue("");
  await reservation.pressSequentially("0");
  await reservation.press("Tab");
  await expect(reservation).toHaveValue("0");

  await page.getByRole("button", { name: "Prefill pool", exact: true }).click();
  await expect(efficiency).toHaveValue("0.65");
  await page.getByRole("button", { name: "Decode pool", exact: true }).click();
  await expect(efficiency).toHaveValue("0.5");

  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Save scenario", exact: true })
    .click();
  const config = JSON.parse(
    await readFile((await (await download).path())!, "utf8"),
  );
  expect(config.input_tokens).toBe(1);
  expect(config.decode.hardware.clock_scale).toBe(1);
  expect(config.decode.hardware.compute_efficiency).toBe(0.5);
  expect(config.decode.runtime_reserved_GB).toBe(0);

  await page
    .getByRole("button", { name: "Reset scenario", exact: true })
    .click();
  await expect(inputLength).toHaveValue("2048");
  await expect(efficiency).toHaveValue("0.65");
  await page.getByText("Memory + tile shape", { exact: true }).click();
  await page
    .getByLabel("Hardware preset", { exact: true })
    .selectOption("h200");
  await expect(
    page.getByLabel("Memory bandwidth", { exact: true }),
  ).toHaveValue("4.8");
});
