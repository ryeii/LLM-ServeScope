import { test } from "@playwright/test";
import { storageScenario } from "../storage-scenario";
test("KV tiers alter executed compute, expose fitted coefficients and replay unchanged request lengths", async ({
  page,
}) => {
  await storageScenario(page);
});
