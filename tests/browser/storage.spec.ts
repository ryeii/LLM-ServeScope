import { test } from "@playwright/test";
import { storageScenario } from "../storage-scenario";
test("KV tiers alter executed compute, expose configured timing and replay published request lengths", async ({
  page,
}) => {
  await storageScenario(page);
});
