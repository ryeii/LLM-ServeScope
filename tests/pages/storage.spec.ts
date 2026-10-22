import { test } from "@playwright/test";
import { storageScenario } from "../storage-scenario";
test("Pages runs configured tier queues with source-preserving published request replay", async ({
  page,
}) => {
  await storageScenario(page);
});
