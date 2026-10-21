import { test } from "@playwright/test";
import { storageScenario } from "../storage-scenario";
test("Pages runs the same tier queues and fitted functions with source-preserving trace replay", async ({
  page,
}) => {
  await storageScenario(page);
});
