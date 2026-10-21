import { test } from "@playwright/test";
import { storageLayoutsScenario } from "../storage-scenario";
test("Pages supports cache layouts and matches native Python cascade spill timing", async ({
  page,
}) => storageLayoutsScenario(page));
