import { test } from "@playwright/test";
import { storageLayoutsScenario } from "../storage-scenario";
test("single-tier and directed spill layouts change placement, transfers and cache reuse", async ({
  page,
}) => storageLayoutsScenario(page));
