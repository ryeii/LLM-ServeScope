import { test } from "@playwright/test";
import { prefixBaselineScenario } from "../storage-scenario";
test("layout selection preserves default batching and the no-retention baseline exactly", async ({
  page,
}) => prefixBaselineScenario(page));
