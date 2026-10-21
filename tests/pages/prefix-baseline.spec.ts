import { test } from "@playwright/test";
import { prefixBaselineScenario } from "../storage-scenario";
test("Pages layout comparison preserves default batching and native baseline timing", async ({
  page,
}) => prefixBaselineScenario(page));
