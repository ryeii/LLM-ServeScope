import { test } from "@playwright/test";
import { prefixReuseScenario } from "../prefix-reuse-scenario";
test("Pages matches Python trace-derived HBM reuse and rejects retired tier settings", async ({
  page,
}) => prefixReuseScenario(page));
