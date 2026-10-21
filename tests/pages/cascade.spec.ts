import { test } from "@playwright/test";
import { cascadeScenario } from "../cascade-scenario";
test("Pages supports the complete cascading derivation on desktop and touch", async ({
  page,
}) => {
  await cascadeScenario(page);
});
