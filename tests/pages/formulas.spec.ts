import { test } from "@playwright/test";
import { formulaScenario } from "../formula-scenario";
test("Pages shows actual equations, coefficients and completed-run values", async ({
  page,
}) => {
  await formulaScenario(page);
});
