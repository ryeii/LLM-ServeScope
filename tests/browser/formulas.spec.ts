import { test } from "@playwright/test";
import { formulaScenario } from "../formula-scenario";
test("result formulas use completed-run inputs and real fitted coefficients on desktop and mobile", async ({
  page,
}) => {
  await formulaScenario(page);
});
