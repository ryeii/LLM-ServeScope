import { test } from "@playwright/test";
import { calibrationScenario } from "../calibration-scenario";

test("fitted GPU function matches local Python and the original roofline remains available", async ({
  page,
}) => {
  await calibrationScenario(page);
});
