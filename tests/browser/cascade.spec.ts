import { test } from "@playwright/test";
import { cascadeScenario } from "../cascade-scenario";
test("result dependencies lead through the executed schedule to fitted work and input sources", async ({
  page,
}) => {
  await cascadeScenario(page);
});
