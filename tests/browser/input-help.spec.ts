import { test } from "@playwright/test";
import { inputHelpScenario } from "../input-help-scenario";
test("every scenario value explains its live meaning, effect and evidence without changing editing", async ({
  page,
}) => inputHelpScenario(page));
