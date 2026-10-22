import { test } from "@playwright/test";
import { inputHelpScenario } from "../input-help-scenario";
test("every scenario value has a concise definition without changing editing", async ({
  page,
}) => inputHelpScenario(page));
