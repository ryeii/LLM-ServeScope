import { test } from "@playwright/test";
import { inputHelpScenario } from "../input-help-scenario";
test("Pages gives every input a concise definition with accessible hover and focus", async ({
  page,
}) => inputHelpScenario(page));
