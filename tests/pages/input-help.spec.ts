import { test } from "@playwright/test";
import { inputHelpScenario } from "../input-help-scenario";
test("Pages explains every scenario input with pool context, provenance and accessible hover/focus", async ({
  page,
}) => inputHelpScenario(page));
