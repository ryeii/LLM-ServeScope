import { test } from "@playwright/test";
import { gdsScenario } from "../gds-scenario";
test("Pages uses the same GDS and asynchronous cache model as Python", async ({
  page,
}) => gdsScenario(page));
