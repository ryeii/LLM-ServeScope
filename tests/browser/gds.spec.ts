import { test } from "@playwright/test";
import { gdsScenario } from "../gds-scenario";
test("Dynamo-style defaults, GDS paths, provenance and Python parity", async ({
  page,
}) => gdsScenario(page));
