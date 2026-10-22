import { test } from "@playwright/test";
import { prefixReuseScenario } from "../prefix-reuse-scenario";
test("HBM-only controls and trace-informed prefix potential preserve GPU context", async ({
  page,
}) => prefixReuseScenario(page));
