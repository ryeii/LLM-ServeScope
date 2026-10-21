import { test } from "@playwright/test";
import { pinnedScenario } from "../pinned-scenario";
test("pinned snapshots, input differences and hold-P preview remain coherent across every result view", async ({
  page,
}) => {
  await pinnedScenario(page);
});
