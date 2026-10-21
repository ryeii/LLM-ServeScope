import { test } from "@playwright/test";
import { activeKVScenario } from "../active-kv-scenario";
test("independent active and retained KV, shared queues, formula cascades and Python parity", async ({
  page,
}) => {
  await activeKVScenario(page);
});
