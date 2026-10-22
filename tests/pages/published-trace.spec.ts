import { test } from "@playwright/test";
import { publishedTraceScenario } from "../published-trace-scenario";
test("Pages reproduces the published request window without device-timing data", async ({
  page,
}) => publishedTraceScenario(page));
