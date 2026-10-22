import { test } from "@playwright/test";
import { publishedTraceScenario } from "../published-trace-scenario";
test("published requests preserve original workload fields with configured-only device timing", async ({
  page,
}) => publishedTraceScenario(page));
