import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
process.chdir(root);
const base = process.env.SERVESCOPE_BASE_PATH || "/LLM-ServeScope/";
if (!base.startsWith("/") || !base.endsWith("/"))
  throw new Error("SERVESCOPE_BASE_PATH must begin and end with /.");
execFileSync(process.execPath, ["node_modules/typescript/bin/tsc", "-b"], {
  stdio: "inherit",
});
execFileSync(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "build",
    "--outDir",
    "dist-pages",
    "--base",
    base,
  ],
  {
    stdio: "inherit",
    env: { ...process.env, VITE_STATIC_PAGES: "1" },
  },
);
const output = resolve("dist-pages");
const copy = (from, to) => {
  mkdirSync(dirname(to), { recursive: true });
  copyFileSync(from, to);
};
const runtime = [
  "pyodide.mjs",
  "pyodide.asm.mjs",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
  "pyodide-lock.json",
];
for (const name of runtime)
  copy(`node_modules/pyodide/${name}`, `${output}/python/${name}`);
for (const name of ["__init__.py", "model.py", "catalog.py"])
  copy(`backend/${name}`, `${output}/python/backend/${name}`);
const wheel = "simpy-4.1.1-py3-none-any.whl";
const wheelHash = createHash("sha256")
  .update(readFileSync(`third_party/${wheel}`))
  .digest("hex");
if (
  wheelHash !==
  "7c5ae380240fd2238671160e4830956f8055830a8317edf5c05e495b3823cd88"
)
  throw new Error("SimPy wheel SHA-256 mismatch.");
copy(`third_party/${wheel}`, `${output}/python/${wheel}`);
for (const name of [
  "LLMServingSim-MIT.txt",
  "SimPy-MIT.txt",
  "Pyodide-MPL-2.0.txt",
  "CPython-3.14-PSF.txt",
  "React-MIT.txt",
  "ReactDOM-MIT.txt",
  "Lucide-ISC.txt",
])
  copy(`licenses/${name}`, `${output}/licenses/${name}`);
copy("THIRD_PARTY_NOTICES.md", `${output}/THIRD_PARTY_NOTICES.md`);
// Serve README imagery from Pages, avoiding dependence on GitHub raw-image delivery.
copy("docs/screenshots/github-pages.png", `${output}/docs/screenshots/github-pages.png`);
writeFileSync(
  `${output}/README.md`,
  `# LLM-ServeScope production deployment

This branch contains the prebuilt GitHub Pages edition. Source code, purpose,
limitations, setup and full documentation are on the [main branch](https://github.com/ryeii/LLM-ServeScope/tree/main).

LLM-ServeScope is a behavioral simulation of LLM serving backed by SimPy. It
estimates throughput, latency, memory, energy and utilization, with per-GPU
performance and tokens per Joule. It is **behavioral, not mechanistic simulation**.

The same Python model runs in a Pyodide browser worker. The compiled native
LLMServingSim/ASTRA-Sim reference engine requires the local installation.

LLM-ServeScope design and implementation: [ryeii (Ryan Z An)](https://github.com/ryeii),
including the behavioral modeling extension, interface, visualizations, browser
execution layer and native backend integration.

Reference backend: LLMServingSim 2.0 by CASYS Lab at KAIST — Jaehong Cho, Hyunmin Choi,
Guseul Heo and Jongse Park. [Upstream source](https://github.com/casys-kaist/LLMServingSim).
See THIRD_PARTY_NOTICES.md and licenses/ for retained credits and license texts.

Deploy this branch from production / (root) in GitHub Pages settings.
`,
);
writeFileSync(`${output}/.nojekyll`, "");
writeFileSync(
  `${output}/build-info.json`,
  JSON.stringify(
    {
      project: "LLM-ServeScope",
      source_commit: execFileSync("git", ["rev-parse", "HEAD"], {
        encoding: "utf8",
      }).trim(),
      base_path: base,
      engine: "Same backend/model.py + SimPy in a Pyodide web worker",
      pyodide: "314.0.7",
      simpy: "4.1.1",
      simpy_wheel_sha256: wheelHash,
    },
    null,
    2,
  ) + "\n",
);
console.log(`GitHub Pages files ready in dist-pages (base ${base}).`);
