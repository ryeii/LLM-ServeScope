import { mkdirSync, copyFileSync, readFileSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
const root = "third_party/mooncake-traces";
const manifest = JSON.parse(readFileSync(root + "/manifest.json", "utf8"));
mkdirSync("public/workloads/mooncake", { recursive: true });
for (const file of manifest.files) {
  const raw = readFileSync(root + "/" + file.file);
  if (createHash("sha256").update(raw).digest("hex") !== file.sha256)
    throw Error("Mooncake trace integrity failure: " + file.file);
  copyFileSync(
    root + "/" + file.file,
    "public/workloads/mooncake/" + file.file,
  );
}
for (const name of ["manifest.json", "UPSTREAM-README.md", "LICENSE-APACHE"])
  copyFileSync(root + "/" + name, "public/workloads/mooncake/" + name);
// Do not republish retired device-benchmark artifacts left by older builds.
rmSync("public/storage-study", { recursive: true, force: true });
mkdirSync("public/trace-reuse", { recursive: true });
for (const name of ["hit-fit.png", "hit-fit.svg"])
  copyFileSync(
    "docs/trace-reuse/" + name,
    "public/trace-reuse/" + name,
  );
