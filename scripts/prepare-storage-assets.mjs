import { mkdirSync, copyFileSync, readFileSync } from "node:fs";
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
mkdirSync("public/storage-study", { recursive: true });
for (const name of [
  "io-fit.png",
  "io-fit.svg",
  "hit-fit.png",
  "hit-fit.svg",
  "io-candidates.png",
  "io-candidates.svg",
  "hit-candidates.png",
  "hit-candidates.svg",
])
  copyFileSync(
    "docs/storage-2026-10-20/" + name,
    "public/storage-study/" + name,
  );
