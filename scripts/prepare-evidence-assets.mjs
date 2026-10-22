import { mkdirSync, copyFileSync, readFileSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
const manifest = JSON.parse(
  readFileSync("third_party/mooncake-traces/manifest.json", "utf8"),
);
const summary = JSON.parse(readFileSync("backend/prefix-reuse.json", "utf8"));
for (const file of manifest.files) {
  const hash = createHash("sha256")
    .update(readFileSync("third_party/mooncake-traces/" + file.file))
    .digest("hex");
  const id = file.file.replace("_trace.jsonl", "");
  if (hash !== file.sha256 || summary.workloads[id].source_sha256 !== hash)
    throw Error("Published prefix evidence integrity failure: " + file.file);
}
// These fixed generated destinations belonged to the retired tiering UI.
// Original published traces and research evidence remain in third_party/.
for (const path of ["public/storage-study", "public/trace-reuse", "public/workloads/mooncake"])
  rmSync(path, { recursive: true, force: true });
mkdirSync("public/prefix-reuse", { recursive: true });
for (const name of ["reuse-potential.png", "reuse-potential.svg"])
  copyFileSync("docs/prefix-reuse/" + name, "public/prefix-reuse/" + name);
