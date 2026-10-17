import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync, spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
process.chdir(root);
const git = (args, options = {}) =>
  execFileSync("git", args, { encoding: "utf8", ...options }).trim();
if (git(["status", "--porcelain"]))
  throw new Error("Commit or stash source changes before publishing Pages.");
const source = git(["rev-parse", "HEAD"]);
const remote = git(["remote", "get-url", "origin"]);
console.log(`Preparing production deployment from ${source} to ${remote}`);
git(["fetch", "--no-tags", "origin", "main"]);
if (git(["rev-parse", "FETCH_HEAD"]) !== source)
  throw new Error(
    "Push this source commit to origin/main before publishing Pages.",
  );
execFileSync(process.execPath, ["scripts/build-pages.mjs"], {
  stdio: "inherit",
});
const remoteProduction = git([
  "ls-remote",
  "--heads",
  "origin",
  "refs/heads/production",
]);
let parent;
if (remoteProduction) {
  git(["fetch", "--no-tags", "origin", "production"]);
  parent = git(["rev-parse", "FETCH_HEAD"]);
}
const localResult = spawnSync(
  "git",
  ["rev-parse", "--verify", "refs/heads/production"],
  { encoding: "utf8" },
);
const previous =
  localResult.status === 0 ? localResult.stdout.trim() : undefined;
if (previous && previous !== parent) {
  const ancestor =
    parent &&
    spawnSync("git", ["merge-base", "--is-ancestor", previous, parent]);
  if (!ancestor || ancestor.status !== 0)
    throw new Error(
      "Local production has unpublished history. Resolve it before publishing; no force push is performed.",
    );
}
const temporary = mkdtempSync(join(tmpdir(), "servescope-pages-"));
const output = resolve("dist-pages");
const isolated = {
  cwd: output,
  env: { ...process.env, GIT_INDEX_FILE: join(temporary, "index") },
};
const args = ["--git-dir", resolve(".git"), "--work-tree", output];
try {
  git([...args, "read-tree", "--empty"], isolated);
  git([...args, "add", "--force", "--all", "--", "."], isolated);
  const tree = git([...args, "write-tree"], isolated);
  const commit = git([
    "commit-tree",
    tree,
    ...(parent ? ["-p", parent] : []),
    "-m",
    `Deploy LLM-ServeScope from ${source}`,
  ]);
  // Push the prepared commit first: a failed network/auth attempt leaves no
  // unpublished local production branch that would block a safe retry.
  execFileSync("git", ["push", "origin", `${commit}:refs/heads/production`], {
    stdio: "inherit",
  });
  git([
    "update-ref",
    "refs/heads/production",
    commit,
    previous || "0".repeat(40),
  ]);
  console.log(
    `Published production ${commit}. Select production / (root) in GitHub Pages settings.`,
  );
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
