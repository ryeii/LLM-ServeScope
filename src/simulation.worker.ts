import type { PyodideAPI } from "pyodide";

const assets = new URL(
  `${import.meta.env.BASE_URL}python/`,
  self.location.origin,
);
let runtime: Promise<PyodideAPI> | undefined;
async function source(name: string) {
  const response = await fetch(new URL(name, assets));
  if (!response.ok)
    throw new Error(`Could not load the bundled Python asset: ${name}`);
  return response;
}
function loadRuntime(): Promise<PyodideAPI> {
  if (runtime) return runtime;
  runtime = (async () => {
    // All runtime/model/SimPy assets are served from this same Pages deployment.
    const { loadPyodide } = await import(
      /* @vite-ignore */ new URL("pyodide.mjs", assets).href
    );
    const py: PyodideAPI = await loadPyodide({ indexURL: assets.href });
    py.FS.mkdirTree("/servescope/backend");
    py.FS.mkdirTree("/servescope/site-packages");
    const modules = ["__init__.py", "catalog.py", "model.py"];
    const files = await Promise.all(
      modules.map(async (name) => [
        name,
        await (await source(`backend/${name}`)).text(),
      ]),
    );
    for (const [name, contents] of files)
      py.FS.writeFile(`/servescope/backend/${name}`, contents);
    py.unpackArchive(
      await (await source("simpy-4.1.1-py3-none-any.whl")).arrayBuffer(),
      "zip",
      { extractDir: "/servescope/site-packages" },
    );
    await py.runPythonAsync(`
import sys, json
sys.path[:0] = ['/servescope', '/servescope/site-packages']
from backend.model import catalog, validate, simulate
`);
    return py;
  })();
  return runtime;
}
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      if (data.path === "health") {
        self.postMessage({
          id: data.id,
          result: {
            status: "ready",
            behavioral: true,
            native: {
              available: false,
              local_only: true,
              message:
                "Native LLMServingSim requires the local Python/ASTRA installation. Browser behavioral simulation is available.",
            },
          },
        });
        return;
      }
      if (!["catalog", "validate", "simulate"].includes(data.path))
        throw new Error(
          "This operation requires the local LLMServingSim backend.",
        );
      const py = await loadRuntime();
      py.globals.set(
        "_request_json",
        JSON.stringify({ path: data.path, body: data.body }),
      );
      const result = await py.runPythonAsync(`
_request = json.loads(_request_json)
if _request['path'] == 'catalog':
    _response = catalog()
elif _request['path'] == 'validate':
    _response = {'config': validate(_request['body'])}
else:
    _response = simulate(_request['body'])
    _response['provenance']['execution_runtime'] = 'Browser: Pyodide 314.0.7 / SimPy 4.1.1'
json.dumps(_response, allow_nan=False)
`);
      self.postMessage({ id: data.id, result: JSON.parse(result) });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // Keep useful model validation errors without the Python traceback noise.
      const validation = message.match(/ValueError: (.+)/);
      self.postMessage({
        id: data.id,
        error: validation ? validation[1] : message,
      });
    }
  });
};
