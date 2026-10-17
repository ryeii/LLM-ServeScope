export const STATIC_PAGES = import.meta.env.VITE_STATIC_PAGES === "1";

type Pending = {
  resolve: (data: any) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};
let worker: Worker | undefined;
let sequence = 0;
const pending = new Map<number, Pending>();
function stopWorker(message: string) {
  worker?.terminate();
  worker = undefined;
  for (const item of pending.values()) {
    clearTimeout(item.timer);
    item.reject(new Error(message));
  }
  pending.clear();
}
function browserApi(path: string, body?: any): Promise<any> {
  if (!worker) {
    worker = new Worker(new URL("./simulation.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data }) => {
      const item = pending.get(data.id);
      if (!item) return;
      clearTimeout(item.timer);
      pending.delete(data.id);
      if (data.error) item.reject(new Error(data.error));
      else item.resolve(data.result);
    };
    worker.onerror = () =>
      stopWorker(
        "The browser simulation worker failed. Reload the page and use a current browser with WebAssembly support.",
      );
  }
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(
      () =>
        stopWorker(
          "Browser simulation exceeded the 120-second limit. Reduce the scenario size and run again.",
        ),
      120000,
    );
    pending.set(id, { resolve, reject, timer });
    worker!.postMessage({ id, path, body });
  });
}
export async function api(path: string, body?: any) {
  if (STATIC_PAGES) return browserApi(path, body);
  const response = await fetch(
    `${import.meta.env.BASE_URL}api/${path}`,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}
