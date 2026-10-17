"""Loopback-only local API and static frontend. Run: .venv/bin/python -m backend.server."""

from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, mimetypes, os, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .model import ROOT, catalog, simulate, validate

JOBS = {}
LOCK = threading.Lock()
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)
NATIVE_PY = ROOT / ".venv-sim/bin/python"
RUNNER = ROOT / "integration/native_runner.py"


def native_status():
    if not NATIVE_PY.exists() or not RUNNER.exists():
        return dict(
            available=False,
            message="Native backend is being prepared. Behavioral modeling is ready.",
        )
    try:
        p = subprocess.run(
            [str(NATIVE_PY), str(RUNNER), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, ValueError, OSError) as e:
        return dict(
            available=False,
            message="Native readiness check failed: " + type(e).__name__,
        )


def native_job(identifier, config):
    path = ROOT / "runs" / identifier
    path.mkdir(parents=True)
    command = [
        str(NATIVE_PY),
        str(RUNNER),
        "--input-tokens",
        str(config["input_tokens"]),
        "--output-tokens",
        str(config["output_tokens"]),
        "--requests",
        str(config["requests"]),
        "--tp",
        str(config["tp"]),
        "--hardware",
        config["hardware"],
        "--memory-gib",
        str(config["memory_gib"]),
        "--memory-bandwidth-gbps",
        str(config["memory_bandwidth_gbps"]),
        "--link-bandwidth-gibps",
        str(config["link_bandwidth_gibps"]),
        "--link-latency-ns",
        str(config["link_latency_ns"]),
        "--output",
        str(path),
    ]
    with LOCK:
        JOBS[identifier].update(status="running", started_at=time.time())
    try:
        with (path / "runner.log").open("w") as log:
            p = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                raise RuntimeError(
                    "Native run exceeded the 180-second interactive limit. Logs retained."
                )
        result_path = path / "result.json"
        if not result_path.exists():
            raise RuntimeError(
                f"Native runner exited {p.returncode}; no result.json. See retained runner.log."
            )
        result = json.loads(result_path.read_text())
        if p.returncode or result.get("status") in ("failed", "error"):
            raise RuntimeError(
                result.get("error", f"Native process exited {p.returncode}")
            )
        with LOCK:
            JOBS[identifier].update(
                status="complete", result=result, finished_at=time.time()
            )
    except Exception as e:
        with LOCK:
            JOBS[identifier].update(
                status="failed", error=str(e), finished_at=time.time()
            )


class Handler(BaseHTTPRequestHandler):
    server_version = "MemAnalysis/0.1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/catalog":
            return self.send_json(catalog())
        if u.path == "/api/health":
            return self.send_json(
                dict(status="ready", behavioral=True, native=native_status())
            )
        if u.path.startswith("/api/jobs/"):
            identifier = u.path.rsplit("/", 1)[-1]
            with LOCK:
                value = JOBS.get(identifier)
            return self.send_json(
                value or {"error": "Job not found"}, 200 if value else 404
            )
        if u.path == "/api/native/artifact":
            q = parse_qs(u.query)
            identifier = q.get("job", [""])[0]
            name = q.get("file", [""])[0]
            if identifier not in JOBS:
                return self.send_json({"error": "Unknown job"}, 404)
            base = (ROOT / "runs" / identifier).resolve()
            path = (base / name).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                return self.send_json({"error": "Artifact not found"}, 404)
            return self.send_file(path)
        if u.path.startswith("/api/"):
            return self.send_json({"error": "Not found"}, 404)
        base = (ROOT / "dist").resolve()
        path = (base / u.path.lstrip("/")).resolve()
        if not path.is_relative_to(base):
            return self.send_json({"error": "Not found"}, 404)
        if not path.is_file():
            path = base / "index.html"
        if not path.exists():
            return self.send_json(
                {"error": "Run npm run build or use the Vite development server."}, 404
            )
        return self.send_file(path)

    def send_file(self, path):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 128):
                self.wfile.write(chunk)

    def do_POST(self):
        # Local tools should not accept drive-by cross-origin submissions.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc not in (
            "127.0.0.1:8788",
            "localhost:8788",
            "127.0.0.1:4180",
            "localhost:4180",
        ):
            return self.send_json({"error": "Origin not allowed"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 256000:
                raise ValueError("Request must be nonempty JSON smaller than 256 KB.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/validate":
                return self.send_json({"config": validate(body)})
            if self.path == "/api/simulate":
                return self.send_json(simulate(body))
            if self.path == "/api/native/run":
                config = {}
                for name, lo, hi in [
                    ("input_tokens", 1, 2048),
                    ("output_tokens", 2, 32),
                    ("requests", 1, 8),
                    ("tp", 1, 2),
                ]:
                    value = body.get(name)
                    if type(value) is not int or not lo <= value <= hi:
                        raise ValueError(f"{name} must be an integer in [{lo},{hi}].")
                    config[name] = value
                hardware = body.get("hardware", "RTXPRO6000")
                if hardware not in ("RTXPRO6000", "RTX4090"):
                    raise ValueError("Unsupported native hardware profile.")
                if hardware == "RTX4090" and config["tp"] != 1:
                    raise ValueError("RTX4090 has only a TP1 profile in this demo.")
                config["hardware"] = hardware
                from .model import number

                for field, default, lo, hi in [
                    ("memory_gib", 96 if hardware == "RTXPRO6000" else 24, 1, 2048),
                    (
                        "memory_bandwidth_gbps",
                        1597 if hardware == "RTXPRO6000" else 1008,
                        1,
                        100000,
                    ),
                    ("link_bandwidth_gibps", 16, 0.01, 10000),
                    ("link_latency_ns", 1000, 0, 1000000),
                ]:
                    config[field] = number(body.get(field, default), field, lo, hi)
                if not native_status().get("available"):
                    return self.send_json(
                        {"error": "Native backend is not ready."}, 503
                    )
                with LOCK:
                    if any(j["status"] in ("queued", "running") for j in JOBS.values()):
                        return self.send_json(
                            {
                                "error": "A native run is already active. Wait for it to finish."
                            },
                            409,
                        )
                    if len(JOBS) >= 20:
                        raise ValueError(
                            "Restart the API to clear the 20-run session limit. Artifacts stay on disk."
                        )
                    identifier = uuid.uuid4().hex
                    JOBS[identifier] = dict(
                        id=identifier,
                        status="queued",
                        config=config,
                        created_at=time.time(),
                    )
                POOL.submit(native_job, identifier, config)
                return self.send_json({"id": identifier, "status": "queued"}, 202)
            return self.send_json({"error": "Not found"}, 404)
        except (ValueError, KeyError, TypeError) as e:
            return self.send_json({"error": str(e)}, 400)
        except Exception as e:
            return self.send_json({"error": "Simulation failed: " + str(e)}, 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8788)
    a = parser.parse_args()
    print(f"LLM MemAnalysis local API + app: http://127.0.0.1:{a.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
