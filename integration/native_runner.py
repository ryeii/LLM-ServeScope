"""Bounded adapter for pristine KAIST LLMServingSim + ASTRA-Sim.

Run with .venv-sim/bin/python. All configuration/workload/results are separate
from upstream source. Workload token IDs below are synthetic identity markers,
not model-generated text. No GPU inference occurs in a native serving run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.vendor' / 'LLMServingSim'
PYTHON = ROOT / '.venv-sim' / 'bin' / 'python'
BINARY = ROOT / '.native' / 'astra-build' / 'astra' / 'bin' / 'AstraSim_Analytical_Congestion_Unaware'
PINS = {
    'LLMServingSim': 'b09c7aaab4b072d73136641841cc2688616c49eb',
    'ASTRA-Sim': 'd3d068f4050a2fb9a7538cc671a982152c2ba9b7',
    'Chakra': '5e6e9b28438bc1899105dbdd68eed4cf61bee6fd',
}
PRESETS = [
    {'hardware': 'RTXPRO6000', 'model': 'meta-llama/Llama-3.1-8B', 'tensor_parallel': [1, 2],
     'memory_gib': 96, 'memory_bandwidth_gbps': 1597,
     'provenance': 'Upstream measured kernel-latency tables; cluster links are explicit scenario parameters.'},
    {'hardware': 'RTX4090', 'model': 'meta-llama/Llama-3.1-8B', 'tensor_parallel': [1],
     'memory_gib': 24, 'memory_bandwidth_gbps': 1008,
     'provenance': 'Upstream measured kernel-latency tables; memory capacity/bandwidth from vendor specification.'},
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def status():
    return {'available': PYTHON.exists() and BINARY.exists(),
            'python': str(PYTHON.relative_to(ROOT)), 'binary': str(BINARY.relative_to(ROOT)),
            'pins': PINS, 'presets': PRESETS,
            'models': ['meta-llama/Llama-3.1-8B'],
            'hardware': ['RTXPRO6000', 'RTX4090'],
            'allowed_fields': ['input_tokens', 'output_tokens', 'requests', 'tp', 'hardware',
                               'memory_gib', 'memory_bandwidth_gbps', 'link_bandwidth_gibps',
                               'link_latency_ns', 'cpu_memory_bandwidth_gbps', 'arrival_interval_ms'],
            'limits': {'max_requests': 16, 'max_input_tokens': 4096,
                       'max_output_tokens': 64, 'timeout_s': 120, 'max_rss_gib': 2.0}}


def provenance(hardware):
    roots = {'LLMServingSim': UPSTREAM, 'ASTRA-Sim': UPSTREAM / 'astra-sim',
             'Chakra': UPSTREAM / 'astra-sim/extern/graph_frontend/chakra'}
    sources = {}
    for name, path in roots.items():
        pin = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if pin != PINS[name]:
            raise RuntimeError(f'{name} source pin differs from audited revision')
        diff = subprocess.check_output(['git', '-C', str(path), 'diff', '--no-ext-diff', '--ignore-submodules'], text=True)
        if diff:
            raise RuntimeError(f'{name} has modified tracked source; refusing an unrecorded execution')
        sources[name] = {'commit': pin, 'tracked_source_clean': True}
    profile = UPSTREAM / 'profiler/perf' / hardware / 'meta-llama/Llama-3.1-8B/bf16'
    files = {str(p.relative_to(UPSTREAM)): sha256(p) for p in sorted(profile.rglob('*')) if p.is_file()}
    return {'sources': sources, 'profile_files': files, 'binary_sha256': sha256(BINARY),
            'runner_sha256': sha256(__file__),
            'cmake_wrapper_sha256': sha256(ROOT / 'integration/native-cmake/CMakeLists.txt'),
            'protobuf_cpp_source_commit': 'f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c',
            'protobuf_schema_sha256': sha256(UPSTREAM / 'astra-sim/extern/graph_frontend/chakra/schema/protobuf/et_def.proto'), 'source_url': 'https://github.com/casys-kaist/LLMServingSim',
            'build_adaptation': 'Isolated CMake wrapper adds generated protobuf3.21.12 bindings and enables libc++ legacy binary_function declaration; upstream sources unmodified.'}


def stop_tree(child):
    import psutil
    try:
        members = psutil.Process(child.pid).children(recursive=True)
    except psutil.Error:
        members = []
    for p in reversed(members):
        try:
            p.terminate()
        except psutil.Error:
            pass
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(members, timeout=1)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def bounded_run(command, output, timeout_s=120, max_rss_gib=2.0):
    import psutil
    started = time.monotonic()
    peak = 0
    reason = None
    observations = []
    with (output / 'simulator.log').open('w') as log:
        child = subprocess.Popen(command, cwd=UPSTREAM, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    proc = psutil.Process(child.pid)
                    members = [proc] + proc.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in members if p.is_running())
                except psutil.Error:
                    rss = 0
                available = psutil.virtual_memory().available
                peak = max(peak, rss)
                observations.append({'wall_elapsed_s': round(elapsed, 3), 'rss_gib': rss / 2**30,
                                     'host_available_gib': available / 2**30})
                if elapsed > timeout_s:
                    reason = 'wall-clock timeout'
                elif rss > max_rss_gib * 2**30:
                    reason = 'process-tree memory cap'
                elif available < 0.2 * 2**30:
                    reason = 'host available memory floor'
                if reason:
                    stop_tree(child)
                    break
                time.sleep(0.2)
        except BaseException:
            stop_tree(child)
            raise
    resource = {'returncode': child.returncode, 'reason': reason,
                'wall_elapsed_s': time.monotonic() - started, 'peak_rss_gib': peak / 2**30,
                'timeout_s': timeout_s, 'max_rss_gib': max_rss_gib, 'samples': observations}
    dump(output / 'resources.json', resource)
    if reason or child.returncode != 0:
        raise RuntimeError(reason or f'Native simulator exited with code {child.returncode}; inspect simulator.log')
    return resource


def parse_graph(path):
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node, NodeType
    from chakra.src.third_party.utils.protolib import decodeMessage
    nodes, edges = [], []
    with path.open('rb') as stream:
        metadata = GlobalMetadata()
        decodeMessage(stream, metadata)
        while True:
            node = Node()
            if not decodeMessage(stream, node):
                break
            attrs = {}
            for attr in node.attr:
                kind = attr.WhichOneof('value')
                value = getattr(attr, kind) if kind else None
                if hasattr(value, 'values'):
                    value = list(value.values)
                if isinstance(value, bytes):
                    value = value.hex()
                attrs[attr.name] = value
            nodes.append({'id': str(node.id), 'name': node.name, 'type': NodeType.Name(node.type),
                          'profile_cost_ns': int(node.duration_micros), 'attributes': attrs})
            edges.extend({'source': str(parent), 'target': str(node.id), 'kind': 'data'} for parent in node.data_deps)
            edges.extend({'source': str(parent), 'target': str(node.id), 'kind': 'control'} for parent in node.ctrl_deps)
    return {'source': str(path.relative_to(ROOT)),
            'nodes': nodes, 'edges': edges,
            'timing_note': 'Upstream LLM converter places nanosecond profile costs in the protobuf duration_micros field; these are costs, not scheduled node start/end timestamps.'}


def execute(args):
    if not status()['available']:
        raise RuntimeError('Native backend is not built; run integration/setup_native.sh')
    preset = next(p for p in PRESETS if p['hardware'] == args.hardware)
    if args.tp not in preset['tensor_parallel']:
        raise ValueError('No upstream measured latency profile for requested hardware/TP combination')
    if args.memory_gib is None:
        args.memory_gib = preset['memory_gib']
    if args.memory_bandwidth_gbps is None:
        args.memory_bandwidth_gbps = preset['memory_bandwidth_gbps']
    bounds = [('input_tokens', 1, 4096), ('output_tokens', 2, 64), ('requests', 1, 16)]
    for name, low, high in bounds:
        value = getattr(args, name)
        if not low <= value <= high:
            raise ValueError(f'{name} must be in [{low}, {high}]')
    for name in ['memory_gib', 'memory_bandwidth_gbps', 'link_bandwidth_gibps', 'cpu_memory_bandwidth_gbps']:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not 0 <= args.link_latency_ns <= 1000000 or not 0 <= args.arrival_interval_ms <= 10000:
        raise ValueError('Link latency or request arrival interval outside supported bounds')
    output = Path(args.output).resolve()
    output.relative_to(ROOT / 'runs')
    if (output / 'result.json').exists() or (output / 'requests.csv').exists():
        raise ValueError('Output contains an existing native run; choose a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'status.json', {'status': 'running'})
    try:
        source = provenance(args.hardware)
        run_id = 'memanalysis_' + uuid.uuid4().hex
        # Upstream intentionally launches this fixed, gitignored build path.
        # Its symlink targets our isolated binary; no tracked source is changed.
        expected = UPSTREAM / 'astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra'
        expected.parent.mkdir(parents=True, exist_ok=True)
        if not expected.exists() and not expected.is_symlink():
            expected.symlink_to(os.path.relpath(BINARY, expected.parent))
        if expected.resolve() != BINARY.resolve():
            raise RuntimeError('Upstream expected backend path points to another binary')
        config = {
            'num_nodes': 1, 'link_bw': args.link_bandwidth_gibps,
            'link_latency': args.link_latency_ns,
            'nodes': [{'num_instances': 1,
                       'cpu_mem': {'mem_size': 256, 'mem_bw': args.cpu_memory_bandwidth_gbps, 'mem_latency': 0},
                       'instances': [{'model_name': preset['model'], 'hardware': args.hardware,
                                      'npu_mem': {'mem_size': args.memory_gib,
                                                  'mem_bw': args.memory_bandwidth_gbps,
                                                  'mem_latency': 0, 'mem_util': 0.9},
                                      'num_npus': args.tp, 'tp_size': args.tp, 'pd_type': None}]}]}
        dump(output / 'cluster.json', config)
        # Distinct markers avoid accidentally sharing prefixes across requests.
        workload = [{'input_toks': args.input_tokens, 'output_toks': args.output_tokens,
                     'arrival_time_ns': round(i * args.arrival_interval_ms * 1e6),
                     'input_tok_ids': list(range(10000 + i * 8192, 10000 + i * 8192 + args.input_tokens)),
                     'output_tok_ids': list(range(300000 + i * 128, 300000 + i * 128 + args.output_tokens))}
                    for i in range(args.requests)]
        (output / 'workload.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in workload))
        inputs = output / 'astra-inputs'
        command = [str(PYTHON), '-m', 'serving', '--cluster-config', str(output / 'cluster.json'),
                   '--dataset', os.path.relpath(output / 'workload.jsonl', UPSTREAM), '--output', str(output / 'requests.csv'),
                   '--run-id', run_id, '--inputs-root', str(inputs), '--save-trace-text', '--keep-inputs',
                   '--max-num-seqs', str(args.requests), '--max-num-batched-tokens', '2048',
                   '--no-enable-prefix-caching', '--log-level', 'INFO']
        dump(output / 'command.json', {'argv': command, 'cwd': str(UPSTREAM.relative_to(ROOT))})
        resource = bounded_run(command, output)
        with (output / 'requests.csv').open() as stream:
            raw = list(csv.DictReader(stream))
        requests = []
        for row in raw:
            request = {k: int(v) for k, v in row.items() if k not in ['model', 'ITL']}
            request['model'] = row['model']
            request['ITL'] = ast.literal_eval(row['ITL'])
            requests.append(request)
        if len(requests) != args.requests:
            raise RuntimeError('Incomplete request CSV; refusing a successful result')
        requests.sort(key=lambda r: r['request id'])
        first_arrival = min(r['arrival'] for r in requests)
        last_end = max(r['end_time'] for r in requests)
        total_output = sum(r['output'] for r in requests)
        total_tokens = sum(r['input'] + r['output'] for r in requests)
        clocks_match = re.search(r'Total clocks \(ns\):\s+(\d+)', (output / 'simulator.log').read_text())
        if not clocks_match:
            raise RuntimeError('Missing native total-clock summary')
        native_clocks = int(clocks_match.group(1))
        def percentile(values, p):
            values = sorted(values)
            at = (len(values)-1) * p
            lo, hi = math.floor(at), math.ceil(at)
            return (values[lo] * (hi-at) + values[hi] * (at-lo)) if lo != hi else values[lo]
        metrics = {'request_count': len(requests), 'output_tokens': total_output,
                   'native_total_clocks_ns': native_clocks,
                   'simulated_duration_ms': native_clocks / 1e6,
                   'request_span_ms': (last_end-first_arrival) / 1e6,
                   'output_tokens_per_second': total_output / (native_clocks / 1e9),
                   'request_span_output_tokens_per_second': total_output / ((last_end-first_arrival) / 1e9),
                   'all_tokens_per_second': total_tokens / (native_clocks / 1e9),
                   'request_throughput_per_second': len(requests) / (native_clocks / 1e9),
                   'mean_ttft_ms': statistics.mean(r['TTFT'] for r in requests) / 1e6,
                   'mean_tpot_ms': statistics.mean(r['TPOT'] for r in requests) / 1e6,
                   'mean_latency_ms': statistics.mean(r['latency'] for r in requests) / 1e6,
                   'mean_queue_ms': statistics.mean(r['queuing_delay'] for r in requests) / 1e6,
                   'p50_ttft_ms': percentile([r['TTFT'] for r in requests], .5)/1e6,
                   'p99_ttft_ms': percentile([r['TTFT'] for r in requests], .99)/1e6,
                   'p50_tpot_ms': percentile([r['TPOT'] for r in requests], .5)/1e6,
                   'p99_tpot_ms': percentile([r['TPOT'] for r in requests], .99)/1e6}
        texts = sorted((inputs / 'trace').rglob('*.txt'))
        paths = sorted((p for p in (inputs / 'workload').rglob('llm.0.et') if 'instance' in str(p)),
                       key=lambda p: int(re.search(r'batch(\d+)', str(p)).group(1)) if re.search(r'batch(\d+)', str(p)) else -1)
        graphs = []
        for path in paths[:2]:
            graphs.append(parse_graph(path))
        timeline = []
        for r in requests:
            first_token = r['arrival'] + r['TTFT']
            timeline.extend([
                {'request_id': r['request id'], 'phase': 'time-to-first-token',
                 'start_ms': r['arrival']/1e6, 'end_ms': first_token/1e6},
                {'request_id': r['request id'], 'phase': 'decode-after-first-token',
                 'start_ms': first_token/1e6, 'end_ms': r['end_time']/1e6}])
        for i, graph in enumerate(graphs):
            dump(output / f'graph-{i}.json', graph)
        artifacts = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                     for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['status.json', 'result.json']]
        result = {'status': 'completed', 'engine': 'KAIST LLMServingSim 2.0 + ASTRA analytical backend',
                  'config': vars(args), 'metrics': metrics, 'requests': requests, 'timeline': timeline,
                  'graphs': graphs, 'trace_count': len(texts), 'graph_file_count': len(list((inputs / 'workload').rglob('*.et'))),
                  'provenance': source, 'resources': {k:v for k,v in resource.items() if k != 'samples'},
                  'artifacts': artifacts,
                  'limitations': [
                      'No model inference or generated natural-language output: request lengths and distinct synthetic token-ID markers are simulator inputs.',
                      'Compute/kernel latency uses upstream measured profiles; changing memory/network specifications does not reprofile kernels.',
                      'Native DAG profile costs are not scheduled node start/end times. Request timeline boundaries come directly from the native request CSV.',
                      'Native total-clock duration/throughput include simulator control/idle intervals; request-span metrics use the CSV completion window.',
                      'Roofline/SRAM spill/hierarchical strategy outputs in MemAnalysis are a separately labelled behavioral extension, not native KAIST results.'
                  ]}
        dump(output / 'result.json', result)
        dump(output / 'status.json', {'status': 'completed', 'result': str((output/'result.json').relative_to(ROOT))})
        return result
    except Exception as exc:
        dump(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', '--check', dest='status', action='store_true')
    parser.add_argument('--hardware', choices=[p['hardware'] for p in PRESETS], default='RTXPRO6000')
    parser.add_argument('--input-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=8)
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--tp', type=int, choices=[1, 2], default=1)
    parser.add_argument('--memory-gib', type=float, default=None)
    parser.add_argument('--memory-bandwidth-gbps', type=float, default=None)
    parser.add_argument('--link-bandwidth-gibps', type=float, default=16)
    parser.add_argument('--link-latency-ns', type=float, default=1000)
    parser.add_argument('--cpu-memory-bandwidth-gbps', type=float, default=50)
    parser.add_argument('--arrival-interval-ms', type=float, default=0)
    parser.add_argument('--output', default='runs/native-' + uuid.uuid4().hex[:12])
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status()))
        return
    if Path(sys.executable).resolve() != PYTHON.resolve():
        os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    result = execute(args)
    print(json.dumps({'status': result['status'], 'metrics': result['metrics'],
                      'result': str((Path(args.output).resolve()/'result.json').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
