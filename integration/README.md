# Native LLMServingSim integration

`native_runner.py` launches the real KAIST Python serving scheduler, Chakra
converter, and compiled ASTRA-Sim congestion-unaware analytical backend. It
uses the upstream measured layer-latency tables; it does not execute an LLM or
invent a page-level memory trace. The strategy model in ServeScope is a
separate behavioral extension.

## Install and run

On this macOS ARM64 host, CMake and Python 3.10 are required. Dependencies,
build products, and virtual environment remain local and ignored by Git.

```bash
./integration/setup_native.sh
.venv-sim/bin/python integration/native_runner.py --check
.venv-sim/bin/python integration/native_runner.py \
  --input-tokens 128 --output-tokens 8 --requests 4 --tp 1 \
  --output runs/native-demo
```

`--check` prints readiness, pinned commits, allowed fields, hardware/model
presets, and run bounds without launching simulation. An output directory may
already contain the API job log; an existing result or request CSV is rejected.
The adapter can be launched with another Python and reexecutes its isolated
`.venv-sim` before a run. Run directories must be inside `runs/`.

Supported measured-profile combinations are Llama-3.1-8B BF16 on RTX PRO 6000
(TP 1 or 2) and RTX 4090 (TP 1). Hardware refers to the *simulated* accelerator;
these CPU-only runs execute locally without either GPU present. Memory
capacity/bandwidth, CPU bandwidth, and link bandwidth/latency are explicit
scenario parameters. Changing them does not regenerate the measured compute
latencies, so a compute-chip redesign needs new latency profiles or the
separately labelled behavioral model.

A run has a 120-second wall timeout, 2-GiB summed process-tree RSS cap, and
0.2-GiB host available-memory floor. The watchdog terminates only its child
process group and descendants. Request count is limited to 16, input length to
4096, output length to 64. The UI may impose tighter bounds. Distinct synthetic
token ID markers prevent accidental prefix sharing; they are workload inputs
and are not generated text. Prefix caching is disabled for this demo.

## Retained output

Each successful run writes:

- `cluster.json`, `workload.jsonl`, and `command.json`: exact native inputs.
- `simulator.log`, `requests.csv`: original upstream output, including TTFT,
  TPOT, ITL, queue time, and completion times in nanoseconds.
- `astra-inputs/`: network/system/memory configuration, all native text traces
  and Chakra `.et` graphs.
- `graph-0.json`, `graph-1.json`: decoded first two native batch Chakra DAGs.
- `resources.json`: process-tree watchdog observations.
- `result.json`: UI summary, native request data, graphs, per-request timeline,
  source/build/profile hashes, limitations, and hashed artifact inventory.

`graphs[].nodes[].profile_cost_ns` is a per-node profile cost, not a scheduled
start/end timestamp. KAIST's converter stores nanoseconds in the protobuf field
named `duration_micros`, and its backend consumes that value without a 1000x
conversion. Edges are the actual Chakra data/control dependencies.
`timeline` uses the CSV's arrival, TTFT, and completion boundaries. The TTFT
interval includes queueing and prefill; it must not be relabelled as pure
prefill computation. The adapter does not invent per-stage scheduled intervals.

Native throughput is completed output tokens divided by the upstream total
simulation clocks, matching its printed throughput summary.
`request_span_ms` and `request_span_output_tokens_per_second` additionally
report the interval from earliest arrival to last completion; that span can
exclude a final idle/control interval included in the native total clocks.
`all_tokens_per_second` additionally counts input tokens. Metrics refer to the
simulated workload, not local CPU wall time.

## Provenance and native build

Pinned source commits:

| Component | Commit |
| --- | --- |
| LLMServingSim | `b09c7aaab4b072d73136641841cc2688616c49eb` |
| ASTRA-Sim | `d3d068f4050a2fb9a7538cc671a982152c2ba9b7` |
| Chakra | `5e6e9b28438bc1899105dbdd68eed4cf61bee6fd` |
| C++ protobuf 3.21.12 | `f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c` |

The runner refuses modified tracked upstream sources or a different pin.
Upstream source remains pristine. The isolated CMake wrapper adds generated
protobuf C++ bindings from `.native/generated`, links static protobuf's zlib
dependency, and enables libc++'s legacy `std::binary_function` declaration.
This adapts macOS compilation without changing simulator logic. A gitignored
symlink in upstream's expected build directory points to the isolated binary.

Upstream's Python 3.10 NumPy/Pandas/Matplotlib pins are retained. Chakra's
current required Python protobuf runtime is installed; its LLM converter is
installed without the unused HolisticTraceAnalysis/GPU conversion dependency.
The C++ and Python bindings use the same pinned `.proto` schema and were
verified together by real end-to-end runs. Full dependency versions are in
`requirements-native.txt`.

Sources: [KAIST repository](https://github.com/casys-kaist/LLMServingSim),
[upstream installation guide](https://llmservingsim.ai/docs/getting-started/installation/simulator),
[trace format](https://llmservingsim.ai/docs/reference/trace-format).

## Analytical versus profile-backed comparison

See the [nine-scenario comparison and reproduction instructions](../docs/benchmarks/roofline-vs-profiles-2026-10-19/README.md) for measured-profile versus analytical timing differences.
