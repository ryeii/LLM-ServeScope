# Prefix-layout comparisons and the batching regression

These are **ServeScope simulation results**, not GPU/SSD measurements. The original UI
silently forced prefill/decode batch caps from four to one when a layout was selected.
That made the comparison invalid: the disabled default and the tier runs had different
batching. Selecting a layout now preserves model, workload, batch caps and replicas.

Uniform-length requests use the same fixed microbatch formation, GPU operation shapes,
decode splitting and resource queues. Prefix mode reserves active KV for a full group
atomically. A fixed rectangular prefill batch uses the minimum restored prefix among its
members; a higher-hit member recomputes excess tokens. Variable-length request replay
still requires batch size one. Unsupported replica counts are rejected explicitly.
There is no automatic reduction of batching or replica counts. Imported records
keep their original arrivals and token lengths when retention is turned off;
retention-off replay allocates no retained cache capacity.

## Reproduction

Llama 3.1 70B, BF16, separate H100 TP4 prefill/decode pools, 16 requests,
2048 input tokens, 64 output tokens, 8 requests/s. Original/default batch caps: 4/4.
The configured SSD profile is manual: read 3 GB/s + 100 us per I/O chunk; write
2 GB/s + 200 us per chunk; chunk size 1 MiB. HBM prefix capacity 0.125 GB/GPU,
host DRAM 0.25 GB aggregate, SSD 4 GB aggregate. The fitted local file profile is
not selected in this comparison. Publication and spill costs block before handoff.

| Configuration | Prefill / decode caps | Generation throughput (tok/s) | Reused prompt tokens |
| --- | --- | ---: | ---: |
| Original disabled baseline | 4 / 4 | 186.634944 | 0 |
| Batch change alone, no prefix retention | 1 / 1 | 58.124347 | 0 |
| No retained prefix cache | 4 / 4 | 186.634944 | 0 |
| HBM only | 4 / 4 | 186.628130 | 0 |
| DRAM only | 4 / 4 | 182.901078 | 0 |
| SSD only | 4 / 4 | 127.884536 | 2048 |
| HBM → DRAM spill | 4 / 4 | 183.353044 | 0 |
| DRAM → HBM spill | 4 / 4 | 179.533767 | 0 |
| HBM → SSD spill | 4 / 4 | 131.931347 | 2048 |
| DRAM → SSD spill | 4 / 4 | 129.762550 | 2048 |
| SSD → HBM spill | 4 / 4 | 127.884536 | 2048 |
| SSD → DRAM spill | 4 / 4 | 127.884536 | 2048 |
| HBM → DRAM → SSD spill | 4 / 4 | 133.931027 | 2048 |
| Custom inclusive copies · original policy | 4 / 4 | 129.240231 | 2048 |

“No retained prefix cache” matches **every original default performance metric and
per-request token timestamp exactly**. Its optional active-memory ledger reports
observed allocations, while disabled mode retains the original conservative resident
KV accounting. Under memory pressure, explicit group admission can introduce waits;
this regression proves equality for the documented setup, not every possible policy.

## What SSD selection means

This controlled comparison holds **active attention KV in GPU HBM**. The layout controls
retained prompt prefixes used between requests. A hit restores ready prefix blocks
before prefill; new retained blocks and evicted blocks incur publication/spill work.
Decode does not stream its working KV from SSD at each output token. Thus SSD prefix
retention is not a simulation of active attention-KV offload.

Active decode placement is now a separate control, with explicit per-step reads,
append writes, staging capacity and shared I/O queues. See the
[active-KV model](active-kv-model.md) for equations and its unvalidated serialized
streaming assumptions. No active-offload accuracy claim is inferred from these
prefix-cache experiments or from local file-I/O calibration.

A prefix layout can have a modest effect when decoding dominates, few ready prefixes
are reused, capacity is small or arrival rate limits the finite run. Prefix reuse also
saves prefill work while reads/publication add cost, so throughput need not be ordered
solely by the name of the retention tier.

[Resolved configurations and results](../examples/prefix-layout-comparison.json)
· [Reproduction script](../integration/storage/comparison.py)

```bash
.venv/bin/python -B -m integration.storage.comparison
.venv/bin/python -B -m unittest tests.test_storage
```
