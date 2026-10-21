# Independent KV placement comparison

These are model predictions, **not measured serving benchmarks or validation**.
All runs hold the original default model, hardware, workload, batch caps and manual I/O settings fixed. The default hybrid resident budget is 1 GB per decode GPU; pure backing modes retain no active KV in HBM beyond staging.

| Active decode placement | Retained prefixes | Generation tok/s | Mean TPOT ms | Remote decode KV reads GB |
| --- | --- | ---: | ---: | ---: |
| hbm | disabled | 186.6349 | 39.1740 | 0.0000 |
| dram | disabled | 29.3516 | 501.5716 | 686.6967 |
| ssd | disabled | 3.0389 | 5178.3482 | 686.6967 |
| hbm-dram | disabled | 42.0040 | 284.6552 | 438.5167 |
| hbm-ssd | disabled | 4.7127 | 2840.7616 | 438.5167 |
| hbm | ssd-only | 127.8845 | 64.5463 | 0.0000 |
| dram | ssd-only | 27.3738 | 522.3801 | 686.6967 |
| ssd | ssd-only | 3.0163 | 5178.3482 | 686.6967 |
| hbm-dram | ssd-only | 38.0680 | 359.6775 | 438.5167 |
| hbm-ssd | ssd-only | 4.6587 | 3339.8371 | 438.5167 |

See [model assumptions](active-kv-model.md), [resolved configurations and numbers](../examples/active-kv-comparison.json), and [generator](../integration/storage/active_comparison.py).

```bash
.venv/bin/python -B -m integration.storage.active_comparison
```
