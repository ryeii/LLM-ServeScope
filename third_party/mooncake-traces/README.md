# Archived Mooncake request dumps

These JSONL files are unchanged published workloads from Mooncake's FAST25 release,
pinned at e4229791f39d1f4cdb00df1b28366130f742deb0. See manifest.json for source URLs,
counts and SHA-256 hashes, UPSTREAM-README.md for original field semantics, and
LICENSE-APACHE for the retained license. Credit belongs to Mooncake contributors.

ServeScope analyzes the unchanged records offline to estimate input-token-weighted
earlier-arrival prefix reuse potential. No trace requests are replayed in its
current uniform-workload UI. Optional reuse fractions assume the corresponding
KV is already ready in HBM; they are not measured hits or serving validation.
See [the analysis](../../docs/prefix-reuse/README.md). No raw text, hardware cache
outcomes or physical memory events are present.

This does not change MemScope's separate dumped-as-is-trace visualization.
