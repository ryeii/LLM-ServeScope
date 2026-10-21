# Archived Mooncake request dumps

These JSONL files are unchanged published workloads from Mooncake's FAST25 release,
pinned at e4229791f39d1f4cdb00df1b28366130f742deb0. See manifest.json for source URLs,
counts and SHA-256 hashes, UPSTREAM-README.md for original field semantics, and
LICENSE-APACHE for the retained license. Credit belongs to Mooncake contributors.

ServeScope selects explicit contiguous windows as behavioral-simulation inputs.
Original arrivals, input/output lengths and known 512-token prefix hashes are
preserved; computed cache hits, transfers and completions are simulated outputs.
No token text, hardware cache outcomes or physical memory events are present.

This does not change MemScope's separate dumped-as-is-trace visualization.
