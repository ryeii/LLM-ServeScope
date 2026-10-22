"""Configured SSD transfer paths; no measured GDS accuracy claim.

The storage service function accounts for the file/device path. A direct GDS
transfer is a single streaming operation limited by storage or the GPU link,
plus a configured startup. It must not also charge a host--GPU bounce copy.
"""


def direct_service(storage_ms, nbytes, bandwidth_GBps, latency_us):
    link_ms = nbytes / (bandwidth_GBps * 1e6)
    startup_ms = latency_us / 1000
    return dict(
        storage_ms=storage_ms,
        link_ms=link_ms,
        startup_ms=startup_ms,
        duration_ms=startup_ms + max(storage_ms, link_ms),
        bandwidth_GBps=bandwidth_GBps,
        equation="startup_ms + max(storage_ms, payload_bytes / (GDS_GBps * 1e6))",
        calibration="configured-gds-assumption",
    )


def active_paths(backing, transfer_mode, operation):
    if backing != "ssd":
        return ["pcie"]
    if transfer_mode == "gds":
        return ["ssd"]
    return ["ssd", "pcie"] if operation == "read" else ["pcie", "ssd"]
