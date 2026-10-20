"""Offline KAIST profile reader used only for training/data conformance.

Interpolation adapted from LLMServingSim, Copyright (c) 2024 CASYS Lab at KAIST.
MIT license retained in licenses/LLMServingSim-MIT.txt. Not imported by simulation.
"""
import bisect

def interpolate(points, query):
    """Pinned upstream linear blend: clamp below grid, extrapolate above it."""
    if not points:
        raise ValueError("Empty calibration axis")
    if len(points) == 1 or query <= points[0][0]:
        return float(points[0][1])
    i = bisect.bisect_right(points, query, key=lambda p: p[0])
    lo, hi = (len(points)-2, len(points)-1) if i == len(points) else (i-1, i)
    x0, y0 = points[lo]
    x1, y1 = points[hi]
    return y0 + (query-x0)/(x1-x0)*(y1-y0)


def axis_info(points, query):
    keys = [p[0] for p in points]
    return query in keys, keys[0] <= query <= keys[-1]


def lookup(profile, name, n, phase="decode", pc=0, kp=0, nd=1, kv=0):
    """Return integer ns and provenance for a uniform, pure-phase batch.

    Pure-phase slices preserve upstream interpolation exactly. Mixed prefill/decode
    and heterogeneous KV lengths are outside ServeScope's fixed-batch scheduler.
    """
    if name != "attention":
        points = profile["per_sequence" if name in ("lm_head", "sampler") else "dense"][name]
        value = max(1, int(interpolate(points, max(1, int(n)))))
        exact, inside = axis_info(points, max(1, int(n)))
    else:
        outer = profile["attention"][phase]
        x, inner_x = (int(pc), float(kp)) if phase == "prefill" else (int(nd), int(kv))
        slices = []
        inner_exact, inner_inside = {}, {}
        for coordinate, points in outer:
            slices.append([coordinate, interpolate(points, max(0, inner_x))])
            inner_exact[coordinate], inner_inside[coordinate] = axis_info(points, max(0, inner_x))
        value = max(1, int(interpolate(slices, max(0, x))))
        outer_exact, outer_inside = axis_info(slices, max(0, x))
        # Only the bracket actually used determines coverage. Unused corners do
        # not turn a valid pure-phase lookup into an out-of-range estimate.
        keys = [p[0] for p in slices]
        if x in keys:
            active = [x]
        elif x < keys[0]:
            active = [keys[0]]
        else:
            i = bisect.bisect_right(keys, x)
            active = keys[max(0, min(i, len(keys)-1)-1):min(i+1, len(keys))]
        inside = outer_inside and all(inner_inside[k] for k in active)
        exact = outer_exact and inner_exact[x]
    return value, "profile-grid" if exact else "profile-interpolated" if inside else "profile-shape-extrapolated"
