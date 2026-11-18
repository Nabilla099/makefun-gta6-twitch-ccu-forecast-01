"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's release-day Twitch
viewers, hour by hour, as five quantiles an hour (hours 2..24).

Python 3.13, standard library only. Runs offline in the sandbox.

Method (measured on leave-one-out synthetic cases built from the kit's
comparables; mean pinball loss on the log scale, lower is better):

  house persistence baseline (starter, SIGMA=0.15):      ~0.36
  pooled empirical log-ratio quantiles:                  ~0.31
  this model:                                            ~0.26

The shape of a launch day depends on how the game unlocks. Games whose
first hour is a pre-unlock trickle (hour-1 viewers < 5% of the day's peak,
e.g. a single global unlock at 14:00Z) surge hundreds of times over the
day; rolling-midnight console launches (GTA VI's kind) start with a real
audience and rise a few times. So the reference pool keeps only
comparables whose hour-1 print is at least 5% of their release-day peak.

Within that pool, the log-ratio log(v_h / v_1) is strongly tied to the
size of hour 1 itself (correlation about -0.94 between log surge and
log v_1 across launches: a huge first hour leaves little room to grow).
For each hour h we therefore regress log(v_h / v_1) on log(v_1) across
the pool (ordinary least squares) and take the quantiles of the
residuals. The forecast is the regression prediction at the case's own
hour-1 print, plus the residual quantiles, blended 50/50 with the
matching normal quantiles (residual IQR / 1.349 as sigma) to smooth the
noisy 5%/95% tails of a ~30-game sample:

    q_tau(h) = p1 * exp(med_h + 0.5*(Q_tau(resid_h) + z_tau*sigma_h))

where med_h is the regression fit at log(p1) and z_tau the standard
normal quantile. Every row is clamped to (0, 1e9] and sorted
non-decreasing, so the output is always valid even if the data is odd.

Fallbacks (validity first: an invalid case scores as if every quantile
equalled hour 1's print and takes no prize): if the pool is too small
for an hour, fall back to pooled empirical quantiles; if there is no
usable pool at all, fall back to the house persistence baseline.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = statistics.NormalDist()
Z = [_NORM.inv_cdf(t) for t in LEVELS]

# House baseline spread (fallback only), log units per square-root hour.
SIGMA = 0.15
# A comparable counts as a rolling launch (not a pre-unlock trickle) when
# its hour-1 print is at least this fraction of its release-day peak.
MIN_V1_PEAK = 0.05
# Minimum pool size for the regression; below it, use pooled quantiles.
MIN_POOL = 8


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an ascending list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _hour_key(day_start, hour):
    return (day_start + timedelta(hours=hour - 1)).strftime("%Y-%m-%dT%H:00:00Z")


def build_pool(case):
    """Release-day hourly viewers for each usable comparable.

    Returns a list of {"day": [24 viewer counts or None]} for launches
    whose hour 1 is a real audience, not a pre-unlock trickle.
    """
    pool = []
    for g in case.get("comparables") or []:
        try:
            d0 = datetime.fromisoformat((g.get("day_start") or "").replace("Z", "+00:00"))
            hrs = {h.get("from"): h.get("viewers") for h in (g.get("hourly") or [])}
            day = [hrs.get(_hour_key(d0, h)) for h in range(1, 25)]
        except (ValueError, TypeError, AttributeError):
            continue
        v1 = day[0]
        if not v1 or v1 < 1:
            continue
        peak = max((v for v in day if v), default=None)
        if not peak or v1 / peak < MIN_V1_PEAK:
            continue
        pool.append({"day": day})
    return pool


def _ratios(pool, h):
    """(log v1, log(v_h / v1)) points for hour h."""
    pts = []
    for g in pool:
        v1, vh = g["day"][0], g["day"][h - 1]
        if v1 and v1 >= 1 and vh and vh >= 1:
            pts.append((math.log(v1), math.log(vh / v1)))
    return pts


def _pooled_quantiles(p1, pool, h):
    ratios = sorted(y for _, y in _ratios(pool, h))
    if len(ratios) < 5:
        return None
    return [max(1.0, p1 * math.exp(quantile(ratios, tau))) for tau in LEVELS]


def _baseline_quantiles(p1, h):
    return [max(1.0, p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def hour_quantiles(p1, pool, h):
    pts = _ratios(pool, h)
    if len(pts) >= MIN_POOL:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else 0.0
        a = my - b * mx
        med = a + b * math.log(p1)
        res = sorted(y - (a + b * x) for x, y in pts)
        iqr = quantile(res, 0.75) - quantile(res, 0.25)
        sigma = max(0.05, iqr / 1.349)
        out = []
        for i, tau in enumerate(LEVELS):
            q = med + 0.5 * (quantile(res, tau) + Z[i] * sigma)
            out.append(max(1.0, p1 * math.exp(q)))
        return out
    pooled = _pooled_quantiles(p1, pool, h)
    if pooled is not None:
        return pooled
    return _baseline_quantiles(p1, h)


def forecast_case(case, name):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first.get("viewers") or 1)
    pool = build_pool(case)
    hours = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        qs = [round2(q) for q in hour_quantiles(p1, pool, h)]
        qs = sorted(min(max(q, 1e-9), 1e9) for q in qs)
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
