"""gta6-trophy-rarity-forecast-01 entry: forecast every GTA VI PS5 trophy's
Exophase earn rate at the case's target, as five quantiles per trophy.

Model (fitted at run time from the case's own history, standard library only):

A trophy's earn rate is a ratio, earners / players tracked, and the two sides
move for different reasons, so the model forecasts them separately:

  * players tracked N(t): fit ln N against ln(1 + hours since release) over
    the distinct page states in the case's history (Exophase refreshes its
    counts in steps, so repeated identical reads are one state), then read
    the fit at the target, blended with a damped-slope version of the same
    fit so a hot launch week is not extrapolated at full tilt forever.
  * earners E_i(t) per trophy: the same log-log fit on ln(E_i + 0.5), with
    each trophy's slope shrunk toward the median slope of its tier (a trophy
    with four earners has a noisy slope; its tier's does not).

The structural median share is E_i(target) / N(target). It is blended on
the logit scale with a logit-trend extrapolation of the trophy's own
recent rates (the strongest single signal: earn rates move almost
linearly in ln(1 + hours) over a launch week) and with persistence (the
last rate), then the move away from the last rate is damped once more.
The spread around the median grows with the square root of the horizon,
widens when the structural and trend views disagree, and widens for
trophies held by only a handful of players. The blend weights and spread
were set on synthetic backtests (cohort-arrival worlds with known ground
truth, including post-checkpoint surge and slump variants), where this
blend beat both house baselines; see the repository README.

A history that shows a single frozen page (the practice cases, cut before
Exophase's first refresh) cannot fit any of this: the model then expects
the refresh it has not seen, shifting shares down by a dilution factor
(newly tracked players hold fewer trophies) with a wide spread.

Deterministic: no randomness, no clock, no network. Runs in milliseconds.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Fit and blend constants (chosen by the synthetic backtests in work/).
N_DAMP = 0.72          # slope damping for the cautious players-tracked fit
N_BLEND = 0.62         # weight of the full-slope fit in the N forecast
E_SMOOTH = 0.5         # earner smoothing so ln(E) survives E = 0
SHRINK_K = 3.0         # earner-slope shrinkage strength toward the tier median
SHRINK_K_T = 3.0       # trend-slope shrinkage strength toward the tier median
FIT_WINDOW_H = 48.0    # hours of history the trajectory fits may use
TREND_DAMP = 0.8       # damping of the trend extrapolation's slope
W_STRUCT = 0.20        # median blend weights: structural / trend / last
W_TREND = 0.65
W_LAST = 0.15
MOVE_DAMP = 0.88       # final damping of the median's move from the last rate
SIGMA_DAY = 0.45       # logit spread per square-root day of horizon
SIGMA_FLOOR = 0.50
SIGMA_CAP = 2.30
DISAGREE_K = 0.55      # spread added per unit of structural/trend disagreement


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def expit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> float:
    """ISO 8601 UTC -> epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def wls_slope(xs, ys, ws):
    """Weighted least squares; returns (intercept, slope)."""
    sw = sum(ws)
    if sw <= 0:
        return (ys[-1] if ys else 0.0), 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    return my - slope * mx, slope


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def distinct_states(history):
    """Collapse consecutive reads with identical page counts into one state.

    Exophase refreshes in steps; between refreshes every read shows the same
    page, and only the refresh points carry trajectory information.
    """
    states = []
    for read in history:
        key = (read.get("players_tracked"), read.get("trophies_earned"))
        if states and states[-1][0] == key:
            continue
        states.append((key, read))
    return [read for _, read in states]


def tier_of(trophy):
    return trophy.get("tier") or "bronze"


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = max(case["horizon_minutes"], 1) / 1440.0

    # Time origin: midnight UTC on the release date, else the first read.
    try:
        rel = case["game"]["release_date"]
        t0 = datetime.fromisoformat(rel).replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        t0 = parse_time(history[0]["at"]) if history else 0.0

    def hours(at_text):
        return max(0.0, (parse_time(at_text) - t0) / 3600.0)

    target_h = hours(case["target"])

    if not history:
        # No data at all: stay valid with flat, wide rows by tier.
        prior = {"platinum": 0.1, "gold": 0.8, "silver": 2.5, "bronze": 6.0}
        rows = []
        for t in trophies:
            lg = logit(prior.get(tier_of(t), 5.0))
            rows.append({"id": t["id"], "q": [round4(expit(lg + z * 1.2)) for z in Z]})
        return {"trophies": rows}

    last = history[-1]
    n_last = float(last.get("players_tracked") or 0)
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            r = read["rates"][i] if i < len(read["rates"]) else None
            if r is not None:
                rate = r
                break
        last_rates.append(rate if rate is not None else 0.0)

    states = distinct_states(history)
    span_h = hours(states[-1]["at"]) - hours(states[0]["at"]) if states else 0.0

    if len(states) < 3 or span_h < 6.0 or n_last <= 0:
        return cold_forecast(trophies, last_rates, horizon_days)

    # Fit window: the last FIT_WINDOW_H hours of states, at least the last 6.
    h_last_state = hours(states[-1]["at"])
    window = [s for s in states if hours(s["at"]) >= h_last_state - FIT_WINDOW_H]
    if len(window) < 6:
        window = states[-6:]
    if len(window) < 3:
        window = states

    xs = [math.log1p(hours(s["at"])) for s in window]
    m = len(xs)
    ws = [0.35 + 0.65 * (j / (m - 1) if m > 1 else 1.0) for j in range(m)]
    x_last = xs[-1]
    x_tgt = math.log1p(target_h)

    # --- players tracked -------------------------------------------------
    ln_n = [math.log(max(float(s.get("players_tracked") or 1), 1.0)) for s in window]
    a_n, b_n = wls_slope(xs, ln_n, ws)
    ln_n_full = a_n + b_n * x_tgt
    ln_n_damped = math.log(max(n_last, 1.0)) + N_DAMP * b_n * (x_tgt - x_last)
    ln_n_tgt = N_BLEND * ln_n_full + (1.0 - N_BLEND) * ln_n_damped
    ln_n_tgt = max(ln_n_tgt, math.log(max(n_last, 1.0) * 1.02))
    ln_n_tgt = min(ln_n_tgt, math.log(max(n_last, 1.0) * 40.0))
    n_tgt = math.exp(ln_n_tgt)

    # --- per-trophy earner and trend fits --------------------------------
    raw = []          # per trophy: (b_e, b_trend, n_points)
    for i in range(len(trophies)):
        ex, ey, ty = [], [], []
        for j, s in enumerate(window):
            r = s["rates"][i] if i < len(s["rates"]) else None
            if r is None:
                continue
            earners = r / 100.0 * float(s.get("players_tracked") or 0)
            ex.append(xs[j])
            ey.append(math.log(earners + E_SMOOTH))
            ty.append(logit(r))
        if len(ex) >= 3:
            _, b_e = wls_slope(ex, ey, [0.35 + 0.65 * (k / (len(ex) - 1)) for k in range(len(ex))])
            _, b_t = wls_slope(ex, ty, [0.35 + 0.65 * (k / (len(ex) - 1)) for k in range(len(ex))])
        else:
            b_e, b_t = None, None
        raw.append((b_e, b_t, len(ex)))

    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        tier_slopes[tier] = (
            median([raw[i][0] for i in range(len(trophies)) if tier_of(trophies[i]) == tier and raw[i][0] is not None]),
            median([raw[i][1] for i in range(len(trophies)) if tier_of(trophies[i]) == tier and raw[i][1] is not None]),
        )
    all_e = median([r[0] for r in raw if r[0] is not None])
    all_t = median([r[1] for r in raw if r[1] is not None])

    rows = []
    for i, trophy in enumerate(trophies):
        tier = tier_of(trophy)
        r_last = last_rates[i]
        lg_last = logit(r_last)
        e_last = r_last / 100.0 * n_last
        b_e, b_t, n_pts = raw[i]
        pool_e = tier_slopes[tier][0] if tier_slopes[tier][0] != 0.0 else all_e
        pool_t = tier_slopes[tier][1] if tier_slopes[tier][1] != 0.0 else all_t
        if b_e is None:
            b_e = pool_e
        if b_t is None:
            b_t = pool_t
        b_e_shrunk = (b_e * n_pts + SHRINK_K * pool_e) / (n_pts + SHRINK_K)
        b_t_shrunk = (b_t * n_pts + SHRINK_K_T * pool_t) / (n_pts + SHRINK_K_T)

        e_tgt = (e_last + E_SMOOTH) * math.exp(b_e_shrunk * (x_tgt - x_last)) - E_SMOOTH
        e_tgt = min(max(e_tgt, 0.0), 0.999 * n_tgt)
        share_struct = 100.0 * e_tgt / n_tgt if n_tgt > 0 else r_last
        lg_struct = logit(share_struct)
        lg_trend = lg_last + TREND_DAMP * b_t_shrunk * (x_tgt - x_last)

        lg_med = W_STRUCT * lg_struct + W_TREND * lg_trend + W_LAST * lg_last
        lg_med = lg_last + MOVE_DAMP * (lg_med - lg_last)

        # A trophy everyone already holds can only dilute from ~100%.
        if r_last >= 99.9:
            lg_med = min(lg_med, logit(99.5))

        disagree = min(abs(lg_struct - lg_trend), 1.5)
        small = min(0.6, 1.2 / math.sqrt(max(e_last, 1.0)))
        sigma = math.sqrt((SIGMA_DAY * math.sqrt(horizon_days)) ** 2
                          + (DISAGREE_K * disagree) ** 2 + small ** 2)
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        if lg_med >= lg_last:
            sig_lo, sig_hi = sigma * 0.92, sigma * 1.08
        else:
            sig_lo, sig_hi = sigma * 1.08, sigma * 0.92

        qs = []
        for z in Z:
            s = sig_lo if z < 0 else sig_hi
            qs.append(round4(expit(lg_med + z * s)))
        # Enforce non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def cold_forecast(trophies, last_rates, horizon_days):
    """One frozen page state: expect a refresh the history has not seen.

    A refresh swaps the pre-release crowd for the real one: the tracked
    count jumps and every rate dilutes, because the new players hold fewer
    trophies. Shift shares down (less for the cheap early trophies new
    players earn at once), with a wide, honest spread.
    """
    dil = {"platinum": 0.80, "gold": 0.78, "silver": 0.74, "bronze": 0.70}
    depth = min(1.0, math.sqrt(horizon_days / 4.52))
    sigma = 0.85 + 0.40 * math.sqrt(horizon_days)
    rows = []
    for t, r_last in zip(trophies, last_rates):
        factor = 1.0 - (1.0 - dil.get(tier_of(t), 0.72)) * max(depth, 0.35)
        if r_last >= 99.9:
            med = 99.3
        elif r_last <= 0:
            med = 0.05
        else:
            med = clamp_pct(r_last * factor)
        lg = logit(med)
        qs = [round4(expit(lg + z * sigma)) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
