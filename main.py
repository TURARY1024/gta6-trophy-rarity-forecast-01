"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads one case (makefun/gta6-trophies-case/v1): every Exophase read of the game's
PS5 trophy page up to the checkpoint, and writes, for each trophy, five quantiles
of its earn rate (percent of tracked players holding it) at the case's target.

Model (see METHODOLOGY.md for the full write-up)
-----------------------------------------------
An earn rate is a ratio, rate = N / P, where P is the players Exophase tracks and
N is how many of them hold the trophy.  The two move for different reasons, so
they are forecast separately and divided:

* P(t): tracked players grow as more people buy the game, play it and sync a
  profile.  log P is fit as a straight line in x = ln(1 + hours since release),
  the line a least-squares fit over the case's distinct reads (consecutive reads
  with identical values are one read: Exophase refreshes its counts hours apart,
  so the page sits flat between refreshes).  With fewer than two distinct
  values the slope falls back to a prior of 1.1, the slope the documented
  launch-night refresh implies (638 -> 3,868 tracked players over ~4.5 hours).
* N_i(t): how many tracked players hold trophy i.  log N_i is fit the same way,
  its slope shrunk toward a prior that depends on what the trophy asks for
  (story chapters rise steeply, 250-stunt grinds barely move) and on how far
  into the launch the case sits (in the first hours almost nobody has had time
  to earn anything past the opening mission, so the priors are small early and
  mature later).  The fit is anchored on the recent reads, so the slope it
  finds is the slope the trophy has now, not its launch-night average.
* The answer is the page at the target as of Exophase's last refresh before
  it, so the forecast is evaluated half a refresh interval before the target
  (the interval estimated from the gaps between distinct reads in the case,
  prior 4 h, lag capped at 2 h).
* The median rate is N_hat / P_hat at that effective time, capped by a
  per-class ceiling (no trophy can exceed the share of players who will ever
  plausibly earn it by day 7: 80% for Chapter 1, 2% for the Platinum) -- and
  by a saturating fit to the rate series itself (rate = C * sigmoid(a + b*x),
  C grid-searched): a rate's growth decelerates as the fast players finish
  and the slow ones remain, so the forecast takes the lower of the trend
  extrapolation and the saturating fit.
* Spread: a logit sigma combining the two fits' prediction intervals (wide
  when the history is flat: a flat page says nothing about the next refresh),
  a random-walk floor growing with the horizon, a term for the refresh jump
  (a refresh can move the denominator a lot in the first days), and a flat
  0.25 for model misspecification.  Quantiles are the median shifted by the
  normal quantiles times sigma, on the logit scale.

Deterministic: pure float arithmetic, fixed iteration order, no randomness,
no network, standard library only.  Two runs on the same case write the same
bytes.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs: python3 -P -s /bot/main.py, reading /input/<id>.in and writing
/output/<id>.out, at most 8 s a case.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- priors ---------------------------------------------------------------

# Slope prior for log tracked-players against x = ln(1 + hours since release),
# used when the case's reads show no movement.  Launch night: 638 -> 3,868
# tracked players over ~4.5 h is a slope of ~1.06 in this space.
PRIOR_P_SLOPE = 1.1
PRIOR_P_WEIGHT = 2.0  # pseudo-observations the prior counts as

# Prior for the refresh gap (hours) when the case shows no refresh at all.
PRIOR_REFRESH_GAP_H = 4.0
MAX_LAG_H = 2.0

# Shrinkage weight of the per-trophy slope prior, in pseudo-observations.
PRIOR_SLOPE_WEIGHT = 3.0

# Spread pieces, in logit units.
SIGMA_FLAT_HISTORY = 0.9   # no distinct reads: the next refresh is a guess
SIGMA_MISSPEC = 0.25        # the model is wrong in ways the fits cannot see
SIGMA_RW_PER_SQRT_DAY = 0.30
SIGMA_MIN = 0.30
SIGMA_MAX = 2.0

# Per-class priors: (early slope, mature slope, day-7 ceiling as a fraction).
# "Early" is the slope in the first ~6 h after release.  Launch night shows why
# it is small: between the first probe and the first refresh (~4.5 h) the
# non-first trophies grew only 305 -> 408 earned while tracked players grew
# 638 -> 3,868, an average log-earners slope of ~0.17 -- in the first hours
# almost nobody has had time to earn anything past the opening mission.
# "Mature" is the slope once players are deep into the game (past ~30 h).
# Between them the prior interpolates linearly in hours since release.
# Slopes are for log earners against x = ln(1 + hours since release).
# Ceilings are the share of tracked players who could plausibly hold the
# trophy by day 7 at all.
CLASS_PRIORS = {
    "first": (None, None, 1.00),       # the opening trophy: earners == tracked
    "story1": (0.35, 2.0, 0.80),
    "story2": (0.30, 1.8, 0.65),
    "story3": (0.25, 1.6, 0.50),
    "story4": (0.20, 1.4, 0.40),
    "story_final": (0.15, 1.2, 0.30),
    "side_mission": (0.30, 1.4, 0.45),
    "side_strand": (0.25, 1.3, 0.35),
    "easy": (0.40, 1.6, 0.60),
    "mid": (0.25, 1.1, 0.30),
    "grind": (0.15, 0.9, 0.15),
    "grind_hard": (0.10, 0.7, 0.06),
    "explore": (0.15, 0.9, 0.20),
    "completion": (0.05, 0.6, 0.03),
    "platinum": (0.05, 0.6, 0.02),
    "misc": (0.25, 1.1, 0.30),
}


def classify(trophy: dict) -> str:
    """What kind of trophy this is, from its name, description and tier."""
    name = trophy.get("name", "") or ""
    desc = trophy.get("description", "") or ""
    text = (name + " " + desc).lower()
    tier = trophy.get("tier", "")

    if tier == "platinum":
        return "platinum"
    if "100% completion" in text or "attain 100%" in text:
        return "completion"
    m = re.search(r"complete chapter (\d)", text)
    if m:
        return {"1": "story1", "2": "story2", "3": "story3", "4": "story4"}.get(m.group(1), "story4")
    if "complete the final mission" in text and "for " not in text:
        return "story_final"
    if "complete the final mission for" in text:
        return "side_strand"
    if "gold medal in all" in text:
        return "grind_hard"
    num = re.search(r"(\d+)\s*(stunts|locations|challenges|races|vehicles)", text)
    if num and int(num.group(1)) >= 100:
        return "grind_hard"
    if num and int(num.group(1)) >= 20:
        return "grind"
    if re.search(r"\b(250|100)\b", text) and ("stunt" in text or "location" in text):
        return "grind_hard"
    if "complete 3 collectible card sets" in text:
        return "grind_hard"
    if "complete 8 mission strands" in text:
        return "side_mission"
    if text.startswith("complete ") or "complete " in text:
        return "side_mission"
    if "every spot" in text or "explore all" in text or "meditate" in text:
        return "explore"
    if ("try each type" in text or "arcade game" in text or "cocktail" in text
            or "bowling" in text or "bowl 'em over" in text):
        return "easy"
    if re.search(r"\b(rob|spend|earn|purchase|register|hunt|beat|win|lose|take down|make an)\b", text):
        return "mid"
    return "misc"


# --- small numeric helpers -------------------------------------------------

def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ols(xs: list[float], ys: list[float], ws: list[float] | None = None):
    """Weighted least squares of y on x.  Returns (intercept, slope, s, xbar, sxx, n)
    with s the residual standard deviation (0 when it cannot be estimated)."""
    n = len(xs)
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return ybar, 0.0, 0.0, xbar, 0.0, n
    slope = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys)) / sxx
    intercept = ybar - slope * xbar
    if n > 2:
        rss = sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        dof = max(n - 2, 1)
        s = math.sqrt(rss / dof)
    else:
        s = 0.0
    return intercept, slope, s, xbar, sxx, n


def pred_interval_var(s: float, n: int, xbar: float, sxx: float, x_target: float) -> float:
    """Variance of a new observation's log value at x_target, from an OLS fit."""
    if n < 3 or sxx <= 0:
        return SIGMA_FLAT_HISTORY ** 2
    s2 = min(max(s, 0.05), 1.2) ** 2
    return s2 * (1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx)


def saturating_rate_fit(xs, rates_pct, ws, x_target, c_class_pct):
    """Fit a saturating curve to an observed rate series: rate = C * sigmoid(a + b*x).

    A rate cannot keep rising on a straight line in logit space: as the early
    wave of players finishes a chapter, the remaining players are slower and
    the rate's growth decelerates toward a ceiling.  C is grid-searched (with a
    prior pulling it toward the class ceiling); a and b are least squares on
    logit(rate / C).  Returns the fitted rate at x_target (percent), or None
    when the series cannot support a fit.
    """
    obs = [(x, r, w) for x, r, w in zip(xs, rates_pct, ws)
           if r is not None and r > 0.0]
    if len(obs) < 3:
        return None
    rmax = max(r for _, r, _ in obs)
    lo = max(rmax * 1.05, 0.02)
    hi = 99.0
    if lo >= hi:
        return None
    cands = set()
    n_grid = 25
    for k in range(n_grid):
        f = k / (n_grid - 1)
        cands.add(round(math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo))), 4))
    if lo < c_class_pct < hi:
        cands.add(round(c_class_pct, 4))
    best = None
    for c in sorted(cands):
        ys = [logit(100.0 * r / c) for _, r, _ in obs]
        fxs = [x for x, _, _ in obs]
        fws = [w for _, _, w in obs]
        a, b, _s, _xb, _sxx, _n = ols(fxs, ys, fws)
        rss = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(fws, fxs, ys))
        score = rss + 1.0 * (math.log(c / c_class_pct)) ** 2
        if best is None or score < best[0]:
            best = (score, a, b, c)
    _score, a, b, c = best
    return c * inv_logit(a + b * x_target) / 100.0


# --- the forecast -----------------------------------------------------------

def distinct_states(case: dict):
    """Collapse the history to distinct reads: Exophase's page sits unchanged
    between refreshes, so consecutive identical reads carry no new information.
    Returns [(time, players_tracked, rates tuple)] at first-appearance times."""
    states = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev:
            continue
        prev = key
        states.append((parse_time(read["at"]), read["players_tracked"], tuple(read["rates"])))
    return states


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    release = parse_time(case["game"].get("release_date", "2026-11-19"))
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def x_of(t: datetime) -> float:
        hours = (t - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    states = distinct_states(case)
    times = [s[0] for s in states]
    last_time = times[-1]
    h_last = max(0.0, (last_time - release).total_seconds() / 3600.0)

    # Refresh gap from the gaps between distinct reads; prior 4 h.
    gaps = [(times[i + 1] - times[i]).total_seconds() / 3600.0
            for i in range(len(times) - 1)]
    gap_h = sorted(gaps)[len(gaps) // 2] if gaps else PRIOR_REFRESH_GAP_H
    gap_h = min(max(gap_h, 0.5), 12.0)
    # The answering read shows the page as of the last refresh before it:
    # forecast half a refresh interval early.
    lag_h = min(0.5 * gap_h, MAX_LAG_H)
    eff_target = datetime.fromtimestamp(target.timestamp() - lag_h * 3600, tz=timezone.utc)
    x_target = x_of(eff_target)
    x_last = x_of(last_time)

    # --- tracked players P(t): log-linear in x, slope shrunk to the prior. ---
    xs = [x_of(t) for t in times]
    # Recency weights: a read a day old counts half.
    ws = [0.5 ** ((last_time - t).total_seconds() / 86400.0) for t in times]
    log_p = [math.log(max(1, p)) for _, p, _ in states]
    if len(set(log_p)) >= 2:
        a_p, b_p, s_p, xbar_p, sxx_p, n_p = ols(xs, log_p, ws)
        b_p = (n_p * b_p + PRIOR_P_WEIGHT * PRIOR_P_SLOPE) / (n_p + PRIOR_P_WEIGHT)
        a_p = sum(w * (y - b_p * x) for w, x, y in zip(ws, xs, log_p)) / sum(ws)
    else:
        # Flat page: anchor the prior slope on the first read, not the last
        # (the last read is stale by up to a refresh interval).
        a_p = log_p[0] - PRIOR_P_SLOPE * xs[0]
        b_p = PRIOR_P_SLOPE
        s_p, xbar_p, sxx_p, n_p = 0.0, xs[0], 0.0, 1
    log_p_target = a_p + b_p * x_target
    p_target = math.exp(log_p_target)
    var_p = pred_interval_var(s_p, n_p, xbar_p, sxx_p, x_target)

    # Refresh-jump uncertainty: one refresh can move the denominator by the
    # growth over a gap, evaluated at the target's time scale.
    dx_gap = math.log1p(max(0.0, (eff_target - release).total_seconds() / 3600.0 + gap_h)) \
        - math.log1p(max(0.0, (eff_target - release).total_seconds() / 3600.0))
    sigma_jump = 0.5 * abs(b_p) * dx_gap
    sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))

    # How far the slope prior has matured: 0 in the first 6 h, 1 past 30 h.
    maturity = min(1.0, max(0.0, (h_last - 6.0) / 24.0))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        cls = classify(trophy)
        early, mature, ceiling = CLASS_PRIORS[cls]
        # The opening trophy: every tracked player holds it, so its earners
        # grow exactly with the tracked population.
        if cls == "first" or (states and states[0][2][i] is not None and states[0][2][i] >= 99.99):
            cls = "first"
            early, mature, ceiling = CLASS_PRIORS["first"]
            b0 = b_p
        else:
            b0 = early + maturity * (mature - early)

        # Earned counts per distinct read (rates are 2-decimal, so counts are
        # approximate; 0.5 stands in for zero in the log fit).
        pts = []
        for (t, p, rates) in states:
            r = rates[i]
            if r is None:
                continue
            w = 0.5 ** ((last_time - t).total_seconds() / 86400.0)
            pts.append((x_of(t), math.log(max(0.5, round((r / 100.0) * p))), w))
        if len(pts) >= 2 and len({y for _, y, _ in pts}) >= 2:
            fxs = [p[0] for p in pts]
            fys = [p[1] for p in pts]
            fws = [p[2] for p in pts]
            a_n, b_n, s_n, xbar_n, sxx_n, n_n = ols(fxs, fys, fws)
            b_n = (n_n * b_n + PRIOR_SLOPE_WEIGHT * b0) / (n_n + PRIOR_SLOPE_WEIGHT)
            a_n = sum(w * (y - b_n * x) for w, x, y in zip(fws, fxs, fys)) / sum(fws)
            var_n = pred_interval_var(s_n, n_n, xbar_n, sxx_n, x_target)
        elif pts:
            # One distinct value: the prior slope, anchored on that read.
            x0, y0, _w = pts[0]
            a_n = y0 - b0 * x0
            b_n = b0
            var_n = SIGMA_FLAT_HISTORY ** 2
        else:
            # Never seen earned: the prior alone, from a tiny base.
            a_n = math.log(0.5) - b0 * x_last
            b_n = b0
            var_n = SIGMA_FLAT_HISTORY ** 2

        log_n_target = a_n + b_n * x_target
        median = 100.0 * math.exp(log_n_target) / p_target
        # Saturation guard: the earners-over-players trend extrapolates the
        # recent slope in a straight line, but a rate's growth decelerates as
        # the fast players finish and the slow ones remain.  The saturating
        # fit reads that deceleration from the series itself; take whichever
        # of the two is lower, so the forecast reflects the stronger of the
        # two forces pulling a rate down (saturation, or the denominator
        # growing faster than the earners).
        r_sat = saturating_rate_fit(
            [x_of(t) for (t, _, _) in states],
            [rates[i] for (_, _, rates) in states],
            [0.5 ** ((last_time - t).total_seconds() / 86400.0) for (t, _, _) in states],
            x_target, ceiling * 100.0)
        if r_sat is not None:
            median = min(median, r_sat)
        median = min(median, ceiling * 100.0)
        median = clamp_pct(median)

        if cls == "first":
            # Everyone tracked holds it; the only risk is a tracked player
            # whose first trophy is a different one.
            median = 99.9
            sigma = max(math.sqrt(var_n + var_p + sigma_rw ** 2
                                  + sigma_jump ** 2 + SIGMA_MISSPEC ** 2), 0.7)
        else:
            sigma = math.sqrt(var_n + var_p + sigma_rw ** 2
                              + sigma_jump ** 2 + SIGMA_MISSPEC ** 2)
        sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)

        lg = logit(median)
        qs = [round4(inv_logit(lg + z * sigma)) for z in Z]
        qs = [min(max(q, 0.01), 100.0) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
