# Methodology: forecasting GTA VI trophy earn rates at 2026-11-26T00:00:00Z

## What is being forecast

For each of the 55 PlayStation 5 trophies of Grand Theft Auto VI, the earn rate
on Exophase's trophy page
(`https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/`): the percent
of the players Exophase tracks for the game who have earned that trophy, as
five quantiles (5%, 25%, 50%, 75%, 95%), at 2026-11-26T00:00:00Z — a week
after release. Scored by mean pinball loss on the logit scale against the
page's rates at the target.

## The model

An earn rate is a ratio, `rate = N / P`:

- `P` — the players Exophase tracks. Exophase tracks a player once they have
  earned at least one trophy and synced a profile, so `P` grows as more people
  buy the game, play it and sync.
- `N_i` — how many tracked players hold trophy `i`. It grows as players
  progress through the game.

The two move for different reasons, so they are forecast separately and
divided. Both are fit as straight lines in `x = ln(1 + hours since release)`
against log counts, over the case's **distinct** reads (consecutive reads with
identical values are one read: Exophase refreshes its counts hours apart, so
the page sits flat between refreshes). Each fit is recency-weighted
(half-life 24 h) so the slope it finds is the slope the series has now.

### Tracked players

`log P` is fit by least squares in `x`, its slope shrunk toward a prior of
1.1. The prior comes from the launch-night refresh the competition's rules
document: 638 → 3,868 tracked players over ~4.5 hours is a slope of ~1.06 in
this space. With a flat history the prior is anchored on the *first* read, not
the last — the last read is stale by up to a refresh interval.

### Earners, per trophy

`log N_i` is fit the same way, its slope shrunk toward a prior that depends
on what the trophy asks for and on how far into the launch the case sits.
Trophies are classified from their name, description and tier: story chapters
1–4, the final mission, side missions and side strands, easy one-offs
(arcade game, cocktail, bowling), mid-effort feats (robberies, spending),
grinds (250 stunts, 100 locations, 3 card sets, gold medals), exploration,
100% completion and the Platinum.

The priors are small in the first hours and mature later. Launch night shows
why: between the first probe and the first refresh (~4.5 h) the non-first
trophies grew only 305 → 408 earned while tracked players grew 638 → 3,868 —
an average log-earners slope of ~0.17. In the first hours almost nobody has
had time to earn anything past the opening mission. By day two the slopes are
mature (story chapters ~1.4–2.0 in log-earners against `x`).

### Saturation

A rate's growth decelerates: as the fast players finish a chapter, the
remaining players are slower and the rate bends toward a ceiling. So a second
fit is made on the rate series itself — `rate = C · sigmoid(a + b·x)` with `C`
grid-searched and a prior pulling it toward the trophy class's ceiling — and
the forecast takes the **lower** of the trend extrapolation (`N̂/P̂`) and the
saturating fit. The lower of the two reflects the stronger of the two forces
pulling a rate down: saturation, or the denominator growing faster than the
earners.

### The refresh lag

The answer is the page at the target *as of Exophase's last refresh before
it*, so the forecast is evaluated half a refresh interval before the target
(the interval estimated from the gaps between distinct reads in the case,
prior 4 h, lag capped at 2 h). At a week out this moves the forecast by ~1%;
on the 90-minute practice cases it matters more, and it is the honest median
there too.

### Ceilings

The median is capped by a per-class day-7 ceiling — the share of tracked
players who could plausibly hold the trophy by day 7 at all: 80% for
Chapter 1, 65% for Chapter 2, 50% for Chapter 3, 40% for Chapter 4, 30% for
the final mission, 45% for side missions, 60% for easy one-offs, 15% for
grinds, 6% for the hardest grinds, 3% for 100% completion, 2% for the
Platinum. The opening trophy ("Just Like Paradise", held by 100% of tracked
players at launch) is forecast at 99.9%: every tracked player holds it, and
the only risk is a tracked player whose first trophy is a different one.

### Spread

Each trophy's logit sigma combines: the two fits' prediction intervals (wide
when the history is flat — a flat page says nothing about the next refresh), a
random-walk floor growing with the horizon (`0.30 · √days`), a term for the
refresh jump (one refresh can move the denominator a lot in the first days),
and a flat 0.25 for model misspecification. Sigma is clamped to [0.30, 2.0].
Quantiles are the median shifted by the normal quantiles (−1.645, −0.674, 0,
0.674, 1.645) times sigma, on the logit scale, rounded to 4 decimals.

## Why this beats persistence

The house baseline says every rate stays where the last read put it. That
fails twice over: first, every Exophase refresh dilutes the rates as thousands
of new players arrive who have earned almost nothing yet (launch night: the
first refresh took tracked players 638 → 3,868 and every rate but the 100%
one fell); second, over a week the story trophies climb steeply as players
finish chapters. Fitting the two sides of the ratio separately catches both.

## Local validation

No final case exists before the deadline, so the model was tested on two
synthetic final-like cases built from the practice data (120 reads, 16
distinct refreshes, checkpoint 2026-11-21T11:30Z, target 2026-11-26T00:00Z):

| synthetic world | this model (loss / typical miss / 90% coverage) | house baseline |
|---|---|---|
| A — rates keep rising on a logit-linear path | 0.184 / 0.47 pp / 0.98 | 0.288 / 2.72 pp / 0.98 |
| B — rates saturate and players surge over Black Friday weekend | 0.219 / 1.97 pp / 0.98 | 0.225 / 0.49 pp / 0.98 |

The saturation guard is what keeps world B competitive: without it the model
scored 0.250 there, losing to the baseline. The practice cases (flat launch
night, 90-minute horizon) validate the refresh handling: the model's medians
imply ~640 non-first trophies earned at the 04:30 refresh against the 408
actually recorded — high, because the honest median assumes half a refresh
interval of staleness while that particular refresh happened to land right at
the target; the practice pays nothing and the lag is worth ~1% on the final.

## Sources and what is speculation

- **Trophy list and launch-night rates** (55 trophies, 638 tracked players,
  the 04:30 refresh to 3,868): the competition kit's own case files and rules.
  These are observed data, not estimates.
- **GTA V long-run trophy rates** (PocketPSN: final mission ~7.4%, 100%
  completion ~0.2%, Platinum ~0.1%, first mission ~38.6%): used only to set
  the *shape* of the ceilings and priors. They are long-run rates on a
  self-selected tracker population, not week-one rates — the ceilings are
  deliberately set above the GTA V long-run values for early story trophies
  and below them for the Platinum, because a launch week's tracked players are
  far more engaged than a 13-year population.
- **GTA VI campaign ~50 hours, five chapters, ~100 missions; Platinum ~200+
  hours** (Polygon / Insider Gaming, citing people who played the game): the
  reason story trophies are expected to rise steeply through the week while
  the Platinum and 100% completion barely move. Unofficial; Rockstar has not
  confirmed it.
- **Launch-week sales** (Compound base case ~14M copies in week one;
  Sensor Tower ~24.6M digital pre-orders by Nov 18; Newzoo $3.25–5.2B
  first-week revenue; ~74% PlayStation share of pre-orders): used only as a
  sanity check that tracked players in the hundreds of thousands by day 7 is
  plausible. Analyst estimates, not official figures.
- **Day-7 rate levels in the synthetic tests** (e.g. Chapter 1 ~40%, final
  mission ~3%, Platinum ~0.1%): judgement calls built from the sources above,
  not measurements. The model's forecasts do not hard-code them — they enter
  only through the class priors and ceilings, and the fits are driven by the
  case's own history wherever it has signal.
- **The trophy list itself is reported, not official**: GamesRadar+ / PC
  Gamer print 55 trophies; Rockstar has published no list. The case file is
  authoritative for the entry.
