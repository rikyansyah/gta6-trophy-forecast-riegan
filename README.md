# gta6-trophy-forecast-riegan

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rikyansyah/gta6-trophy-forecast-riegan/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rikyansyah/gta6-trophy-forecast-riegan/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of Exophase-tracked players who will have earned it one week after launch (2026-11-26T00:00:00Z), as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), from the house's recording of the game's Exophase trophy page up to the deadline.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=vrsa6rr4&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## How it works

`main.py` (Python 3.13, standard library only) reads one case file at a
time from `/input` (or `$MAKEFUN_INPUT_DIR`) and writes
`/output/<id>.out`: one row per trophy, five quantiles of its earn rate
at the case's target. It is deterministic and runs in ~0.1 s per case.

An earn rate is a ratio — earners over players tracked — and the two
sides move for different reasons during a launch week, so the model
forecasts them separately and blends three views on the logit scale:

- **Trend (weight 0.65).** A weighted least-squares line of the trophy's
  logit earn rate against ln(1 + hours since release) over the distinct
  page states of the last 48 hours (Exophase refreshes its counts in
  steps, so repeated identical reads are one state), its slope shrunk
  toward its tier's median slope and damped to 0.8, read at the target.
  Earn rates move almost linearly on this scale over a launch week, and
  the recent slope is the strongest single signal.
- **Structural (weight 0.20).** Earners and players tracked are each
  extrapolated by their own log-log fits (earner slopes shrunk toward
  tier medians; the players fit blended with a damped-slope variant),
  and the median share is earners(target) / players(target). This is
  what keeps the forecast honest about dilution: a fast-growing tracked
  population is a young population, and young players hold the mid- and
  late-game trophies at lower rates.
- **Persistence (weight 0.15).** The last read's rate, as an anchor.

The blended move away from the last rate is damped once more (×0.88).
The spread around the median grows with the square root of the horizon
(σ = 0.45·√days on the logit scale), widens when the structural and
trend views disagree, and widens for trophies held by only a handful of
players, whose rates are mostly rounding noise. A trophy everyone
already holds is capped just under 100%: it can only dilute.

A history that shows a single frozen page — the practice cases, cut
before Exophase's first refresh — cannot fit any trajectory. The model
then expects the refresh it has not seen: shares shift down by a
per-tier dilution factor (the real launch-night refresh took the
tracked count from 638 to 3,868 and every rate but one fell), with a
wide spread.

## Local validation

There is no local grade for this competition (the server grades every
entry), so the methodology was validated on synthetic final cases with
known ground truth, built three ways: cohort-arrival worlds (players
arrive on a growth curve and earn each trophy after a per-trophy delay,
so dilution emerges from the mechanics), smooth-share worlds, and
cohort worlds whose arrival growth surges or slumps after the
checkpoint. Scored with the contest's own metric (mean pinball loss on
the logit scale), against the two house baselines:

| world | this model | persistence | curve baseline |
|---|---|---|---|
| cohort (8 scenarios) | **0.247** | 0.507 | 0.279 |
| smooth-share (6 scenarios) | **0.285** | 0.301 | 0.319 |
| cohort, post-checkpoint surge | **0.275** | — | 0.344 |
| cohort, post-checkpoint slump | 0.308 | — | **0.244** |

The blend weights and spread were chosen for robustness across these
worlds rather than for the best score in any one of them.
