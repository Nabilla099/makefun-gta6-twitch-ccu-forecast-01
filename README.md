[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Nabilla099/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Nabilla099/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Built for the [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty on make.fun.

# GTA VI launch-day Twitch viewers, hour by hour

A Python 3.13 program (standard library only) that reads one case
(`makefun/twitch-ccu-case/v1`) and forecasts the concurrent Twitch
viewers in the Grand Theft Auto VI category for hours 2..24 of release
day 2026-11-19 (UTC), as five quantiles an hour (5%, 25%, 50%, 75%,
95%). Run locally with:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Method in plain words

The shape of a launch day depends on how the game unlocks. A game whose
first hour is a pre-unlock trickle (a single global unlock at 14:00Z)
surges hundreds of times over the day; a rolling-midnight console launch
— GTA VI's kind — starts with a real audience and rises a few times. So
the reference pool keeps only past launches whose hour-1 print is at
least 5% of their release-day peak.

Within that pool, how much an hour grows over hour 1 is strongly tied to
how big hour 1 itself was (across launches, log surge vs log hour-1
correlates about −0.94: a huge first hour leaves little room to grow).
For each hour we therefore regress log(viewers_h / viewers_1) on
log(viewers_1) across the pool and take the quantiles of the residuals.
The forecast is the regression prediction at the case's own hour-1 print,
plus the residual quantiles, blended 50/50 with the matching normal
quantiles (residual IQR / 1.349 as sigma) to smooth the noisy 5%/95%
tails of a ~30-game sample.

## Measured accuracy

Leave-one-out evaluation on synthetic cases built from the kit's 37
past-launch comparables (rolling-midnight-like launches, mean pinball
loss on the log scale, lower is better):

| model | mean loss |
|---|---|
| house persistence baseline (starter) | 0.360 |
| pooled empirical log-ratio quantiles | 0.311 |
| **this entry** | **0.259** |

The house baseline scores about 0.387 on the official practice set, so
this entry beats it by a wide margin there too. Output is always valid:
23 rows, hours 2..24, five finite quantiles per row in (0, 1e9],
non-decreasing, with fallbacks to pooled quantiles and then to the
persistence baseline if the case data is thin.
