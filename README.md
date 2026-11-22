# Token Launch Radar — radar-5

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/AllureCurtain/token-radar-01/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/AllureCurtain/token-radar-01/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

Which pre-token protocols would be worth the most if they launched today — fitted from the market, not from names.

## What it does

The program reads the sealed DefiLlama snapshot (`input/snapshot.json`, 758 protocols as of 2026-11-14), fits a value model on the protocols whose token already trades, and ranks the protocols that record no token by fitted implied value. The dashboard shows the evidence on its first screen: a scatter of implied value against locked value with the fitted curve, a table of every value term with its fitted coefficient and held-out error, and the best farms ranked by net expected value per hands-on hour. Every row explains why it ranks where it does, and every protocol has a detail view with its rank interval, farm plan and sensitivity table.

## The value model

Log market cap is regressed on log TVL, seven-day growth measured against the traded median (clamped at ±30 points), and — where they earn their place — log age, category shifts and chain shifts. Every term is fitted separately and tested held-out; blocks that do not improve held-out error are dropped and marked as dropped in the term table. On this snapshot the load-bearing term is log TVL; growth is kept on the time-split evidence, while the curvature, age, category and chain blocks are dropped.

Validation, all reported on the dashboard:

- Time split: fit on older listings, test on newer listings.
- 5-fold cross-validation across protocols, with the spread across folds.
- Rolling walk-forward: refit as listing dates advance, test the next window.
- Rank intervals from 50 bootstrap refits.

No protocol name is used anywhere in the model. Implied values use a smearing correction for log-normal errors.

## Farm plans

Every protocol gets a concrete plan: steps specific to its category and chain, each with a dollar cost (network fees on its chain plus capital) and a time. The verdict at $1,000 and $10,000 stakes (worth it / thin / skip) comes from a stated assumption: a launch distributes 6% of implied value to farmers, the farmer count grows with TVL (crowding), and points scale as stake^0.7 against a $2,500 median farmer. A sensitivity table shows the verdict if the token lists 5, 10 or 20% below implied value.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input input/snapshot.json
```

Node 22 or later, standard library only. The program writes `out/board.json`, `out/dashboard/` (the page, its stylesheet, its script and its data) and `out/dossier.json`, and copies `screenshots/` into the output.

## Layout

- `main.ts` — entrypoint: reads the snapshot, fits, ranks, writes the outputs.
- `lib/model.ts` — the fit, the validation, the rank intervals and the farm plans.
- `dashboard/` — `index.html`, `app.css`, `app.js` and the generated `data.js`.
- `vendor/makefun-board.ts` — the scorer's own dossier library, copied byte for byte as shipped.
- `screenshots/` — the three dashboard views, taken with a headless browser from the real dashboard running on the real board data.

## License

MIT. See `LICENSE`.
