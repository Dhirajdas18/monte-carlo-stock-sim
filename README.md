# Monte Carlo Stock Simulator (NSE)

![CI](https://github.com/Dhirajdas18/monte-carlo-stock-sim/actions/workflows/ci.yml/badge.svg)

A Monte Carlo risk engine for Indian equities. It simulates price paths for single stocks, correlated portfolios and trading-strategy returns. Before trusting any model, it checks the model's forecasts against real NSE history.

> Monte Carlo gives a range of outcomes under a model's assumptions. It is not a price forecast. That is why this project validates every model walk-forward before using it.

![Simulated price paths](docs/img/gbm-fan.png)

## Features

- [x] **Data pipeline with corporate-action checks.**
  - Daily prices for RELIANCE, HDFCBANK, INFY, TCS, ITC and NIFTY, 2013–2026, from Yahoo Finance or a local NSE bhavcopy lake.
  - True traded prices are rebuilt from split ratios.
  - Splits and bonuses are detected automatically, so they can't show up as fake crashes.
- [x] **Return and volatility analytics.** Simple vs log returns, annualised volatility, a test of the √time scaling rule on real data, and fat-tail diagnostics
- [x] **GBM path simulation.** Validated against closed-form results, with antithetic variance reduction and a drift-uncertainty analysis
- [x] **Risk measures.** VaR and CVaR (simulated vs historical), percentile bands, and the probability of touching a price vs ending beyond it
- [ ] Walk-forward coverage testing of each model's forecast bands
- [ ] Alternative models: Student-t, Merton jump-diffusion, historical bootstrap and block bootstrap
- [ ] Correlated portfolio simulation (Cholesky and joint-day bootstrap), with a check on how correlation changes in crashes
- [ ] Strategy risk: drawdown distributions, the probability that an edge is luck, and Kelly position sizing

## Findings so far

| Finding | Evidence |
|---|---|
| Unadjusted prices invent crashes | 9 one-day drops of 31–51% across the 5 stocks are splits and bonuses, not losses ([chart](docs/img/data-raw-vs-adjusted.png)) |
| One missed corporate action corrupts a whole history | A data source that missed ITC's 2016 1:2 bonus reports 0.9%/yr. The real figure is 6.9%/yr. The pipeline flags any adjusted one-day drop over 20% |
| Log returns are the right unit over time | Averaging daily simple returns says RELIANCE grew 18.8%/yr; it actually grew 15.9%/yr, which daily log returns reproduce exactly |
| Real returns have fat tails | Under a normal distribution a beyond-4σ day should happen ~0.2 times in 14 years; each stock had 16–22 ([chart](docs/img/returns-vs-normal.png)) |
| The √time rule holds short-term only | Scaling daily volatility by √days matches realised volatility to within ~5% up to 1 month, but overstates it by 5–30% at 3–12 months ([chart](docs/img/returns-sqrt-time.png)) |
| The historical mean return is not a forecast | RELIANCE's 13.6-year drift is 18.7%/yr ± 7.3% (1 SE). Plausible drifts move the median 1-year price from ₹1,327 to ₹1,851, so risk simulations use zero drift ([chart](docs/img/gbm-drift-uncertainty.png)) |
| Volatility drag | At zero drift, 55% of simulated paths end below today's price; the median sits ~3.6%/yr under the mean ([chart](docs/img/gbm-mean-vs-median.png)) |
| Antithetic sampling pays off | Pairing each draw with its mirror cuts the error of the simulated mean 3–5× at the same path count ([chart](docs/img/gbm-paths-needed.png)) |
| VaR hides the tail; CVaR doesn't | On ₹1 lakh of RELIANCE for 1 month, historical VaR 95% (₹10.1k) looks safer than the GBM model's (₹12.1k), yet the historical average loss in the worst 1% of months is ₹22.6k vs the model's ₹18.8k ([chart](docs/img/risk-var-cvar.png)) |
| Stops trigger on touches | A −10% level is touched within 3 months 41% of the time but is still breached at the end only 24% of the time ([chart](docs/img/risk-touch-vs-end.png)) |
| Same market, very different outcomes | ₹1 lakh invested in Jan 2013 became anywhere from ₹2.5L (ITC) to ₹7.8L (RELIANCE). NIFTY, excluding dividends, ended at ₹4.3L |

## Quick start

```bash
uv sync
uv run python scripts/fetch_data.py   # prices from Yahoo Finance into data/
uv run mc-report data                 # charts into out/, key numbers to stdout
uv run mc-report returns
uv run mc-report gbm
uv run mc-report risk
uv run pytest
```

All randomness is seeded, so every reported number can be reproduced.

## Design

- `src/mc/`: one module per concern (`data`, `returns`, `gbm`, `risk`, `plots`, `reports`, plus the simulation modules as they land).
- `mc-report <topic>` regenerates a topic's charts and numbers from raw data, so nothing in this README is typed by hand.
- Tests use synthetic data with known answers. They never depend on downloaded files. CI runs `ruff` and `pytest` on every push.
- No market data is committed. Set `MC_LAKE_DIR` to read a local NSE bhavcopy lake (`panel.parquet` and `market.parquet`) instead of Yahoo.

Stack: Python 3.12 · uv · NumPy · pandas · SciPy · Matplotlib · pytest · GitHub Actions.

MIT licence.
