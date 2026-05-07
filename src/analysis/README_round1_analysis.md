# Round 1 Research Pack

This analysis script builds a strategy research dataset and plot pack from both:
- `data/ROUND_1/prices_round_1_day_*.csv`
- `data/ROUND_1/trades_round_1_day_*.csv`

It computes market microstructure features (spread, imbalance, microprice gap, OFI proxy), trade-flow features (signed quantity/notional), forward returns, and reversal diagnostics.

## Run

```powershell
py -3 -u -m prosperity4.analysis.round1_research
```

## Outputs

Generated under `prosperity4/analysis/round1_outputs/`:
- `coverage_checks.csv`
- `liquidity_summary.csv`
- `signal_scorecard.csv`
- `autocorr_mid_diff.csv`
- `reversal_prob_by_move_bucket.csv`
- `01_...html` to `13_...html` interactive plots

## Notes

- Forward-return features are explicitly shifted to avoid look-ahead leakage.
- Trades are joined to prices by `(day, timestamp, product)` for side inference.
- Missing one-sided books are handled by falling back to provided `mid_price`.

