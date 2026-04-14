# Prosperity 4 Playbook (Using this Prosperity 3 repo as a template)

## 1) What this repository is for

This repository is a proven tournament workflow from Prosperity 3. You can reuse the same architecture for Prosperity 4:

- Keep algorithm source code separate from snapshots you submit.
- Keep analysis notebooks separate from production trading code.
- Archive logs and artifacts after each submission.
- Keep one clean entrypoint (`Trader.run`) that the platform calls.

The key idea is that this repository is not only "a strategy file"; it is a full iteration system:
research -> code -> backtest -> submit -> inspect logs -> repeat.

## 2) Folder-by-folder structure

Current structure in this repo:

- `logs/`: archived logs from submissions and final round results.
- `prosperity3/algorithms/`: active strategy development files (`empty.py`, `hybrid.py`).
- `prosperity3/analysis/`: notebooks and helper code for data exploration (`data.py`).
- `prosperity3/manual/`: notebooks for manual rounds.
- `prosperity3/submissions/`: frozen copies of each round's submitted code.

How to mirror this for Prosperity 4:

- Rename package to `prosperity4/`.
- Keep these subfolders: `algorithms/`, `analysis/`, `manual/`, `submissions/`.
- Keep `logs/` at repo root for simulator/submission logs.

Suggested Prosperity 4 tree:

```
.
├── logs/
├── prosperity4/
│   ├── algorithms/
│   │   ├── empty.py
│   │   └── hybrid.py
│   ├── analysis/
│   │   ├── data.py
│   │   └── round1.ipynb
│   ├── manual/
│   └── submissions/
├── pyproject.toml
└── README.md
```

## 3) The core trading contract you must implement

Your runtime entrypoint is a `Trader` class with:

- Method: `run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]`
- Return values:
  - Orders by symbol
  - Number of conversions
  - Serialized `traderData` string for state persistence

In this repository:

- The minimal template is in `prosperity3/algorithms/empty.py`.
- A production multi-strategy implementation is in `prosperity3/algorithms/hybrid.py`.

Prosperity 4 compatibility rule used in this port:

- Keep `run(state)` as the canonical website entrypoint.
- Add `trade(state)` as an alias only for local tools that still call `trade`.

## 4) How `hybrid.py` is organized (and why it scales)

`prosperity3/algorithms/hybrid.py` uses composable strategy classes:

- `Strategy`: base class for one symbol.
- `StatefulStrategy`: strategy with save/load persistence support.
- `SignalStrategy`, `MarketMakingStrategy`: reusable behavior families.
- Product strategies (example): `RainforestResinStrategy`, `PicnicBasket1Strategy`, `VolcanicRockStrategy`.
- `Trader`: orchestration layer that runs all strategies and aggregates orders.

Scalable pattern to copy into Prosperity 4:

1. Keep shared helpers in base classes.
2. Keep per-product logic in small strategy classes.
3. Keep `Trader` mostly declarative (symbol -> strategy map + limits).
4. Serialize state (`traderData`) only for strategies that need history.

## 5) Data analysis and backtesting workflow

In this repo, analysis support lives in `prosperity3/analysis/data.py`:

- It loads historical days and returns a DataFrame for notebook analysis.

For Prosperity 4 tutorial data in this port:

- Data path: `data/tutorial_round/`
- Price files: `prices_round_0_day_*.csv`
- Trade files: `trades_round_0_day_*.csv`
- Current tutorial symbols observed: `EMERALDS`, `TOMATOES`

Typical loop:

1. Inspect historical prices/trades in notebooks.
2. Design a strategy hypothesis.
3. Implement in `algorithms/hybrid.py` (or a dedicated file).
4. Backtest locally.
5. Promote candidate code into `submissions/roundX.py`.
6. Submit and archive resulting logs in `logs/`.

## 6) Interaction model with the challenge website

The exact Prosperity 4 website UI can change each year, but the stable workflow is:

1. Build a valid `Trader` file compatible with the current year's datamodel.
2. Submit source code through the official submission page (or CLI if available for that year).
3. Wait for score/log output.
4. Download/store logs and compare with previous submissions.
5. Iterate quickly with small controlled changes.

Practical advice:

- Track each submission with a version label (`roundX-vY`).
- Save both "submission log" and "final log" when available.
- Keep a simple changelog entry per submission: what changed and expected impact.

## 7) How to migrate this repo to Prosperity 4 quickly

Checklist:

1. Create a new package folder `prosperity4/` by copying `prosperity3/`.
2. Replace all Prosperity 3-specific imports/datamodel classes with Prosperity 4 equivalents.
3. Start from `algorithms/empty.py`; confirm a "no-op" strategy runs end-to-end.
4. Port your strategy framework (`Strategy`, `StatefulStrategy`, logger/truncation logic).
5. Re-add strategies one by one, validating each with backtests.
6. Keep each round submission snapshot under `prosperity4/submissions/`.

Additional migration guardrails used here:

- Use dual import strategy for datamodel (`datamodel` first, local fallback second).
- Keep the uploaded file standalone-friendly in `submissions/`.
- Validate with tutorial CSVs before first website submission.

High-risk migration points:

- Changed field names/types in `TradingState` or observations.
- New products and different position limits.
- Changed conversion mechanics.
- Different log size/truncation constraints.

## 8) Suggested operating routine for your own algorithms

Use this lightweight cadence:

- Daily research: notebook checks for spread/volatility/regime behavior.
- Implementation: one or two isolated strategy changes.
- Validation: local backtest vs prior baseline.
- Submission: only if change has clear expected edge.
- Postmortem: annotate logs and preserve the exact submitted file.

This keeps your process reproducible and reduces "mystery alpha" that cannot be re-created.

## 9) First-week Prosperity 4 action plan

- Day 1: Set up repo skeleton + no-op trader + first successful submission.
- Day 2: Build data readers/notebooks and baseline metrics.
- Day 3: Implement one market-making strategy.
- Day 4: Add one signal strategy with persistent state.
- Day 5: Build submission checklist and archive discipline.
- Day 6-7: Tune thresholds, reduce risk, and simplify fragile logic.

If you follow this structure, you can move fast while keeping your trading stack organized and auditable.

