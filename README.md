# Prosperity 4 Bubulle's team codebase

## INTRODUCTION

TOP 702 / 18 000
Contains my solo team attempt at Prosperity 4.
Pablo Ferreira, @2026
[LinkedIn](https://www.linkedin.com/in/pablo-frr)
[GitHub](https://www.github.com/PapayaSupreme)


## Folder layout

- `data/`: contains the price & trades history in .csv of each round & day
- `src/`
  - `algorithms/`: templates for strategy development (hedgehog is a prosperity 3 team)
  - `insider_trading/`: data crunchers using traders IDs to detect informed behavior, and log it.
  - `analysis/`: data readers and notebooks for research
  - `manual/`: manual-round notes/algorithms
  - `submissions/`: immutable snapshots of files actually uploaded/tested
- `utilities/`: scripts used on their own for a specific goal

## Entrypoint compatibility

The website entrypoint should be `Trader.run(state)` returning:

- `dict[Symbol, list[Order]]` (orders by symbol)
- `int` (conversions)
- `str` (`traderData` payload)

In this port, both `empty.py` and `hybrid.py` implement:

- `run(state)` for website compatibility
- `trade(state)` as an alias for local tools that still call `trade`
- 
## Snapshot workflow for round X, version Y, change Z

Every time you backtest, copy the exact uploaded code into:

- `/src/submissions/roundX/vX.YZ.py`

Once you submit, copy the exact uploaded code into:
- `/src/submissions/final/vX.Y.py`


Treat `/src/submissions/final/` as immutable history.

## LOGS VISUALIZER ONCE UPLOADED (not mine)

https://prosperity.equirag.com/
