# Prosperity 4 Structure (Ported Template)

This folder mirrors the previous year's proven layout and is ready for the Prosperity 4 tutorial data workflow.

## Folder layout

- `algorithms/`: active strategy development (`empty.py`, `hybrid.py`)
- `analysis/`: data readers and notebooks for research
- `manual/`: manual-round notes/notebooks
- `submissions/`: immutable snapshots of files actually uploaded

## Entrypoint compatibility

The website entrypoint should be `Trader.run(state)` returning:

- `dict[Symbol, list[Order]]` (orders by symbol)
- `int` (conversions)
- `str` (`traderData` payload)

In this port, both `empty.py` and `hybrid.py` implement:

- `run(state)` for website compatibility
- `trade(state)` as an alias for local tools that still call `trade`

## Datamodel import strategy

In `algorithms/empty.py` and `algorithms/hybrid.py`:

- First try `from datamodel import ...` (website runtime)
- Fallback to `from prosperity4.algorithms.datamodel import ...` (local testing)

This lets one codebase work in both environments.

## Using tutorial data

`analysis/data.py` reads files from `data/tutorial_round/`:

- `read_tutorial_prices(day)`
- `read_tutorial_trades(day)`
- `read_all_tutorial_prices()`
- `read_all_tutorial_trades()`

## Snapshot workflow

When you submit, copy the exact uploaded code into:

- `submissions/round0_tutorial.py`
- then `submissions/round1.py`, `submissions/round2.py`, etc.

Treat `submissions/` as immutable history.

## Current starter assets

- Main development strategy: `prosperity4/algorithms/hybrid.py`
- Upload-ready single file snapshot: `prosperity4/submissions/round1.py`
- Tutorial notebook starter: `prosperity4/analysis/round0_tutorial.ipynb`
- Tutorial data helpers: `prosperity4/analysis/data.py`

## VISUALIZER ONCE UPLOADED

https://kevin-fu1.github.io/imc-prosperity-4-visualizer/
