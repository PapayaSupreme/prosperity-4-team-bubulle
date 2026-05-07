# Round 5 Comprehensive Product Analysis

## Overview

`round5_analysis.py` provides a comprehensive analysis of all ~50 tradable goods discovered in Round 5 data. The script:

1. **Discovers all unique products** from price and trade CSVs (days 2-4)
2. **Analyzes market microstructure** for each product:
   - Price statistics (mean, std, min, max, range)
   - Bid-ask spread analysis
   - Depth and imbalance metrics
   - Trading activity and volumes
   - Price volatility (returns)

3. **Produces multi-perspective ergonomic output**:
   - Top 20 most-traded products
   - Top 15 most-liquid products (by depth)
   - Top 15 most-volatile products
   - Sorted list by anchor price level
   - Overall market statistics
   - Full CSV export for external analysis

## Usage

```bash
cd prosperity-4-team-bubulle
python prosperity4/analysis/round5_analysis.py
```

## Output

### Console Output

The script prints organized tables to stdout showing:

- **Most Traded Products**: Ranked by trade count, with price stats and volatility
- **Most Liquid Products**: Ranked by average bid depth, useful for identifying symbols with tight spreads
- **Most Volatile Products**: Ranked by price standard deviation, useful for identifying range-trading opportunities
- **By Anchor Price**: All products sorted by mid-price level
- **Overall Statistics**: Market-wide averages

### CSV Export

A full analysis is exported to `prosperity4/analysis/round5_analysis_output.csv` with all metrics
for all products, enabling further filtering and sorting in Excel or other tools.

## Key Metrics

For each product:

| Metric | Description |
|--------|-------------|
| **mid_mean** | Average mid-price across all observations |
| **mid_std** | Price volatility (standard deviation) |
| **mid_range** | Max price - min price |
| **spread_mean** | Average bid-ask spread |
| **depth_avg_bid** | Average bid-side market depth (volume) |
| **depth_avg_ask** | Average ask-side market depth (volume) |
| **imbalance_mean** | Average order book imbalance (-1 to 1); positive = buy pressure |
| **returns_std** | Volatility of mid-price returns (log changes) |
| **num_trades** | Total number of trades |
| **total_volume** | Total quantity traded |
| **avg_trade_quantity** | Average trade size |

## Example Interpretation

**High `num_trades` + Low `spread_mean`**: "Efficient" liquid market
→ Good for market-making strategies

**High `mid_std` + Low `num_trades`**: "Thin" volatile market
→ Good for mean-reversion strategies, but beware of fills

**High `depth_avg_bid/ask` + High `spread_mean`**: "Thick" but wide market
→ Suggests infrequent trades or informed flow

## Customization

The script can be extended with:
- Correlation analysis between products
- Time-of-day volatility patterns
- Order flow toxicity metrics
- Individual trader flow analysis (from trades CSV)

To do so, edit the analysis functions in `round5_analysis.py` and re-run.

