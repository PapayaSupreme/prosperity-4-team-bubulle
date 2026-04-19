import numpy as np
import matplotlib.pyplot as plt

def research(x: int) -> int:
    """returns the PnL produced by a research investment of x% of 50K XIRECS."""
    if x > 100:
        return -1
    else:
        pnl = 200000 * np.log(1 + x) / np.log(1 + 100)
        return pnl

def scale(x: int) -> float:
    """returns the PnL's multiplier produced by a scale investment of x% of 50K XIRECS. """
    if x > 100:
        return -1
    else:
        return x * 0.07

def speed(x: int) -> float:
    """Returns the PnL's multiplier produced by a speed investment of x% of 50K XIRECS."""
    if x > 100:
        return -1
    return 0.1 + 0.008 * x

def max_with_speed(x: int, details: bool) -> int:
    """returns the maximum value obtainable with x% of 50K XIRECS allocated to speed"""
    pnls = np.zeros(101)
    speed_value = speed(x)
    # 0 -> research(0), scale(100)   100 -> research(100), scale(0)
    for y in range(101 - x):
        pnls[y] = round(research(y) * scale(100 - x - y) * speed_value)
    if details:
        print("Research % for best PnL: ",  np.where(pnls == np.max(pnls)))
        print("Speed % for best PnL: ", x)
        print("Scale % for best PnL: 100 - Research - Speed")
    return np.max(pnls)

pnl_all = np.zeros(101)
for speed_iter in range(101):
    pnl_all[speed_iter] = max_with_speed(speed_iter, False)

x_abs = np.arange(0, 101)
plt.plot(x_abs, pnl_all)
plt.xlabel('Speed Investment (%)')
plt.ylabel('Max PnL combination of Research & Scale')
plt.title('PnL vs Speed Investment')
plt.grid()
plt.show()

print("Best PnL: ", np.max(pnl_all),  " at index: ",  np.where(pnl_all == np.max(pnl_all)))
#best is at index 36
print("Combination for best Pnl: ", max_with_speed(36, True))
# Best: 16, 48, 36 -> 160 065

