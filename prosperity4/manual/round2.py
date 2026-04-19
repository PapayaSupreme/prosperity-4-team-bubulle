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

def max_with_speed(x: int) -> int:
    """returns the maximum value obtainable with x% of 50K XIRECS allocated to speed"""
    pnls = np.zeros(101)
    # 0 -> research(0), scale(100)   100 -> research(100), scale(0)
    for y in range(101 - x):
        pnls[y] = round(round(research(y)) * scale(100 - x - y) * 0.1)
    if np.max(pnls) <= 70000:
        return 0
    return np.max(pnls)

speed_mult = 0.2
speed_value = 4
pnl_all = np.zeros(101)
# 0 -> research(0), scale(100)   100 -> research(100), scale(0)
for i in range(101 - speed_value):
    pnl_all[i] = int(int(research(i)) * scale(100 - speed_value - i) * speed_mult)
#plot pnl_all as y and [0, 100] as x
#print(np.max(pnl_all), np.where(pnl_all == np.max(pnl_all)))
"""for i in range(101):
    pnl_all[i] = max_with_speed(i)
plt.xlabel('Speed Investment (%)')
plt.ylabel('Max PnL combination of Research & Scale')
plt.title('PnL vs Speed Investment')"""
x_abs = np.arange(0, 101)
plt.plot(x_abs, pnl_all)
plt.xlabel('Speed Investment (%)')
plt.ylabel('Max PnL combination of Research & Scale')
plt.title('PnL vs Speed Investment')
plt.grid()
plt.show()
print(np.max(pnl_all), np.where(pnl_all == np.max(pnl_all)))

# 100: 23 * 77 : 74 233
# 99: 23 * 76 : 73 269
# 98: 23 * 75 : 72 305
# 97: 23 * 74 : 71 341
# 96: 22 * 74 : 70 385
