import numpy as np
import matplotlib.pyplot as plt

def research(x: int) -> int:
    """returns the PnL produced by a research investment of x% of 50K XIRECS."""
    if x > 100:
        return -1
    else:
        pnl = 200_000 * np.log(1 + x) / np.log(1 + 100)
        return pnl

def scale(x: int) -> float:
    """returns the PnL's multiplier produced by a scale investment of x% of 50K XIRECS. """
    if x > 100:
        return -1
    else:
        return x * 0.7

pnl_all = []
# 0 -> research(0), scale(100)   100 -> research(100), scale(0)
for i in range(101):
    pnl_all.append(research(i) * scale(100 - i))
#plot pnl_all as y and [0, 100] as x
pnl_all = np.array(pnl_all)
print(np.max(pnl_all), np.where(pnl_all == np.max(pnl_all)))
x = np.arange(0, 101)
plt.plot(x, pnl_all)
plt.xlabel('Research Investment (%) - remaining goes to Scale')
plt.ylabel('PnL')
plt.title('PnL vs Research Investment')
plt.grid()
plt.show()

