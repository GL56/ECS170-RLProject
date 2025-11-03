import re
import numpy as np
import matplotlib.pyplot as plt

LOG_FILE = "raw_training_log.txt"

# --- parse "Episode   X/500 | return= Y ..." lines ---
episodes, returns = [], []
pat = re.compile(r"Episode\s+(\d+)\s*/\s*\d+\s*\|\s*return=\s*([-+]?\d+(?:\.\d+)?)")

with open(LOG_FILE, "r") as f:
    for line in f:
        m = pat.search(line)
        if m:
            episodes.append(int(m.group(1)))
            returns.append(float(m.group(2)))

if not episodes:
    raise RuntimeError(
        "No lines matched. If your log format differs, paste 2-3 lines and I’ll tweak the regex."
    )

episodes = np.asarray(episodes)
returns  = np.asarray(returns)

# --- helpers: moving average & rolling win rate ---
def moving_average(x, k):
    if k <= 1 or len(x) < k: return None
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")

def rolling_win_rate(rets, k):
    """Win = return > 0. Returns array of size len(rets)-k+1 with percent wins."""
    if k <= 1 or len(rets) < k: return None
    wins = (rets > 0).astype(float)
    # moving average of wins -> fraction; multiply by 100 for percent
    wr = moving_average(wins, k)
    return wr * 100.0 if wr is not None else None

# window sizes (tweak if you like)
ret_window = 10
wr_window  = 25

ret_ma = moving_average(returns, ret_window)
wr_ma  = rolling_win_rate(returns, wr_window)

# --- plotting ---
fig, ax1 = plt.subplots(figsize=(9, 5))

# 1) raw returns
ax1.plot(episodes, returns, marker='.', linewidth=0.6, alpha=0.45, label='Return (raw)')
# 2) smoothed returns
if ret_ma is not None:
    ax1.plot(episodes[ret_window-1:], ret_ma, linewidth=2.0, label=f'Return MA ({ret_window})')

ax1.set_xlabel("Episode")
ax1.set_ylabel("Return (sum of rewards)")
ax1.grid(True, alpha=0.3)

# 3) rolling win rate on secondary axis
if wr_ma is not None:
    ax2 = ax1.twinx()
    ax2.plot(episodes[wr_window-1:], wr_ma, linestyle='--', linewidth=2.0, label=f'Win Rate ({wr_window})', alpha=0.9)
    ax2.set_ylabel("Win rate (%)")
    ax2.set_ylim(0, 100)

    # combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
else:
    ax1.legend(loc="best")

plt.title("SARSA Pong — Return & Rolling Win Rate")
plt.tight_layout()
# Optional: save to file too
# plt.savefig("training_return_winrate.png", dpi=150)
plt.show()
