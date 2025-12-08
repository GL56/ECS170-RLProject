import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Folder containing your log text files (run1.txt, run2.txt, ...)
LOG_DIR = "logs"

# Regex to extract "return=..." (handles ints and floats, positive or negative)
RETURN_RE = re.compile(r"return=([-+]?\d*\.?\d+)")

def parse_returns(filepath):
    """
    Extracts all return values from a log file, in the order they appear.
    Returns a plain Python list of floats.
    """
    returns = []
    with open(filepath, "r") as f:
        for line in f:
            match = RETURN_RE.search(line)
            if match:
                try:
                    r = float(match.group(1))
                    returns.append(r)
                except ValueError:
                    print(f"⚠️ Bad return value in {filepath}: {match.group(1)}")
    return returns

def causal_moving_average(x, window_size):
    """
    Causal moving average:
    - For each index i, average x[max(0, i-window_size+1) : i+1]
    - Early points use smaller windows instead of padding with zeros.
    This avoids artificial peaks at the start/end.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if window_size <= 1 or n == 0:
        return x.copy()

    cumsum = np.cumsum(x)
    ma = np.empty_like(x, dtype=float)

    for i in range(n):
        start = max(0, i - window_size + 1)
        count = i - start + 1
        if start == 0:
            window_sum = cumsum[i]
        else:
            window_sum = cumsum[i] - cumsum[start - 1]
        ma[i] = window_sum / count

    return ma

def main():
    all_returns = []

    # Process log files in sorted order: run1.txt, run2.txt, ...
    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(LOG_DIR, filename)
        returns = parse_returns(path)

        if len(returns) == 0:
            print(f"⚠️ No returns found in {filename}, skipping.")
            continue

        print(f"Loaded {len(returns)} episodes from {filename}")
        all_returns.extend(returns)

    if not all_returns:
        print("No data found in any logs.")
        return

    # Convert to NumPy arrays
    all_returns = np.array(all_returns, dtype=float)
    episodes = np.arange(1, len(all_returns) + 1, dtype=int)

    # ---- Smoothing settings ----
    SMOOTH_WINDOW = 25  # try 25, 50, etc.
    smoothed_returns = causal_moving_average(all_returns, SMOOTH_WINDOW)

    # ---- Plotting ----
    plt.figure(figsize=(10, 6))

    # Raw data as light dots
    plt.scatter(
        episodes,
        all_returns,
        s=8,
        alpha=0.25,
        label="Episode returns"
    )

    # Smoothed curve on top (no weird edge peaks)
    plt.plot(
        episodes,
        smoothed_returns,
        linewidth=2,
        label=f"Causal moving average (window={SMOOTH_WINDOW})"
    )

    plt.xlabel("Episode (global across all runs)")
    plt.ylabel("Return")
    plt.title("Sequential SARSA Training Returns (Smoothed)")

    # Adjust this if your returns range is different
    plt.ylim(-21, -10)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
