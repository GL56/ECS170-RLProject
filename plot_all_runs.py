import os
import re
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "logs"   # folder containing your training logs

# Regex to extract "return=..." from log lines
RETURN_RE = re.compile(r"return=([-+]?\d+\.\d+|\d+)")

def parse_returns(filepath):
    """Extracts returns from a log file, in the order they appear."""
    returns = []
    with open(filepath, "r") as f:
        for line in f:
            match = RETURN_RE.search(line)
            if match:
                returns.append(float(match.group(1)))
    return returns  # plain Python list is fine


def main():
    all_returns = []

    # Make sure logs are processed in order: run1, run2, ..., run5
    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(LOG_DIR, filename)
        returns = parse_returns(path)

        if len(returns) == 0:
            print(f"⚠️ No returns found in {filename}, skipping.")
            continue

        all_returns.extend(returns)   # append this file's returns to the big list

    if not all_returns:
        print("No data found in logs.")
        return

    # Now we have one long sequence of returns
    all_returns = np.array(all_returns)
    episodes = np.arange(1, len(all_returns) + 1)

    plt.figure(figsize=(10, 6))

    # Dot plot (scatter) OR line plot – choose ONE of these:

    # 1) Dot plot:
    #plt.scatter(episodes, all_returns, s=10)

    # 2) If you prefer a connected curve instead, comment the scatter above
    #    and uncomment this:
    plt.plot(episodes, all_returns, marker='.', linewidth=1)

    plt.xlabel("Episode (global, across all runs)")
    plt.ylabel("Return")
    plt.title("Sequential Training Returns (Runs 1–5 concatenated)")
    plt.ylim(-21, 21) 
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
