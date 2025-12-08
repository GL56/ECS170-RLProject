import os
import re
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "logs"   # folder containing your training logs

# Regex to extract "Episode X" and "return=Y"
EPISODE_RE = re.compile(r"Episode\s+(\d+)")
RETURN_RE = re.compile(r"return=([-+]?\d+\.\d+|\d+)")

def parse_log(filepath):
    episodes = []
    returns = []

    with open(filepath, "r") as f:
        for line in f:
            ep_match = EPISODE_RE.search(line)
            ret_match = RETURN_RE.search(line)

            if ep_match and ret_match:
                episodes.append(int(ep_match.group(1)))
                returns.append(float(ret_match.group(1)))

    return np.array(episodes), np.array(returns)


def main():
    plt.figure(figsize=(10, 6))

    for filename in os.listdir(LOG_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(LOG_DIR, filename)
            episodes, returns = parse_log(path)

            if len(episodes) == 0:
                print(f"⚠️ No episode/return pairs found in {filename}")
                continue

            plt.plot(episodes, returns, label=filename)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Returns Across Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
