import re
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')   # or 'MacOSX' on Mac; do this before importing pyplot
import matplotlib.pyplot as plt
plt.ioff()  # turn off interactive mode so each plt.show() opens cleanly

import matplotlib.pyplot as plt
import numpy as np

import re
import math

LINE_RE = re.compile(
    r'^Episode\s+(\d+)\s*/\s*\d+\s*\|\s*return=([-\d.]+)\s*\|\s*len=\s*(\d+)\s*\|\s*eps=([-\d.]+)\s*\|\s*loss=([-\d.]+)',
    re.I
)

def parse_training_log(log_file_path):
    """
    EXACT matcher for lines like:
      Episode    1/500 | return=-21.00 | len= 946 | eps=0.997 | loss=0.02247
    Treats len as steps.
    """
    episodes, returns, lengths, epsilons, losses, steps = [], [], [], [], [], []
    unmatched = 0
    preview = []

    with open(log_file_path, "r") as f:
        for line in f:
            s = line.strip()
            m = LINE_RE.match(s)
            if m:
                ep  = int(m.group(1))
                ret = float(m.group(2))
                ln  = int(m.group(3))
                eps = float(m.group(4))
                los = float(m.group(5))

                episodes.append(ep)
                returns.append(ret)
                lengths.append(ln)
                epsilons.append(eps)
                losses.append(los)
                steps.append(ln)  # use length as steps
            else:
                if "eval" not in s.lower():
                    unmatched += 1
                    if len(preview) < 5 and s:
                        preview.append(s)

    data = dict(
        episodes=episodes,
        returns=returns,
        lengths=lengths,
        epsilons=epsilons,
        losses=losses,
        steps=steps
    )

    print("\n--- Parse report ---")
    print({k: len(v) for k, v in data.items()})
    print(f"Unmatched (non-episode) lines: {unmatched}")
    if preview:
        print("Examples of unmatched lines:")
        for p in preview:
            print("  ", p[:140])
    print("--------------------\n")

    return data


def create_visualizations_separate(data, save_dir='.'):
    plt.close('all')  # start clean
    window = 10

    # 1) Returns
    plt.figure(figsize=(8,5))
    plt.title('Episode Returns')
    plt.plot(data['episodes'], data['returns'], alpha=0.7, linewidth=1)
    if len(data['returns']) > window:
        returns_ma = pd.Series(data['returns']).rolling(window=window, center=True).mean()
        plt.plot(data['episodes'], returns_ma, linewidth=2, label=f'MA({window})')
        plt.legend()
    plt.xlabel('Episode'); plt.ylabel('Return'); plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/fig1_returns.png', dpi=300, bbox_inches='tight')
    print('Saved fig1_returns.png')
    plt.show()

    # 2) Episode lengths
    plt.figure(figsize=(8,5))
    plt.title('Episode Lengths')
    plt.plot(data['episodes'], data['lengths'], alpha=0.7, linewidth=1)
    plt.xlabel('Episode'); plt.ylabel('Steps'); plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/fig2_lengths.png', dpi=300, bbox_inches='tight')
    print('Saved fig2_lengths.png')
    plt.show()

    # 3) Epsilon
    plt.figure(figsize=(8,5))
    plt.title('Epsilon (Exploration Rate)')
    plt.plot(data['episodes'], data['epsilons'], linewidth=2)
    plt.xlabel('Episode'); plt.ylabel('Epsilon'); plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/fig3_epsilon.png', dpi=300, bbox_inches='tight')
    print('Saved fig3_epsilon.png')
    plt.show()

    # 4) Loss
    plt.figure(figsize=(8,5))
    plt.title('Training Loss')
    plt.plot(data['episodes'], data['losses'], alpha=0.7, linewidth=1)
    if len(data['losses']) > window:
        loss_ma = pd.Series(data['losses']).rolling(window=window, center=True).mean()
        plt.plot(data['episodes'], loss_ma, linewidth=2, label=f'MA({window})')
        plt.legend()
    plt.xlabel('Episode'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/fig4_loss.png', dpi=300, bbox_inches='tight')
    print('Saved fig4_loss.png')
    plt.show()

    # 5) Cumulative steps
    plt.figure(figsize=(8,5))
    plt.title('Cumulative Training Steps')
    cumulative_steps = np.cumsum(data['steps'])
    plt.plot(data['episodes'], cumulative_steps, linewidth=2)
    plt.xlabel('Episode'); plt.ylabel('Total Steps'); plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/fig5_cum_steps.png', dpi=300, bbox_inches='tight')
    print('Saved fig5_cum_steps.png')
    plt.show()

    # 6) Returns distribution
    plt.figure(figsize=(8,5))
    plt.title('Returns Distribution')
    plt.hist(data['returns'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Return'); plt.ylabel('Frequency'); plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/fig6_returns_hist.png', dpi=300, bbox_inches='tight')
    print('Saved fig6_returns_hist.png')
    plt.show()


def print_summary_statistics(data):
    """
    Print key statistics about the training run
    """
    print("\n" + "="*50)
    print("TRAINING RUN SUMMARY")
    print("="*50)
    print(f"Total episodes: {len(data['episodes'])}")
    print(f"Total steps: {sum(data['steps']):,}")
    print(f"Final epsilon: {data['epsilons'][-1]:.3f}")
    print(f"\nReturns Statistics:")
    print(f"  Average return: {np.mean(data['returns']):.2f}")
    print(f"  Best return: {np.max(data['returns']):.2f}")
    print(f"  Worst return: {np.min(data['returns']):.2f}")
    print(f"  Final return: {data['returns'][-1]:.2f}")
    print(f"\nLoss Statistics:")
    print(f"  Average loss: {np.mean(data['losses']):.4f}")
    print(f"  Final loss: {data['losses'][-1]:.4f}")
    
    # Check for learning progress
    first_half = data['returns'][:len(data['returns'])//2]
    second_half = data['returns'][len(data['returns'])//2:]
    if len(first_half) > 0 and len(second_half) > 0:
        improvement = np.mean(second_half) - np.mean(first_half)
        print(f"  Improvement (2nd half - 1st half): {improvement:.2f}")

def main():
    import time
    log_file = "raw_training_log.txt"  # <-- change if your file has a different name or path

    print("DEBUG: running file:", __file__)
    print("DEBUG: current working dir:", os.getcwd())
    print("DEBUG: expected log_file:", log_file)
    print("DEBUG: exists?", os.path.exists(log_file))
    if os.path.exists(log_file):
        try:
            print("DEBUG: file size (bytes):", os.path.getsize(log_file))
            # print first 3 lines so we know we’re reading the right file
            with open(log_file, "r") as f:
                print("DEBUG: head:")
                for i in range(3):
                    line = f.readline()
                    if not line:
                        break
                    print("   ", line.rstrip())
        except Exception as e:
            print("DEBUG: error reading file head:", e)

    print("DEBUG: about to parse…"); time.sleep(0.1)

    # Use whichever parser you have in the file:
    # data = parse_training_log(log_file, dump_preview=True)          # robust version
    data = parse_training_log(log_file)

    print("DEBUG: parsed counts:", {k: len(v) for k, v in data.items()})

    if not data["episodes"]:
        print("No episode lines matched. Double-check the file name/path or show me the debug head above.")
        return

    print_summary_statistics(data)

    print("\nCreating visualizations (six separate windows)…")
    create_visualizations_separate(data, save_dir=".")
    print("\nDone. Check fig1_... to fig6_... PNGs in this folder.")

if __name__ == "__main__":
    main()