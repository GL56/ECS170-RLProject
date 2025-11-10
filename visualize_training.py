import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_file_path):
    """
    Parse the raw training log and extract metrics
    """
    episodes = []
    returns = []
    lengths = []
    epsilons = []
    losses = []
    steps = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Match the main training log lines
            if 'Episode' in line and 'return=' in line:
                # Extract episode number
                ep_match = re.search(r'Episode\s+(\d+)', line)
                return_match = re.search(r'return=([-\d.]+)', line)
                len_match = re.search(r'len=(\d+)', line)
                eps_match = re.search(r'eps=([\d.]+)', line)
                loss_match = re.search(r'loss=([\d.]+)', line)
                steps_match = re.search(r'steps=(\d+)', line)
                
                if all([ep_match, return_match, len_match, eps_match, loss_match, steps_match]):
                    episodes.append(int(ep_match.group(1)))
                    returns.append(float(return_match.group(1)))
                    lengths.append(int(len_match.group(1)))
                    epsilons.append(float(eps_match.group(1)))
                    losses.append(float(loss_match.group(1)))
                    steps.append(int(steps_match.group(1)))
            
            # Also capture evaluation results
            elif 'Eval over' in line:
                eval_match = re.search(r'mean_return=([-\d.]+)', line)
                if eval_match:
                    print(f"Evaluation result: {eval_match.group(1)}")
    
    return {
        'episodes': episodes,
        'returns': returns,
        'lengths': lengths,
        'epsilons': epsilons,
        'losses': losses,
        'steps': steps
    }

def create_visualizations(data, save_path=None):
    """
    Create comprehensive training visualization plots
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DQN Pong Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Returns over episodes
    axes[0, 0].plot(data['episodes'], data['returns'], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Smooth returns (moving average)
    window = 10
    if len(data['returns']) > window:
        returns_ma = pd.Series(data['returns']).rolling(window=window, center=True).mean()
        axes[0, 0].plot(data['episodes'], returns_ma, 'r-', linewidth=2, label=f'MA({window})')
        axes[0, 0].legend()
    
    # Plot 2: Episode lengths
    axes[0, 1].plot(data['episodes'], data['lengths'], 'g-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    axes[0, 2].plot(data['episodes'], data['epsilons'], 'orange', linewidth=2)
    axes[0, 2].set_title('Epsilon (Exploration Rate)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Loss over episodes
    axes[1, 0].plot(data['episodes'], data['losses'], 'purple', alpha=0.7, linewidth=1)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Smooth loss (moving average)
    if len(data['losses']) > window:
        loss_ma = pd.Series(data['losses']).rolling(window=window, center=True).mean()
        axes[1, 0].plot(data['episodes'], loss_ma, 'r-', linewidth=2, label=f'MA({window})')
        axes[1, 0].legend()
    
    # Plot 5: Cumulative steps
    cumulative_steps = np.cumsum(data['steps'])
    axes[1, 1].plot(data['episodes'], cumulative_steps, 'brown', linewidth=2)
    axes[1, 1].set_title('Cumulative Training Steps')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Steps')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Returns distribution (histogram)
    axes[1, 2].hist(data['returns'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 2].set_title('Returns Distribution')
    axes[1, 2].set_xlabel('Return')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig

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
    log_file = "raw_training_log.txt"  # Change this if your file has a different name
    output_plot = "training_analysis.png"
    
    try:
        # Parse the log file
        print("Parsing training log...")
        data = parse_training_log(log_file)
        
        if not data['episodes']:
            print("No training data found in the log file!")
            return
        
        # Print summary
        print_summary_statistics(data)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(data, save_path=output_plot)
        
        print(f"\nAnalysis complete! Check '{output_plot}' for the plots.")
        
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found!")
    except Exception as e:
        print(f"Error analyzing log file: {e}")

if __name__ == "__main__":
    main()