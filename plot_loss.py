
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor to last smoothed value
    return smoothed

def main():
    parser = argparse.ArgumentParser(description="Plot Loss Curve from CSV")
    parser.add_argument("--dir", type=str, default="output/tiny_adapter_mixed_7030", help="Output directory containing loss.csv")
    args = parser.parse_args()
    
    csv_path = os.path.join(args.dir, "loss.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if len(df) < 2:
        print("Not enough data to plot.")
        return

    steps = df["step"].values
    losses = df["loss"].values
    
    # Smoothing
    smoothed_losses = smooth(losses, 0.8)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, alpha=0.3, label="Raw Loss", color="gray")
    plt.plot(steps, smoothed_losses, label="Smoothed (EMA)", color="#FF5733", linewidth=2)
    
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = os.path.join(args.dir, "loss_curve.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
