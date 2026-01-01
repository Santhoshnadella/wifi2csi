import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="WiFi-CSI Sensing CLI")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "live", "offline"], help="Operation mode")
    parser.add_argument("--gui", action="store_true", help="Launch Streamlit GUI")
    
    args = parser.parse_args()
    
    if args.gui:
        print("Launching GUI...")
        os.system("streamlit run app.py")
    else:
        print(f"Running in {args.mode} mode (Headless/CLI)...")
        # Logic for headless run (saving files/logs)
        # Import here to avoid slow startup for just help
        from utils import generate_synthetic_csi
        data = generate_synthetic_csi()
        print(f"Generated synthetic data shape: {data.shape}")
        print("Model inference step...")
        # (Load model and run one pass for verification)
        pass

if __name__ == "__main__":
    main()
