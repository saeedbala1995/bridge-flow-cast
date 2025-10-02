#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils import load_scaler_from_npz, ensure_3d_input

def predict(args):
    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")

    model = load_model(model_path)

    scaler = None
    if args.scaler:
        scaler = load_scaler_from_npz(args.scaler)

    df = pd.read_csv(args.input)
    req = ['depth','velocity','froude']
    cols = [c for c in df.columns if c.lower() in req]
    if len(cols) < 3:
        raise ValueError("Input CSV must contain columns: depth, velocity, froude.")

    X = df[cols].values.astype(float)
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        denom = (X_max - X_min)
        denom[denom==0]=1.0
        X_scaled = (X - X_min) / denom

    X_final = ensure_3d_input(X_scaled)
    preds = model.predict(X_final)

    cols_out = [f"wl_{i}" for i in range(preds.shape[1])]
    df_out = pd.DataFrame(preds, columns=cols_out)
    df_out.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")

    plt.figure()
    x_space = np.linspace(0, 2.8, preds.shape[1])
    plt.plot(x_space, preds[0], label='Predicted')
    plt.xlabel('X (m)')
    plt.ylabel('Water level (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out.replace('.csv','.png'))
    print("Plot saved to", args.out.replace('.csv','.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to LSTM.h5')
    parser.add_argument('--scaler', required=False, help='Path to scaler file (optional)')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--out', default='predictions.csv', help='Output CSV path')
    args = parser.parse_args()
    predict(args)

if __name__ == '__main__':
    main()
