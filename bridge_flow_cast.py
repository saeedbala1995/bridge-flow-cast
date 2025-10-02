#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils import load_scaler, ensure_3d_input

def predict(args):
    # load model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model}.")
    model = load_model(args.model)

    # load scaler (mandatory)
    if not os.path.exists(args.scaler):
        raise FileNotFoundError(f"Scaler not found at {args.scaler}.")
    scaler = load_scaler(args.scaler)

    # load input
    df = pd.read_csv(args.input)
    req = ['depth','velocity','froude']
    cols = [c for c in df.columns if c.lower() in req]
    if len(cols) < 3:
        raise ValueError("Input CSV must contain columns: depth, velocity, froude.")
    X = df[cols].values.astype(float)

    # scale with trained scaler
    X_scaled = scaler.transform(X)

    # reshape to fit LSTM input
    X_final = ensure_3d_input(X_scaled)

    # predict
    preds = model.predict(X_final)

    # save output
    cols_out = [f"wl_{i}" for i in range(preds.shape[1])]
    df_out = pd.DataFrame(preds, columns=cols_out)
    df_out.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")

    # save plot
    plt.figure()
    x_space = np.linspace(0, 2.8, preds.shape[1])
    plt.plot(x_space, preds[0], label='Predicted')
    plt.xlabel('x/L')
    plt.ylabel('y/(h+D)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out.replace('.csv', '.png'))
    print("Plot saved to", args.out.replace('.csv', '.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to LSTM model (.keras or .h5)')
    parser.add_argument('--scaler', required=True, help='Path to scaler file (.joblib or .npz)')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--out', default='predictions.csv', help='Output CSV path')
    args = parser.parse_args()
    predict(args)

if __name__ == '__main__':
    main()
