# Bridge Flow Cast

LSTM-based predictor for forecasting **water surface profiles over bridges during extreme floods**, developed as part of the study published in *Engineering Applications of Artificial Intelligence* (Elsevier, 2024).

---

## 📦 Quick start

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/bridge-flow-cast.git
   cd bridge-flow-cast
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a prediction** (using provided sample input)
   ```bash
   python bridge_flow_cast.py        --model ./model/LSTM_new.keras        --scaler ./model/scaler.joblib        --input sample/sample_input.csv        --out sample/predictions.csv
   ```

4. **Results**
   - Predicted water surface profiles are saved in `predictions.csv`.  
   - A visualization is automatically generated as `predictions.png`.  

---

## 📘 Model Output Interpretation  

The predictor estimates the **water surface profile** across the bridge under flood scenarios.  

- **X-axis (normalized longitudinal coordinate):**  
  Streamwise distance along the bridge normalized by the bridge length \(L\).  
  x/L
  where \(x\) is the position along the bridge deck and \(L\) is the total bridge length.  
  Predictions are made at 61 discrete non-dimensional points between the upstream and downstream sections.

- **Y-axis (normalized vertical coordinate):**  
  Vertical position of the water surface, expressed as:  
  y/(h+D)
  where:  
  - \( y \) = water level (vertical coordinate),  
  - \( h \) = pier height,  
  - \( D \) = deck thickness.  

  This makes the results **dimensionless**, ensuring comparability across different bridge geometries and scales.

- **Output file (`predictions.csv`):**  
  Contains normalized water surface elevations at 61 points (`wl_0` … `wl_60`).  

- **Output plot (`predictions.png`):**  
  Shows the predicted water surface profile with:  
  - X-axis: normalized streamwise distance \(x/L\),  
  - Y-axis: normalized water surface elevation \(y/(h+D)\).

---

## 📂 Repository Structure

```
bridge-flow-cast/
├─ bridge_flow_cast.py      # main predictor script
├─ utils.py                 # helper functions (scaler + reshaping)
├─ requirements.txt         # dependencies
├─ README.md
├─ model/
│   ├─ LSTM_new.keras       # trained LSTM model (forward-compatible)
│   └─ scaler.joblib        # fitted MinMaxScaler
├─ sample/
│   └─ sample_input.csv     # example input data
├─ tests/
│   └─ test_predictor.py    # simple unit test
└─ .github/workflows/ci.yml # CI test workflow
```

---

## 📑 Notes

- Input CSV must contain at least the following columns:
  - `Flow depth in River`
  - `Mean Flow Velocity`
  - `Froude Number in river`

- Predictions are based on the optimized LSTM model described in the published paper.  
- Both the trained **model (`.keras`)** and **scaler (`.joblib`)** are provided, making the predictor **ready-to-run**.  

---

## 📜 License

MIT (or your preferred license).
