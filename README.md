# Bridge Flow Cast

LSTM-based predictor for forecasting water surface profiles over bridges during extreme floods.

**Important:** This predictor requires both the trained LSTM model (`LSTM_new.keras`) and the trained scaler (`scaler.joblib`) inside the `model/` directory. Without the scaler, predictions will not be valid.

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Be sure `LSTM_new.keras` model and `scaler.joblib` are in the `model/` folder.

3. Run prediction:
   ```bash
   python bridge_flow_cast.py --model ./model/LSTM_new.keras --scaler ./model/scaler.joblib --input sample/sample_input.csv --out sample/predictions.csv
   ```

4. Output:
   - A CSV file with predicted water levels.
   - A PNG plot with the profile of the first prediction.

## License
MIT (or any license you prefer)
