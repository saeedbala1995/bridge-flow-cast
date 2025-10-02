# Bridge Water Surface Predictor

This package provides a simple predictor that loads a pre-trained LSTM model and predicts water surface profiles (61 points).

**Note:** The trained model (`LSTM.h5`) and scaler are **not included** in this repository due to data/privacy restrictions.

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your `LSTM.h5` model in `model/` and (optionally) a scaler file (`scaler.joblib` or `scaler.npz`).

3. Run prediction:
   ```bash
   python predictor.py --model ./model/LSTM.h5 --scaler ./model/scaler.npz --input sample/sample_input.csv --out sample/predictions.csv
   ```

4. Output:
   - A CSV file with predicted water levels.
   - A PNG plot comparing predicted water levels.

## License
MIT (or any license you prefer)
