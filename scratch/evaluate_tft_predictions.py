import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT = Path("/Users/reza/ElliottWaveAnalyzer")
sys.path.insert(0, str(ROOT))

from src.waveconf.wave_model.model import prepare_df_for_tft
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

model_path = "models/wave_model.pt"
data_path = "data/labeled/BTC_1D_labeled.csv"

def main():
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    print("Loading model...")
    device = torch.device("cpu")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=device, weights_only=False)
    model.eval()

    print("Loading data...")
    df = pd.read_csv(data_path)
    df_raw = df.copy() # keep a raw copy for exact price lookup (high, low, close)
    prep_df = prepare_df_for_tft(df)

    print("Rebuilding dataset and running batch predictions...")
    params = model.dataset_parameters
    dataset = TimeSeriesDataSet.from_parameters(params, prep_df)
    
    with torch.no_grad():
        res = model.predict(
            dataset,
            mode="quantiles",
            return_index=True,
            trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_checkpointing": False}
        )

    predictions = res.output.numpy() # shape: (num_samples, 60, 3)
    index = res.index # dataframe with time_idx

    print(f"Generated predictions for {len(index)} samples.")

    # Initialize results list
    eval_results = []

    # Loop through each sample and compute the compounded prices
    # just like in predict_tft in infer.py
    for i in range(len(index)):
        time_idx = int(index.iloc[i]["time_idx"])
        
        # The price at the forecast day (last encoder step)
        last_close = float(df_raw.iloc[time_idx]["close"])
        
        # Get predictions for this sample
        sample_quantiles = predictions[i] # shape: (60, 3)
        
        cumulative_mean = 0.0
        cumulative_var = 0.0
        
        q10_prices = [last_close]
        q50_prices = [last_close]
        q90_prices = [last_close]
        
        for step in range(60):
            y_q10 = float(sample_quantiles[step, 0])
            y_q50 = float(sample_quantiles[step, 1])
            y_q90 = float(sample_quantiles[step, 2])
            
            # Convert to log returns
            r_q10 = np.log(max(0.01, 1.0 + y_q10))
            r_q50 = np.log(max(0.01, 1.0 + y_q50))
            r_q90 = np.log(max(0.01, 1.0 + y_q90))
            
            mean = r_q50
            sd_90 = (r_q90 - r_q50) / 1.28155
            sd_10 = (r_q50 - r_q10) / 1.28155
            sd = max(0.0, (sd_90 + sd_10) / 2.0)
            
            cumulative_mean += mean
            cumulative_var += sd ** 2
            
            cum_sd = np.sqrt(cumulative_var)
            
            cum_r_q10 = cumulative_mean - 1.28155 * cum_sd
            cum_r_q50 = cumulative_mean
            cum_r_q90 = cumulative_mean + 1.28155 * cum_sd
            
            q10_prices.append(last_close * np.exp(cum_r_q10))
            q50_prices.append(last_close * np.exp(cum_r_q50))
            q90_prices.append(last_close * np.exp(cum_r_q90))
            
        # Compile evaluations for 7d, 14d, 30d
        sample_eval = {
            "time_idx": time_idx,
            "last_close": last_close,
        }
        
        for h in [7, 14, 30]:
            # Forecasts at horizon h (index h because index 0 is step 0 i.e. last_close)
            sample_eval[f"q10_{h}d"] = q10_prices[h]
            sample_eval[f"q50_{h}d"] = q50_prices[h]
            sample_eval[f"q90_{h}d"] = q90_prices[h]
            
            # Actual prices at and within horizon h
            future_idx = time_idx + h
            if future_idx < len(df_raw):
                future_window = df_raw.iloc[time_idx + 1 : future_idx + 1]
                actual_close_at_h = float(df_raw.iloc[future_idx]["close"])
                actual_highs = future_window["high"].values
                actual_lows = future_window["low"].values
                actual_closes = future_window["close"].values
                
                sample_eval[f"actual_close_{h}d"] = actual_close_at_h
                sample_eval[f"max_high_{h}d"] = np.max(actual_highs)
                sample_eval[f"min_low_{h}d"] = np.min(actual_lows)
                sample_eval[f"actual_closes_{h}d"] = actual_closes
            else:
                sample_eval[f"actual_close_{h}d"] = None
                sample_eval[f"max_high_{h}d"] = None
                sample_eval[f"min_low_{h}d"] = None
                sample_eval[f"actual_closes_{h}d"] = None
                
        eval_results.append(sample_eval)

    eval_df = pd.DataFrame(eval_results)
    
    print("\n==================================================")
    print("TFT FORECAST EVALUATION REPORT")
    print("==================================================")
    
    for h in [7, 14, 30]:
        print(f"\n--- Horizon: {h} Days ---")
        
        # Filter samples that have complete actual data for this horizon
        h_df = eval_df[eval_df[f"actual_close_{h}d"].notna()].copy()
        total_valid = len(h_df)
        print(f"Valid evaluation samples: {total_valid}")
        
        if total_valid == 0:
            print("No complete actual outcomes for this horizon.")
            continue
            
        # 1. Point Coverage (At the horizon day)
        # Price is inside the [q10, q90] range at day t+h
        inside_range_point = (h_df[f"actual_close_{h}d"] >= h_df[f"q10_{h}d"]) & (h_df[f"actual_close_{h}d"] <= h_df[f"q90_{h}d"])
        breach_below_point = h_df[f"actual_close_{h}d"] < h_df[f"q10_{h}d"]
        breach_above_point = h_df[f"actual_close_{h}d"] > h_df[f"q90_{h}d"]
        
        coverage_point_pct = inside_range_point.mean() * 100
        below_point_pct = breach_below_point.mean() * 100
        above_point_pct = breach_above_point.mean() * 100
        
        print(f"Point Coverage (at day t+{h}):")
        print(f"  Inside [q10, q90] (Empirical Coverage): {coverage_point_pct:.2f}% (Expected: ~80%)")
        print(f"  Breach Below q10 (Downside risk):       {below_point_pct:.2f}% (Expected: ~10%)")
        print(f"  Breach Above q90 (Upside surprise):     {above_point_pct:.2f}% (Expected: ~10%)")

        # 2. Path Coverage (Within the entire h-day period)
        # Does the price stay within the predicted range at all steps from 1 to h?
        # For this we need to check if for all k in 1..h, q10_k <= close_k <= q90_k
        # Let's compute this for each sample
        inside_range_path_list = []
        for idx, row in h_df.iterrows():
            time_idx = row["time_idx"]
            closes = row[f"actual_closes_{h}d"]
            # get predicted q10 and q90 paths up to step h for this sample
            # rebuild the path for this specific row
            last_close = row["last_close"]
            # we can calculate the paths of q10 and q90 for steps 1..h
            sample_quantiles = predictions[idx] # rebuild
            cumulative_mean = 0.0
            cumulative_var = 0.0
            q10_path = []
            q90_path = []
            for step in range(h):
                y_q10 = float(sample_quantiles[step, 0])
                y_q50 = float(sample_quantiles[step, 1])
                y_q90 = float(sample_quantiles[step, 2])
                r_q10 = np.log(max(0.01, 1.0 + y_q10))
                r_q50 = np.log(max(0.01, 1.0 + y_q50))
                r_q90 = np.log(max(0.01, 1.0 + y_q90))
                mean = r_q50
                sd_90 = (r_q90 - r_q50) / 1.28155
                sd_10 = (r_q50 - r_q10) / 1.28155
                sd = max(0.0, (sd_90 + sd_10) / 2.0)
                cumulative_mean += mean
                cumulative_var += sd ** 2
                cum_sd = np.sqrt(cumulative_var)
                cum_r_q10 = cumulative_mean - 1.28155 * cum_sd
                cum_r_q90 = cumulative_mean + 1.28155 * cum_sd
                q10_path.append(last_close * np.exp(cum_r_q10))
                q90_path.append(last_close * np.exp(cum_r_q90))
                
            # check if actual closes stay within q10_path and q90_path at all steps
            # len(closes) is h
            in_range = True
            for step in range(min(len(closes), h)):
                if closes[step] < q10_path[step] or closes[step] > q90_path[step]:
                    in_range = False
                    break
            inside_range_path_list.append(in_range)
            
        coverage_path_pct = np.mean(inside_range_path_list) * 100
        print(f"Path Coverage (remains inside for ALL {h} days):")
        print(f"  Remains within [q10, q90] range:        {coverage_path_pct:.2f}% (Winrate that range isn't hit/breached)")

        # 3. Forecast Accuracy (Mean Absolute Error for q50)
        absolute_errors = (h_df[f"q50_{h}d"] - h_df[f"actual_close_{h}d"]).abs()
        mae_usd = absolute_errors.mean()
        mae_pct = (absolute_errors / h_df[f"actual_close_{h}d"] * 100).mean()
        print(f"q50 Median Forecast Accuracy:")
        print(f"  Mean Absolute Error (MAE):              ${mae_usd:,.2f} ({mae_pct:.2f}%)")

        # 4. Favorable Excursion (MFE) & Adverse Excursion (MAE)
        # We classify forecast as BULLISH if q50_h > last_close, else BEARISH
        h_df["forecast_direction"] = np.where(h_df[f"q50_{h}d"] > h_df["last_close"], "bullish", "bearish")
        
        mfe_list = []
        mae_list = []
        
        for idx, row in h_df.iterrows():
            direction = row["forecast_direction"]
            entry_price = row["last_close"]
            max_high = row[f"max_high_{h}d"]
            min_low = row[f"min_low_{h}d"]
            
            if direction == "bullish":
                mfe_val = (max_high - entry_price) / entry_price * 100
                mae_val = (entry_price - min_low) / entry_price * 100
            else: # bearish
                mfe_val = (entry_price - min_low) / entry_price * 100
                mae_val = (max_high - entry_price) / entry_price * 100
                
            mfe_list.append(mfe_val)
            mae_list.append(mae_val)
            
        h_df["mfe"] = mfe_list
        h_df["mae"] = mae_list
        
        print(f"Trade Excursions (MFE & MAE) based on q50 direction:")
        print(f"  MFE (Maximum Favorable Excursion) %:")
        print(f"    Mean: {h_df['mfe'].mean():.2f}%")
        print(f"    Min:  {h_df['mfe'].min():.2f}%")
        print(f"    Max:  {h_df['mfe'].max():.2f}%")
        print(f"  MAE (Maximum Adverse Excursion) %:")
        print(f"    Mean: {h_df['mae'].mean():.2f}%")
        print(f"    Min:  {h_df['mae'].min():.2f}%")
        print(f"    Max:  {h_df['mae'].max():.2f}%")

if __name__ == "__main__":
    main()
