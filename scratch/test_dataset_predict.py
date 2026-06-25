import os
import sys
import torch
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/reza/ElliottWaveAnalyzer")
sys.path.insert(0, str(ROOT))

from src.waveconf.wave_model.model import prepare_df_for_tft
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

model_path = "models/wave_model.pt"
data_path = "data/labeled/BTC_1D_labeled.csv"

device = torch.device("cpu")
model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=device, weights_only=False)
model.eval()

df = pd.read_csv(data_path)
prep_df = prepare_df_for_tft(df)

# Rebuild TimeSeriesDataSet using the model's parameters
params = model.dataset_parameters
print("Rebuilding dataset from parameters...")
dataset = TimeSeriesDataSet.from_parameters(params, prep_df)
print(f"Dataset has {len(dataset)} samples.")

with torch.no_grad():
    res = model.predict(
        dataset,
        mode="quantiles",
        return_index=True,
        trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_checkpointing": False}
    )

print("res attributes:", dir(res))
predictions = res.output
index = res.index
print(f"Predictions shape: {predictions.shape}")
print(f"Index shape: {index.shape}")
print("Index head:")
print(index.head())
