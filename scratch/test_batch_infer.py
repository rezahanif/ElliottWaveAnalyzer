import os
import sys
import torch
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/reza/ElliottWaveAnalyzer")
sys.path.insert(0, str(ROOT))

from src.waveconf.wave_model.model import prepare_df_for_tft
from pytorch_forecasting import TemporalFusionTransformer

model_path = "models/wave_model.pt"
data_path = "data/labeled/BTC_1D_labeled.csv"

device = torch.device("cpu")
model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=device, weights_only=False)
model.eval()

df = pd.read_csv(data_path)
prep_df = prepare_df_for_tft(df)

with torch.no_grad():
    res = model.predict(
        prep_df,
        mode="quantiles",
        return_index=True,
        trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_checkpointing": False}
    )

print("Type of res:", type(res))
if isinstance(res, tuple) or isinstance(res, list):
    print("Length of res:", len(res))
    for idx, item in enumerate(res):
        print(f"Item {idx} type: {type(item)}")
        if hasattr(item, "shape"):
            print(f"  Shape: {item.shape}")
        elif isinstance(item, pd.DataFrame):
            print(f"  DF head:\n{item.head()}")
else:
    print(res)
