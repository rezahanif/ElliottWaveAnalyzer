from pytorch_forecasting import TimeSeriesDataSet
print([name for name in dir(TimeSeriesDataSet) if "param" in name.lower() or "from" in name.lower()])
