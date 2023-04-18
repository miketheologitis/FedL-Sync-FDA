import pandas as pd

combined_epoch_metrics = pd.read_parquet('epoch_metrics/tmp/')
combined_round_metrics = pd.read_parquet('round_metrics/tmp')

combined_epoch_metrics.to_parquet('epoch_metrics/combined_epoch_metrics.parquet')
combined_round_metrics.to_parquet('round_metrics/combined_round_metrics.parquet')

