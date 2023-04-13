import import_ipynb
import TF_Simulation_FDA_CNN as sim
import pandas as pd
import sys

NUM_CLIENTS_SINGLE_TEST = int(sys.argv[1])

epoch_metrics_filename = f'../simulation_results/epoch_metrics{NUM_CLIENTS_SINGLE_TEST}.parquet'
round_metrics_filename = f'../simulation_results/round_metrics{NUM_CLIENTS_SINGLE_TEST}.parquet'

all_epoch_metrics, all_round_metrics = sim.run_tests(
    num_clients_list=[NUM_CLIENTS_SINGLE_TEST],
    batch_size_list=[32, 64, 128],
    num_steps_until_rtc_check_list=[1],
    theta_list=[0.05, 0.5, 1., 3., 5., 7., 10.],
    num_epochs=10,
    sketch_width=500,
    sketch_depth=7
)

epoch_metrics_df = pd.DataFrame(all_epoch_metrics)
round_metrics_df = pd.DataFrame(all_round_metrics)

epoch_metrics_df.to_parquet(epoch_metrics_filename)
round_metrics_df.to_parquet(round_metrics_filename)