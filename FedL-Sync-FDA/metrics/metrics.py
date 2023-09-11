from collections import namedtuple

"""
RoundMetrics namedtuple:
Attributes:
    - epoch (int): The epoch number at which this round ended.
    - round (int): The round number.
    - total_fda_steps (int): Total FDA steps taken till this round.
    - est_var (float): Estimated variance at the end of this round.
    - actual_var (float): Actual variance at the end of this round.
"""
# Define a named tuple to store round-specific metrics in federated learning.
RoundMetrics = namedtuple("RoundMetrics", ["epoch", "round", "total_fda_steps", "est_var", "actual_var"])


"""
EpochMetrics namedtuple:
Attributes:
    - epoch (int): The epoch number.
    - total_rounds (int): The total number of rounds at the end of this epoch.
    - total_fda_steps (int): Total FDA steps taken till this epoch.
    - accuracy (float): Model's accuracy at the end of this epoch.
"""
# Define a named tuple to store epoch-specific metrics in federated learning.
EpochMetrics = namedtuple("EpochMetrics", ["epoch", "total_rounds", "total_fda_steps", "accuracy"])


"""
TestId namedtuple:
Attributes:
    - dataset_name (str): Name of the dataset used.
    - fda_name (str): FDA method name.
    - num_clients (int): Number of clients in the federated network.
    - batch_size (int): Batch size for training.
    - num_steps_until_rtc_check (int): Number of steps until the RTC check.
    - theta (float): The threshold value for FDA.
    - nn_num_weights (int): Number of weights in the neural network.
    - sketch_width (int): Width of the sketch.
    - sketch_depth (int): Depth of the sketch.
"""
# Define a named tuple to represent test IDs for different experiments.
TestId = namedtuple(
        'TestId',
        ["dataset_name", "fda_name", "num_clients", "batch_size", "num_steps_until_rtc_check",
         "theta", "nn_name", "nn_num_weights", "sketch_width", "sketch_depth"]
)

# Extend the EpochMetrics and RoundMetrics namedtuples to include TestId.
EpochMetricsWithId = namedtuple('EpochMetricsWithId', TestId._fields + EpochMetrics._fields)
RoundMetricsWithId = namedtuple('RoundMetricsWithId', TestId._fields + RoundMetrics._fields)


def process_metrics_with_test_id(epoch_metrics_list, test_id):
    """
    Process the given epoch and round metrics lists to append TestId.
    
    Args:
    - epoch_metrics_list (list of EpochMetrics): List of epoch metrics.
    - round_metrics_list (list of RoundMetrics): List of round metrics.
    - test_id (TestId): An instance of TestId namedtuple.

    Returns:
    - tuple: (epoch_metrics_with_test_id, round_metrics_with_test_id)
    """
    epoch_metrics_with_test_id = [
        EpochMetricsWithId(*test_id, *epoch_metrics)
        for epoch_metrics in epoch_metrics_list
    ]
    
    return epoch_metrics_with_test_id
