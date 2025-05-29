"""
Provides data-analyzing functions for the stat script.
"""

import numpy as np

from utils import csv_to_array
from utils import get_RE_quantiles
from utils import get_min_RE_metrics
from utils import get_identify_shortest_accuracies
from utils import identify_shortest


def verify_runtime_estimator(verify_csv_path):
    """
    Checks the similarity of the estimated runtimes and Qiskit pulse schedule durations.
    Args:
        verify_csv_path (str): The path to the verification data CSV file.
    """

    eagle_compilers = ["sabre0330", "sqgm", "qiskit141"]
    metrics = [
        "estimated_runtime",
        "qiskit_scheduler_runtime",
    ]
    first_col = 1

    data = csv_to_array(
        csv_path=verify_csv_path,
        n_circuits=15,
        n_compilers=len(eagle_compilers),
        n_metrics=len(metrics),
        first_col=first_col,
    )

    # For each circuit and compiler version, get the difference
    # |pulse schedule duration - estimated runtime|
    diffs = np.zeros((data.shape[0], data.shape[1]))
    for circuit_pos in range(data.shape[0]):
        for compiler_pos in range(data.shape[1]):
            diff = (
                data[circuit_pos][compiler_pos][1] - data[circuit_pos][compiler_pos][0]
            )
            diffs[circuit_pos][compiler_pos] = abs(diff)

    exact_matches = 0
    max_diff = 0
    for diff in np.nditer(diffs):
        if diff == 0:
            exact_matches += 1
        if diff > max_diff:
            max_diff = diff

    # Retrieve true runtime of circuit producing max diff
    pair_position = np.where(diffs == max_diff)

    # First two coordinates specify the circuit and compiler, last=1 specifies
    # true runtime instead of estimate
    runtime_at_max_diff = data[pair_position[0][0]][pair_position[1][0]][1]

    print("\n===Runtime Estimator Verification===")
    print("diff := |pulse schedule duration - estimated runtime|")
    print(f"Exact matches (diff==0): \t{exact_matches}")
    print(f"Largest diff: \t\t\t{max_diff} s")
    print(f"True runtime @ largest diff: \t{runtime_at_max_diff} s")


def calc_avg_RE_multiplier(datas, metrics):
    """
    Calculates the average number of times smaller the gate-aware median percent
    relative error (%RE) is compared to traditional and multi-qubit depths' median %RE,
    or in equation form:

        1 / num_devices * sum_{devices}( target metric median %RE / gate-aware median %RE )

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    trad_multiplier_sum = 0
    multi_multiplier_sum = 0
    num_devices = 0

    for device_data in datas.values():
        quantiles = get_RE_quantiles(device_data)

        # Add len(metric)-1 offset because absolute errors are stored in the first half.
        # See get_RE_quantiles for more information.
        gate_aware_index = (len(metrics) - 1) + metrics.index("gate_aware_depth_w=avg")
        gate_aware_median_RE = quantiles[gate_aware_index][2]

        # Multi q2 %RE stored in first row of second half, trad in last row of second half
        multi_multiplier = quantiles[len(metrics) - 1][2] / gate_aware_median_RE
        trad_multiplier = quantiles[-1][2] / gate_aware_median_RE

        trad_multiplier_sum += trad_multiplier
        multi_multiplier_sum += multi_multiplier
        num_devices += 1

    trad_avg_multiplier = trad_multiplier_sum / num_devices
    multi_avg_multiplier = multi_multiplier_sum / num_devices
    print("\n===Average Gate-Aware Error Reduction===")
    print(
        "Average decrease in median %RE relative to target metric := "
        + "average of (target metric median %RE / gate-aware median %RE) over all devices"
    )
    print(f"Relative to traditional depth:\t\t{trad_avg_multiplier:.2f} x")
    print(f"Relative to multi-qubit depth:\t\t{multi_avg_multiplier:.2f} x")


def compare_identify_shortest_accuracies(datas, metrics):
    """
    Compares gate-aware depth's accuracy at identifying the circuit version(s) with shortest
    estimated runtime using device-specific and average architecture weights.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    unique_metrics = get_min_RE_metrics(datas, metrics)

    shared_metrics = {
        "IBM Sherbrooke": "gate_aware_depth_w=avg",
        "IBM Kyiv": "gate_aware_depth_w=avg",
        "IBM Brisbane": "gate_aware_depth_w=avg",
        "IBM Marrakesh": "gate_aware_depth_w=avg",
        "IBM Kingston": "gate_aware_depth_w=avg",
        "IBM Aachen": "gate_aware_depth_w=avg",
    }

    unique_accuracies = get_identify_shortest_accuracies(datas, metrics, unique_metrics)
    shared_accuracies = get_identify_shortest_accuracies(datas, metrics, shared_metrics)

    print("\n===Accuracies in Identifying Shortest Circuit Version===")
    print(
        "Accuracy := % of circuits for which depth correctly identified "
        + "version(s) with shortest true runtime"
    )
    for device_name in datas.keys():
        print(f"+++{device_name}+++")
        print(
            f"Using best metric ({unique_metrics[device_name]}):"
            + f"\t\t{unique_accuracies[device_name][-1]} %"
        )
        print(
            f"Using proposed metric ({shared_metrics[device_name]}):"
            + f"\t\t{shared_accuracies[device_name][-1]} %"
        )


def calc_avg_accuracy_increase(datas, metrics):
    """
    Calculates the average percentage point (pp) increase in the gate-aware depth's accuracy
    at identifying the circuit versions with shortest estimated runtime.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    shared_metrics = {
        "IBM Sherbrooke": "gate_aware_depth_w=avg",
        "IBM Kyiv": "gate_aware_depth_w=avg",
        "IBM Brisbane": "gate_aware_depth_w=avg",
        "IBM Marrakesh": "gate_aware_depth_w=avg",
        "IBM Kingston": "gate_aware_depth_w=avg",
        "IBM Aachen": "gate_aware_depth_w=avg",
    }

    device_accuracies = get_identify_shortest_accuracies(datas, metrics, shared_metrics)

    trad_change_sum = 0
    multi_change_sum = 0
    num_devices = 0

    for device_name in datas.keys():
        # Gate-aware accuracy stored in last column, trad accuracy in 0th col, and multi in 1st col.
        trad_change_sum += (
            device_accuracies[device_name][-1] - device_accuracies[device_name][0]
        )
        multi_change_sum += (
            device_accuracies[device_name][-1] - device_accuracies[device_name][1]
        )
        num_devices += 1

    trad_change_avg = trad_change_sum / num_devices
    multi_change_avg = multi_change_sum / num_devices

    print("\n===Average Gate-Aware Accuracy Increase===")
    print("Average increase in percentage points (pp) of accuracy across devices")
    print(f"Relative to traditional depth:\t\t{trad_change_avg:.2f} pp")
    print(f"Relative to multi-qubit depth:\t\t{multi_change_avg:.2f} pp")


def show_multi_ties(data, metrics):
    """
    Show the number of circuits for which the version with shortest estimated runtime (correctly)
    had the shortest multi-qubit depth, but other versions also tied this depth. In other words,
    this is the number of comparisons where the wrong decision was made due to differences in
    single-qubit gates.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    print("\n===Marrakesh Multi-Qubit Accuracy Analysis===")
    identify_shortest(data, metrics, term_out=True)
