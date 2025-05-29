"""
Provides shared utility functions for analyzing experiment data.
"""

import csv
import os
import math
import itertools as it
import logging

import numpy as np


_logger = logging.getLogger(__name__)


def csv_to_array(csv_path, n_circuits, n_compilers, n_metrics, first_col):
    """
    Converts data in CSVs to array for processing.

    Returns (np.array): Device metric values organized by:
        - Axis 0: Circuit
        - Axis 1: Compiler (or compiled version of circuit)
        - Axis 2: Metric

    Args:
        csv_path (str): The path to the data CSV file.

        n_circuits (int): The number of circuits in the CSV file.

        n_compilers (int): The number of compilers in the CSV file.

        n_metrics (int): The numer of metrics in the CSV file.

        first_col (int): The first column of the CSV containing data values,
            i.e. not a header column.
    """

    # Initialize return array with matching axis dimensions
    data = np.zeros((n_circuits, n_compilers, n_metrics))

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        circuit_position = -1
        for row in reader:
            # Only start recording once we've passed the header
            if len(row) < n_metrics * n_compilers or row[0] == "circuit":
                continue
            circuit_position += 1
            # Reads the row as blocks of metrics belonging to the same compiler
            for compiler_position, i in enumerate(
                range(first_col, len(row) - 1, n_metrics)
            ):
                data[circuit_position][compiler_position] = np.array(
                    row[i : i + n_metrics]
                )

    return data


def csvs_to_data_dict(
    csv_directory, exclude, n_circuits, n_device_compilers, n_metrics, first_col
):
    """
    Converts all CSVs in a directory to arrays, indexed by device name.

    Returns (dict[str: np.array]): Mapping from device name to device raw data.

    Args:
        csv_directory (str): The directory containing the CSV files.

        exclude (list[str]): File names to exclude from reading.

        n_circuits (int): The number of circuits in each CSV file.

        n_device_compilers (dict[str: int]): Mapping from device name to number of compilers in
            the device's CSV file.

        n_metrics (int): The number of metrics in each CSV file.

        first_col (int): The first column of each CSV containing data values,
            i.e. not a header column.
    """

    data_dict = {}

    for file_name in os.listdir(csv_directory):
        if file_name.endswith(".csv") and file_name not in exclude:
            csv_path = os.path.join(csv_directory, file_name)

            # Different devices have different numbers of compilers to
            # compare circuit versions of
            device_name = os.path.splitext(file_name)[0]
            n_compilers = n_device_compilers[device_name]

            # Nicely format keys to use in figures
            label = device_name.split("_")
            label[0] = label[0].upper()
            label[1] = label[1].capitalize()
            label = label[0] + " " + label[1]

            data_dict[label] = csv_to_array(
                csv_path=csv_path,
                n_circuits=n_circuits,
                n_compilers=n_compilers,
                n_metrics=n_metrics,
                first_col=first_col,
            )

    return data_dict


def get_proportional_changes(data):
    """
    Calculates the propotional changes in all metrics between all pairs of compiled versions
    of the same circuit.

    Returns (np.array): Propotional changes in all metrics between every two compiled
            versions of the same circuit. They are organized by:

                - Axis 0: Pairs of compiled versions of the same circuit. They are ordered
                    in blocks of nC2 pairs, where the i-th block contains the nC2 pairs of
                    compiled versions for the i-th circuit in the rows of the CSV. The nC2
                    pairs within each block are ordered in (0,1), (0,2), ... (n-1, n) order,
                    where each number corresponds to the compiler's position in the column
                    blocks of the CSV.

                - Axis 1: Metrics. They are ordered as in the CSV columns.

    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.
    """

    # Initialize return array with matching axis dimensions
    num_comparisons = data.shape[0] * math.comb(data.shape[1], 2)
    prop_changes = np.zeros((num_comparisons, data.shape[2]))

    write_row = 0
    # Iterate over circuits
    for circuit_position in range(data.shape[0]):
        # Iterate over pairs of compilers to compare
        for compiler_position_1, compiler_position_2 in it.combinations(
            range(data.shape[1]),
            2,
        ):
            for metric_position in range(data.shape[2]):

                # Get same metric from both compilers
                baseline = data[circuit_position][compiler_position_1][metric_position]
                comparison = data[circuit_position][compiler_position_2][
                    metric_position
                ]

                proportional_change = (comparison / baseline) - 1.0

                prop_changes[write_row][metric_position] = proportional_change

            # New row when we change circuits or 2 compiled versions we're comparing
            write_row += 1

    return prop_changes


def get_percent_REs(prop_changes):
    """
    Calculates the absolute errors and percent relative errors (%RE) for each pair
    of compiled circuit versions.

    In such a pair, the proportional change in runtime is the true value we wish to know,
    for which proportional changes in depth are an approximation. This function calculates
    the error in that approximation for all pairs and all depth metrics:

        absolute error = |proportional depth change - proportional runtime change|

        %RE = absolute error / |proportional runtime change| * 100

    Returns (np.array): Errors for each compiled circuit version organized by:
        - Axis 0: Compiled circuit version pairs.
        - Axis 1: Errors
            - Cols 0 to len(metrics)-2 are absolute error in multi, gate-aware, trad order.
            - Cols len(metrics)-1 to 2*(len(metrics)-1)-1 are %RE in same order.

    Args:
        prop_changes (np.array): Propotional changes in all metrics between every two
            compiled versions of the same circuit. See get_proportional_changes for
            array structure.
    """

    # Initialize return array with matching axis dimensions
    percent_REs = np.zeros((prop_changes.shape[0], (prop_changes.shape[1] - 1) * 2))

    for pair_position in range(prop_changes.shape[0]):

        # Stop 1 early because final error would be runtime against itself
        for metric_position in range(prop_changes.shape[1] - 1):

            depth_change = prop_changes[pair_position][metric_position]
            runtime_change = prop_changes[pair_position][-1]

            abs_error = abs(depth_change - runtime_change)

            # Handles divide by 0 cases
            if runtime_change == 0 and depth_change == 0:
                percent_rel_error = 0
            elif runtime_change == 0 and depth_change != 0:
                _logger.warning(
                    "Unable to calculate relative error: |est - true|/|true| = |%f-%f|/|%f|",
                    depth_change,
                    runtime_change,
                    runtime_change,
                )
            else:
                percent_rel_error = abs_error / abs(runtime_change) * 100

            percent_REs[pair_position][metric_position] = abs_error
            percent_REs[pair_position][
                metric_position + (prop_changes.shape[1] - 1)
            ] = percent_rel_error

    return percent_REs


def get_RE_quantiles(data):
    """
    Calculates the 0-4th quartile absolute errors and %REs for each metric over
    all pairwise comparisons between compiled circuit versions.

    Each depth metric yields different abs. and rl. errors for different pairs of compiled circuit
    versions. To determine which is 'typically' best, we need to compare the distribution of these
    errrors across all pairs. The median (q2) provides a single measure of center, and the quartiles
    provide additional information about spread.

    Returns (np.array): Error quantiles, organized by:
        - Axis 0: Errors by metric, using same structure as get_percent_REs output does for axis 1
        - Axis 1: Quantiles (q0 (min), q1, q2, q3, q4 (max))

    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.
    """

    prop_changes = get_proportional_changes(data)
    percent_REs = get_percent_REs(prop_changes)

    # Switch percent_REs axes to:
    #   - Axis 0: Errors (see percent_REs for array structure)
    #   - Axis 1: Compiled circuit version pairs
    new_view = np.moveaxis(percent_REs, [0, 1], [1, 0])

    # Initialize return array with matching axis dimensions
    num_summary_stats = 5
    quantiles = np.zeros((new_view.shape[0], num_summary_stats))

    # Get summary stats for each row in axis 0, i.e. for each metric
    for i in range(new_view.shape[0]):

        q0 = np.quantile(new_view[i], 0.0)
        q1 = np.quantile(new_view[i], 0.25)
        q2 = np.quantile(new_view[i], 0.5)
        q3 = np.quantile(new_view[i], 0.75)
        q4 = np.quantile(new_view[i], 1.0)

        quantiles[i] = np.array([q0, q1, q2, q3, q4])

    return quantiles


def get_min_RE_metrics(datas, metrics, manual_weights_only=False, term_out=False):
    """
    Find the metric with the smallest median %RE for each device.

    Returns: (dict[str: str]): Mapping from device name to smallest median %RE metric name.

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.

        manual_weights_only (bool): If True, only considers the gate-aware metrics with manually
            set_weights (Default: False)

        term_out (bool): If True, prints the minimum median %RE and the name of the metric which
            produced it for each device. (Default: False)
    """

    if term_out:
        print("\n===Accuracies in Predicting Relative Runtime Changes===")
        print(
            "Accuracy := | relative change in depth - relative change in runtime | "
            + "/ | relative change in runtime | * 100"
        )

    min_metrics = {}
    for device_name, device_data in datas.items():
        quantiles = get_RE_quantiles(device_data)

        # Switch quantiles axes to:
        #   - Axis 0: Quantiles (q0, q1, q2, q3, q4)
        #   - Axis 1: Metrics (see get_RE_quantiles for array structure)
        quantiles_new_view = np.moveaxis(quantiles, [0, 1], [1, 0])

        # Find the weight minimizing the q2 %RE
        min_q2 = np.inf
        min_metric_pos = None

        # Stops iteration 2 early to avoid considering gate-aware and trad depth
        if manual_weights_only:
            max_metric_pos = quantiles_new_view.shape[1] - 2
        else:
            max_metric_pos = quantiles_new_view.shape[1]

        # Start halfway through row because absolute errors are stored in the first half.
        # We include multi-qubit and trad depth to verify that gate-aware does in fact
        # produce the smallest error, and because multi-qubit depth is equivalent to w=0.
        for metric_pos in range(len(metrics) - 1, max_metric_pos):
            q2 = quantiles_new_view[2][metric_pos]
            if q2 < min_q2:
                min_q2 = q2
                # +1 offset accounts for difference in len(metrics) and the # of metrics in the
                # quantile data. Specifically, the quantiles do not list an error for runtime since
                # it acts as ground truth. Thus the quantile data has 1 fewer column in each half,
                # and we add 1 to get the correct metric back
                min_metric_pos = (metric_pos + 1) % len(metrics)

        min_metric = metrics[min_metric_pos]
        if term_out:
            print(f"+++{device_name}+++")
            print(f"Using traditional depth: \t\t\t\t{quantiles_new_view[2][-1]:e} %")
            print(
                f"Using multi-qubit depth: \t\t\t\t{quantiles_new_view[2][len(metrics)-1]:e} %"
            )
            print(
                f"Using gate_aware_depth_w=avg: \t\t\t\t{quantiles_new_view[2][-2]:e} %"
            )
            print(f"Using best metric ({min_metric}): \t\t{min_q2:e} %")

        min_metrics[device_name] = min_metric

    return min_metrics


def get_device_circtype_data(data, metrics, type_pos):
    """
    Get percent relative errors (%REs) belonging only to circuits of the given type for
    one device.

    Returns: (np.array): %REs, indexed by:

            - Axis 0: Compiled circuit version pairs for circuit at position circ_pos
            - Axis 1: Metrics in trad, multi, gate_aware order
            - Axis 2: Circuit position, in same order as type_pos

    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.

        metrics (list[str]): The names of the metrics to associate with the data values.

        type_pos (list[int]]): List of positions of circuits of a given type in the CSV file,
            measured from top to bottom.
    """

    device_circtype_data = None

    # Get all %REs for given device
    prop_changes = get_proportional_changes(data)
    percent_REs = get_percent_REs(prop_changes)

    # Get the number of pairwise comparisons on this device
    comparisons_per_circuit = math.comb(data.shape[1], 2)

    # Collect only %REs belonging to circuits of the given type
    circ_datas = []
    for circ_pos in type_pos:
        first_comparison_row = circ_pos * comparisons_per_circuit
        last_comparison_row = ((circ_pos + 1) * comparisons_per_circuit) - 1
        comparison_rows = [
            j for j in range(first_comparison_row, last_comparison_row + 1)
        ]

        # Non-contiguous slice indexing %REs for the circuit at the given position by:
        #   - Axis 0: Compiled circuit version pairs for circuit at position circ_pos
        #   - Axis 1: Metrics in trad, multi, gate_aware order
        new_data = percent_REs[np.ix_(comparison_rows, [-1, len(metrics) - 1, -2])]

        circ_datas.append(new_data)

    # Stacking combines %REs from each circuit along a new axis indexed by circuit position.
    # Together, device_circtype_data's axes are:
    #   - Axis 0: Compiled circuit version pairs for circuit at position circ_pos
    #   - Axis 1: Metrics in trad, multi, gate_aware order
    #   - Axis 2: Circuit position, in same order as type_pos
    device_circtype_data = np.stack(circ_datas, axis=-1)

    return device_circtype_data


def convert_device_circtype_data_to_quantiles(device_circtype_data):
    """
    Converts percent relative errors (%REs), belonging only to circuits of the given type,
    to %RE quantiles for one device.

    Returns (np.array): %RE quantiles. indexed by:
        - Axis 0: Metrics in trad, multi, gate_aware order
        - Axis 1: Circuit position, in same order as device_circtype_data
        - Axis 2: Quantiles, in ascending order from q0 to qn

    Args:
        device_circtype_data (np.array): The %REs belonging only to circuits of the given
            type for one device. See get_device_circtype_data for array structure.
    """

    # Create empty target array
    num_metrics = device_circtype_data.shape[1]
    num_quantiles = 5
    num_circs_in_type = device_circtype_data.shape[2]

    device_circtype_quantiles = np.zeros(
        (num_metrics, num_circs_in_type, num_quantiles)
    )

    # Switch device_circtype_data axes to index %REs by:
    #   - Axis 0: Metrics in trad, multi, gate_aware order
    #   - Axis 1: Circuit position, in same order as type_pos
    #   - Axis 2: Compiled circuit version pairs for circuit at position circ_pos
    # This allows us to easily take quantiles over pairwise comparisons of the same circuit
    new_view = np.moveaxis(device_circtype_data, [0, 1, 2], [2, 0, 1])

    # For each metric and circuit type, obtain quantiles over all compiled circuit version pairs
    for metric_pos in range(new_view.shape[0]):
        for circ_pos_pos in range(new_view.shape[1]):

            # Obtain quantiles
            quantiles = []
            for i in range(num_quantiles):
                quantile = np.quantile(
                    new_view[metric_pos][circ_pos_pos], 1 / (num_quantiles - 1) * i
                )
                quantiles.append(quantile)

            # Write computed quantiles row to target array
            # Essentially, this replaces the individual %RE values for each pairwise comparison
            # with the different quantiles over those comparisons.
            device_circtype_quantiles[metric_pos][circ_pos_pos] = quantiles

    return device_circtype_quantiles


def get_device_circtype_quantiles(datas, metrics, type_pos_map):
    """
    Obtains percent relative error (%RE) quantiles for each device and circuit type and, within
    each type, orders the quantile values in the same order given by the circuit type position
    map. It is essentially a bulk automation of get_device_circtype_data and
    convert_device_circtype_data_to_quantiles when multiple devices and circuit types are
    desired together.

    For each (device, circ_type) pair, a given quantile's values are given in the same order as
    the corresponding circuits they come from in the type_pos_map. In context these positions
    represent different numbers of qubits, which allows visualization of the trends in %REs as size
    changes.

    The disaggregation by device and circuit type allows for flexible plotting of a single case
    study circuit type, or a survey grid of many possible case studies.

    Returns: (dict[tuple[str, str]: np.array): Mapping from (device_name, circ_type_name) to %RE
        quantiles. In the array, the %RE quantiles are indexed by:

            - Axis 0: Metrics in trad, multi, gate_aware order
            - Axis 1: Quantiles, in ascending order from q0 to qn
            - Axis 2: Circuit positions, with the same size and order as type_pos_map

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.

        type_pos_map (dict[str: list[int]]): Mapping from circuit type name to list of
            positions of circuits of that type in the CSV file, measured from top to bottom.
    """

    all_device_circtype_quantiles = {}

    # For each device and circuit type pair, obtain quantiles
    for device_name, circ_type_name in it.product(datas.keys(), type_pos_map.keys()):

        device_data = datas[device_name]
        type_pos = type_pos_map[circ_type_name]

        device_circtype_data = get_device_circtype_data(device_data, metrics, type_pos)
        device_circtype_quantiles = convert_device_circtype_data_to_quantiles(
            device_circtype_data
        )

        # Switch device_circtype_quantiles axes to index %REs by:
        #   - Axis 0: Metrics in trad, multi, gate_aware order
        #   - Axis 1: Quantiles, in ascending order from q0 to qn
        #   - Axis 2: Circuit position, in same order as type_pos_map
        # This allows us to easily extract rows of quantiles, in order of circuit size, for plotting
        new_view = np.moveaxis(device_circtype_quantiles, [0, 1, 2], [0, 2, 1])

        # Link data with device and circuit type
        all_device_circtype_quantiles[(device_name, circ_type_name)] = new_view

    return all_device_circtype_quantiles


def get_REs_by_num_qubits(datas, metrics, size_pos_map):
    """
    Organizes the percent relative errors (%REs) of all pairwise comparisons by the
    number of qubits in the circuit and the metric used for comparison. Facilitates visualization
    of overall trend in %RE by circuit size for different metrics.

    Returns (dict[int: np.array]): Mapping from circuit size to %REs for circuit comparisons
        of that size. In the array, the %REs are organized by:
            - Axis 0: %REs in trad, multi, gate_aware order
            - Axis 1: Compiled circuit version pairs for circuits of given size, blocked
                by device

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.

        size_pos_map (dict[int: list[int]]): Mapping from circuit size to list of
            positions of circuits of that size in the CSV file, measured from top to bottom.
    """

    REs_by_num_qubits = {num_qubits: None for num_qubits in size_pos_map.keys()}

    for device_data in datas.values():

        # Collect %REs for each device
        prop_changes = get_proportional_changes(device_data)
        percent_REs = get_percent_REs(prop_changes)

        # Get the number of pairwise comparisons on this device
        comparisons_per_circuit = math.comb(device_data.shape[1], 2)

        # For each size, collect all %REs from circuits of that size
        for num_qubits in size_pos_map.keys():

            new_num_qubits_data = None

            # Obtain row indices for circuits of the given size
            comparison_rows = []
            size_pos = size_pos_map[num_qubits]
            for circ_pos in size_pos:
                first_comparison_row = circ_pos * comparisons_per_circuit
                last_comparison_row = ((circ_pos + 1) * comparisons_per_circuit) - 1
                comparison_rows += [
                    j for j in range(first_comparison_row, last_comparison_row + 1)
                ]

            # Using row indices, obtain non-contiguous slice containing the %REs for each metric
            #   - Axis 0: Compiled circuit version pairs for circuits of given size
            #   - Axis 1: %REs in trad, multi, gate_aware order
            new_num_qubits_data = percent_REs[
                np.ix_(comparison_rows, [-1, len(metrics) - 1, -2])
            ]

            # Switch new_num_qubits_data axes to:
            #   - Axis 0: %REs in trad, multi, gate_aware order
            #   - Axis 1: Compiled circuit version pairs for circuits of given size
            new_view = np.moveaxis(new_num_qubits_data, [0, 1], [1, 0])

            # Append data from this device and size to existing data from otherdevices, if necessary
            if REs_by_num_qubits[num_qubits] is None:
                REs_by_num_qubits[num_qubits] = new_view
            else:
                temp = np.append(REs_by_num_qubits[num_qubits], new_view, axis=1)
                REs_by_num_qubits[num_qubits] = temp

    return REs_by_num_qubits


def identify_shortest(data, metrics, term_out=False):
    """
    Calculates each depth metric's accuracy at identifying the circuit version(s) with
    the shortest estimated runtime for all possible subsets of circuit versions.

    Returns (np.array): Accuracies organized by:
        - Axis 0: Comparison size, or the number of compiled versions compared at once. Ranges
            from 2 to the number of compilers.
        - Axis 1: Depth metric
        - Axis 2: Correct, total, and percent correct comparisons (accuracy)

    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.

        metrics (list[str]): The names of the metrics to associate with the data values.

        term_out (bool): If True, prints the total number of total comparisons, incorrect
            comparisons, and incorrect comparisons for which the correct shortest circuits were
            a subset of the identified shortest circuits. However, this is only printed when
            the maximum number of circuits are being compared simultaneously using multi-qubit
            depth.
    """

    # Switch data axes to:
    #   - Axis 0: Metrics
    #   - Axis 1: Circuits
    #   - Axis 2: Compilers
    # We will traverse the data in this order, and the new view makes this natural
    new_view = np.moveaxis(data, [0, 1, 2], [1, 2, 0])

    # Initialize return array with matching axis dimensions
    comparison_sizes = range(2, new_view.shape[2] + 1)
    num_comparisons = len(comparison_sizes)
    num_depth_metrics = new_view.shape[0] - 1
    num_comparison_stats = 3

    accuracies = np.zeros((num_comparisons, num_depth_metrics, num_comparison_stats))

    # Iterates over the number of circuits we compare at once
    # new_view.shape[2] is the number of compilers, which determines
    # the maximum number of circuits we can compare at once
    for comparison_position, comparison_size in enumerate(comparison_sizes):
        _logger.info("Finding accuracies for %d-circuit comparisons.", comparison_size)

        # Iterates over the depth metrics
        # new_view.shape[0] is the number of metrics, and we subtract 1
        # because we don't want to compare the last metric, runtime, with itself
        for metric_position, metric in zip(range(num_depth_metrics), metrics[:-1]):
            _logger.info("Finding %s accuracy.", metric)

            # Reset totals for each depth metric
            total_comparisons = 0
            correct_comparisons = 0
            incorrect_comparisons = 0
            correct_is_subset = 0

            # Iterates over the circuits
            # new_view.shape[1] is the number of circuits
            for circuit_position in range(new_view.shape[1]):

                # Create all n-choose-k subsets of compiled circuit versions
                # to compare using the specified depth metric
                for combination in it.combinations(
                    zip(
                        new_view[metric_position][circuit_position],
                        new_view[-1][circuit_position],
                    ),
                    comparison_size,
                ):
                    # Creates a new array from the subset, with compilers by column,
                    # depths in row 0, and runtimes in row 1
                    subarray = np.array(combination).transpose()

                    # Get min depth and runtime for the nCk subset
                    depth_min = np.min(subarray[0])
                    runtime_min = np.min(subarray[1])

                    # Collect a list of all indices matching the mins
                    min_depth_idx = []
                    min_runtime_idx = []

                    for j, value in enumerate(np.nditer(subarray[0])):
                        if value == depth_min:
                            min_depth_idx.append(j)

                    for j, value in enumerate(np.nditer(subarray[1])):
                        if value == runtime_min:
                            min_runtime_idx.append(j)

                    # Compare indices of min depth metric and runtime
                    if min_depth_idx == min_runtime_idx:
                        correct_comparisons += 1
                    else:
                        incorrect_comparisons += 1
                        if set(min_runtime_idx).issubset(set(min_depth_idx)):
                            correct_is_subset += 1

                    total_comparisons += 1

            if (
                term_out
                and metric == "multi_qubit_depth"
                and comparison_size == max(comparison_sizes)
            ):
                print(f"Total circuit comparisons:\t\t\t\t\t{total_comparisons}")
                print(
                    f"Incorrect circuit comparisons:\t\t\t\t\t{incorrect_comparisons}"
                )
                print(
                    "Incorrect comparison due to tie with "
                    + f"true fastest version:\t{correct_is_subset}"
                )

            # Log results
            _logger.info("Correct comparisons: %d", correct_comparisons)
            _logger.info("Total comparisons: %d", total_comparisons)
            _logger.info(
                "Correct %%: %f", correct_comparisons / total_comparisons * 100
            )

            # Record results
            current_row = accuracies[comparison_position][metric_position]
            current_row[0] = correct_comparisons
            current_row[1] = total_comparisons
            current_row[2] = correct_comparisons / total_comparisons * 100

    return accuracies


def get_identify_shortest_accuracies(datas, metrics, target_metrics):
    """
    Gets the accuracy of traditional, multi-qubit, and a target depth metric
    for each device.

    Returns (dict[str: list[float]]): Mapping from device names to accuracies, which
        are organized by [trad, multi-qubit, gate-aware].

    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.

        target_metrics (dict[str: str]): Mapping from device names to target depth
            metric names.
    """

    device_accuracies = {key: [] for key in datas.keys()}

    for device_name, device_data in datas.items():

        accuracies = identify_shortest(device_data, metrics)

        # Data we want is in accuracies[x][y][z]
        # x=-1: last column is the largest comparison size available
        # y=?: 0 for multi=qubit, -1 for trad. For gate-aware, we use the target metric name
        # to find the index
        # z=-1: last column is % correct comparisons

        target_metric = target_metrics[device_name]
        target_metric_index = metrics.index(target_metric)

        device_accuracies[device_name] = [
            accuracies[-1][-1][-1],  # Trad accuracy
            accuracies[-1][0][-1],  # Multi-qubit accuracy
            accuracies[-1][target_metric_index][-1],  # Gate-aware accuracy
        ]

    return device_accuracies
