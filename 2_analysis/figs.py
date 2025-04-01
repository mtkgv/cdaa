import copy
import numpy as np
import logging

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import csvs_to_data_dict
from utils import get_proportional_changes
from utils import get_percent_REs
from utils import get_RE_quantiles
from utils import get_min_RE_metrics
from utils import get_identify_shortest_accuracies


"""
Draws the figures used in the results section.
"""


def plot_median_error_marrakesh_ex(data, metrics, best_metric):
    """
    Plots the median errors for the case-study comparison of the TKET- and Qiskit-compiled
    versions of the QAOA16.
    
    Args:
        data (np.array): The metric values for all circuit versions compiled for a single device.

        metrics (list[str]): The names of the metrics to associate with the data values.

        best_metric (str): The name of the lowest %RE metric for the device.
    """

    # Recreate the new view used internally by get_RE_quantiles()
    prop_changes = get_proportional_changes(data)
    percent_REs = get_percent_REs(prop_changes)

    # Switch percent_REs axes to:
    #   - Axis 0: Metrics
    #   - Axis 1: Compiled circuit version pairs
    new_view = np.moveaxis(percent_REs, [0,1],[1,0])

    # Find the circuit and compiler pair above which 50% of errors lie.
    # Note that we cannot find median directly, since it is an average of the two middle values.
    # Hence we use the lower of those two.
    sorted_trad_errors = np.sort(new_view[-1])
    pair_position = np.where(new_view[-1] == sorted_trad_errors[44])[0][0]

    # Find the prop changes for that circuit and compiler pair.
    # In prop change data, circuit and compiler pairs are rows, so we use
    # circ_comp_pair_pos for the row index.
    multi_prop_change = prop_changes[pair_position][0] # Multi in col 0
    best_metric_prop_change = prop_changes[pair_position][metrics.index(best_metric)] # Best weight in same col as pos in metric list
    trad_prop_change = prop_changes[pair_position][-2] # Trad in col -2
    runtime_prop_change = prop_changes[pair_position][-1] # Runtime in col -1

    print(f"Runtime prop change: {runtime_prop_change}")

    fig, ax = plt.subplots(dpi=300)
    plt.rcParams['text.usetex'] = True

    # Add bars
    x = ["traditional\ndepth", "multi-qubit\ndepth", "gate-aware\ndepth\n($w_s=0.53$)",]
    y = [trad_prop_change, multi_prop_change, best_metric_prop_change,]

    bar_list = ax.bar(x, y, alpha=0.5)

    # Set colors
    for i in range(2):
        bar_list[i].set_color("grey")
    bar_list[2].set_color("green")

    # Add bar labels
    for i in range(len(x)):
        plt.text(i, y[i] / 2, f"{y[i]:#.4g}", ha="center")

    # Add line for ground truth runtime change
    ax.plot([-0.5, 2.5], [runtime_prop_change, runtime_prop_change], "--", color="green", label=rf"True runtime change ($\Delta R$): {runtime_prop_change:#.4g}")

    # Add axes & legend
    ax.set_ylabel(r"Proportional change relative to TKET ($\Delta D$)")
    ax.legend()

    plt.savefig("./out/marrakesh-ex.png", bbox_inches="tight")
    plt.rcParams['text.usetex'] = False
    plt.close()


def plot_RE_quantiles(datas, metrics, weights):
    """
    Plots the percent relative error (%RE) quantiles of each metric on all devices.
    
    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.

        weights (list[float]): The weights used for gate-aware depth. Used to plot %RE by
            weight for gate-aware depth.
    """

    fig, ax = plt.subplots(figsize=(8,4), dpi=300)

    legend_marker_handles = []
    legend_labels = []

    num_devices = len(datas)

    # Draw multiple lines
    for i, (device_name, device_data) in enumerate(datas.items()):

        quantiles = get_RE_quantiles(device_data)

        # Switch quantiles axes to:
        #   - Axis 0: Quantiles (q1, q2, q3)
        #   - Axis 1: Metrics (see get_RE_quantiles for array structure)
        new_view = np.moveaxis(quantiles, [0, 1], [1, 0])

        # Create new %RE quantile array to manipulate for graphing
        # Slice from 0.5*new_view.shape[1] to eliminate first half, where abs errors are stored
        [q1, q2, q3] = [copy.deepcopy(new_view[j][int(0.5*new_view.shape[1]):]) for j in range(3)]

        # For each quantile, duplicate multi-qubit %RE, since it is equal to gate_aware_depth_w=0 and
        # we will therefore want to plot that point for both multi-qubit and gate-aware depth
        # Multi-qubit %RE is stored in 0th column.
        [q1, q2, q3] = [np.insert(qn, 1, qn[0]) for qn in [q1, q2, q3]]

        # Rotate trad %RE to first column
        [q1, q2, q3] = [np.roll(qn, 1) for qn in [q1, q2, q3]]

        x = [-2*num_devices, -1*num_devices, 0] + weights

        # Add gate-aware plots
        med, = ax.plot(x[2:],q2[2:], '-')
        mid50 = ax.fill_between(x[2:],q1[2:],q3[2:], alpha=0.2)

        med_color = med.get_color()

        # Add legend entries   
        legend_marker_handles.append((med, mid50))
        legend_labels.append(device_name)

        # "Box" plots for trad and multi depth

        x_group_range = 0.25
        gap = 0.05

        # Add trad "box" plot
        
        # Plot line over mid 50% range
        x_pos = -2*(x_group_range+gap)+(x_group_range*i/num_devices)
        ax.vlines(
            x = x_pos,
            ymin = q1[0],
            ymax = q3[0],
            color = med_color,
            alpha = 0.2,
        )
        # Plot dot over median
        ax.plot([x_pos], [q2[0]], '.', color=med_color)

        # Annotate Marrakesh example
        if device_name == "IBM Marrakesh":
            width = 0.04
            height = 25
            box = Rectangle(
                (x_pos-0.5*width, q2[0]-0.43*height),
                width,
                height,
                fill=False,
                edgecolor="black",
            )
            ax.add_patch(box)
            ax.annotate(
                "QAOA16 example", 
                xy=(x_pos+0.01, q2[0]+0.57*height),
                xytext=(x_pos+0.03, q2[0]*7),
                arrowprops=dict(
                    facecolor='black',
                    shrink=0.1,
                    width=0.1,
                    headwidth=3,
                    headlength=3,
                )
            )

        # Add multi "box" plot
        
        # Plot line over mid 50% range
        x_pos = -1*(x_group_range+gap)+(x_group_range*i/num_devices)
        ax.vlines(
            x = x_pos,
            ymin = q1[1],
            ymax = q3[1],
            color = med_color,
            alpha = 0.2,
        )
        # Plot dot over median
        ax.plot([x_pos], [q2[1]], '.', color=med_color)
    
    # Add axes and legend

    # Create numbered x-ticks (primary)
    ax.set_xlabel("\nDepth")
    ax.set_xticks(
        [x/5 for x in range(0,6)]
    )

    # Create invisible x-ticks for text metric labels (secondary)
    sec_ax = ax.secondary_xaxis(location=0)
    sec_ax.set_xticks(
        [-0.4958, -0.1958, 0.5],
        labels = ["\nTraditional", "\nMulti-qubit", "\nGate-aware (by weight)"]
    )
    sec_ax.tick_params(axis="x", color="white")

    ax.set_ylabel(r"% Relative error (%RE)") 
    ax.set_yscale("log")
    ax.legend(
        tuple(legend_marker_handles), legend_labels
    )

    plt.savefig("./out/RE-quantiles.png", bbox_inches="tight")
    plt.close()


def plot_identify_shortest_accuracies(datas, metrics):
    """
    Plots each depth metric's accuracy at identifying the circuit version(s) with
    the shortest estimated runtime for each device.
    
    Args:
        datas (dict[str: np.array]): Mapping from device names to device raw data.

        metrics (list[str]): The names of the metrics to associate with the data values.
    """

    best_metrics = get_min_RE_metrics(datas, metrics)
    device_accuracies = get_identify_shortest_accuracies(datas, metrics, best_metrics)

    fig, ax = plt.subplots(dpi=300)  

    # Create bar groups
    depth_labels = ("Traditional", "Multi-qubit", "Gate-aware")
    x = np.arange(len(depth_labels)) # Label locations
    width = 0.14  # Bar widths
    multiplier = 0

    for device_name, accuracy in device_accuracies.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, accuracy, width, label=device_name)
        ax.bar_label(rects, fmt="{:#.4g}", padding=-30, rotation="vertical", color="white")
        multiplier += 1
    
    # Add axes and legend
    ax.set_ylabel(r"% Correct identifications")
    ax.set_xticks(x + 2.5*width, depth_labels)
    ax.legend(
        ncols=3,
        bbox_to_anchor=(0.5,1.17),
        loc="upper center"
    )

    plt.savefig("./out/identify-shortest-accuracies.png", bbox_inches="tight")
    plt.close()


# Configuration
num_circuits = 15

eagle_compilers = ["sabre0330", "sqgm", "qiskit141"]
heron_compilers = ["sabre0330", "sqgm", "tket", "qiskit141"]
num_eagle_compilers = len(eagle_compilers)
num_heron_compilers = len(heron_compilers)
n_device_compilers = {
    "ibm_sherbrooke": num_eagle_compilers,
    "ibm_kyiv": num_eagle_compilers,
    "ibm_brisbane": num_eagle_compilers,
    "ibm_marrakesh": num_heron_compilers,
    "ibm_kingston": num_heron_compilers,
    "ibm_aachen": num_heron_compilers,
}

weights = [x/100.0 for x in range(1,101)]
metrics = ["multi_qubit_depth",] 
for weight in weights:
    metrics.append(f"gate_aware_depth_w={weight}")
metrics += ["trad_depth", "runtime" ]
num_metrics = len(metrics)

first_col = 1

datas = csvs_to_data_dict(
    csv_directory = r"../1_depth_runtime/data/csv/",
    exclude = ["verify.csv",],
    n_circuits = num_circuits,
    n_device_compilers = n_device_compilers,
    n_metrics = num_metrics,
    first_col = first_col,
)


# Figures Script
logging.basicConfig(level=logging.WARNING)

marrakesh_data = datas["IBM Marrakesh"]
plot_median_error_marrakesh_ex(marrakesh_data, metrics, "gate_aware_depth_w=0.53")

plot_RE_quantiles(datas, metrics, weights)
plot_identify_shortest_accuracies(datas, metrics)