"""
Draws figures for paper, presentation, and private analysis.
"""

import logging

from utils import csvs_to_data_dict
from figs import plot_case_study
from figs import plot_RE_boxplots
from figs import plot_identify_shortest_accuracies
from figs import plot_REs_by_weight
from figs import plot_REs_by_num_qubits
from figs import plot_device_circtype_size_grid
from figs import plot_abs_runtime_v_metrics
from figs import plot_prop_runtime_v_metrics


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

weights = [x / 100.0 for x in range(1, 101)]
metrics = [
    "multi_qubit_depth",
]
for weight in weights:
    metrics.append(f"gate_aware_depth_w={weight}")
metrics += ["gate_aware_depth_w=avg", "trad_depth", "runtime"]
num_metrics = len(metrics)

first_col = 1

# Draw script
logging.basicConfig(level=logging.WARNING)

datas = csvs_to_data_dict(
    csv_directory=r"../1_depth_runtime/data/csv/",
    exclude=[
        "verify.csv",
    ],
    n_circuits=num_circuits,
    n_device_compilers=n_device_compilers,
    n_metrics=num_metrics,
    first_col=first_col,
)

# Enforce ordering to keep architectures together
order = [
    "IBM Sherbrooke",
    "IBM Kyiv",
    "IBM Brisbane",
    "IBM Marrakesh",
    "IBM Kingston",
    "IBM Aachen",
]
datas = {device_name: datas[device_name] for device_name in order}

marrakesh_data = datas["IBM Marrakesh"]

# Paper Figures
plot_case_study(marrakesh_data, metrics, "gate_aware_depth_w=avg")
plot_RE_boxplots(datas, metrics)
plot_identify_shortest_accuracies(datas, metrics)
plot_REs_by_weight(datas, metrics, weights)

# Presentation / Question Figures
plot_REs_by_num_qubits(datas, metrics)
plot_device_circtype_size_grid(datas, metrics)
plot_abs_runtime_v_metrics(marrakesh_data, metrics)
plot_prop_runtime_v_metrics(marrakesh_data, metrics)
