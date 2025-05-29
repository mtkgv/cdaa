"""
Calculates statistics for claims made in the paper or presentation.
"""

import logging

from utils import csvs_to_data_dict
from utils import get_min_RE_metrics
from stats import verify_runtime_estimator
from stats import calc_avg_RE_multiplier
from stats import compare_identify_shortest_accuracies
from stats import calc_avg_accuracy_increase
from stats import show_multi_ties


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

# Calculation script

# In the original data, there was one circuit comparison which raised a
# divide-by-zero warning when running identify_shortest. This causes a single
# data point to be lost, which affects results only minimally but interrupts
# the analysis print outs. We set the logging level to avoid this.
logging.basicConfig(level=logging.ERROR)

verify_runtime_estimator(r"../1_depth_runtime/data/csv/verify.csv")
get_min_RE_metrics(datas, metrics, term_out=True)
calc_avg_RE_multiplier(datas, metrics)
compare_identify_shortest_accuracies(datas, metrics)
calc_avg_accuracy_increase(datas, metrics)
show_multi_ties(datas["IBM Marrakesh"], metrics)
