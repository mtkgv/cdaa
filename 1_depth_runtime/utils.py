"""
Provides shared utility functions and classes for obtaining experiment data.
"""

from io import StringIO
import os
from datetime import datetime
import ast

import numpy as np

from qiskit.transpiler import InstructionDurations
from qiskit_ibm_runtime import QiskitRuntimeService

try:
    from qiskit.scheduler import ScheduleConfig
    from qiskit.scheduler.schedule_circuit import schedule_circuit
except:
    print(
        "Could not find qiskit.scheduler objects. Qiskit runtime scheduling will be unavailable."
    )

from bqskit.ext import bqskit_to_qiskit
from bqskit.ir.circuit import Circuit


def translate_ibm_instruction_durations(instruction_durations):
    """
    Translate a Qiskit InstructionDurations object to a duration map readable
    by drtester._estimate_runtime().

    Args:
        instruction_durations (InstructionDurations | str): An InstructionDurations object
            from a Qiskit backend, or a file path path to a text file containing the printout
            of this object.
    """

    duration_map = {}
    buffer = StringIO(newline="\n")

    # Write instruction durations to buffer for parsing
    if isinstance(instruction_durations, str):
        if not os.path.isfile(instruction_durations):
            raise RuntimeError(f"No file exists at {instruction_durations}")
        else:
            with open(instruction_durations, "r", encoding="utf-8") as file:
                for line in file:
                    buffer.write(line)

    elif isinstance(instruction_durations, InstructionDurations):
        buffer = StringIO(newline="\n")
        buffer.write(str(instruction_durations))

    # Convert each buffer line to dictionary entry
    buffer.seek(0)
    for line in buffer.readlines():

        # Import data
        line = line.strip()
        gate_name, end = line.split("(")
        end = "(" + end
        location, end = end.split(":")
        _, time, _ = end.split(" ")
        location = ast.literal_eval(location)
        time = float(time)
        if gate_name in [
            "id",
            "rx",
            "rz",
            "sx",
            "x",
        ]:
            num_qudits = 1
        elif gate_name in [
            "cz",
            "ecr",
        ]:
            num_qudits = 2
        else:
            num_qudits = -1

        # Add entry to dictionary
        if num_qudits not in duration_map.keys():
            duration_map[num_qudits] = {}
        if location not in duration_map[num_qudits].keys():
            duration_map[num_qudits][location] = {}
        if gate_name not in duration_map[num_qudits][location].keys():
            duration_map[num_qudits][location][gate_name] = time

    return duration_map


class DRTester:
    """
    A DRTester automates the process of obtaining and recording metric
    values for all circuits compiled for a single device.
    """

    def __init__(
        self,
        device_name,
        weight_map,
        compilers,
        metrics,
        run_local=False,
        inst_dur_path=None,
    ):
        """
        Construct a DRTester.
        Args:
            device_name (str): The name of the target device the circuits have been compiled
                for. Supports IBM devices only.

            weight_map (dict[str: float]): Mapping from native gate set names to gate weights.

            compilers (list[str]): The names of compilers that have been used to produce
                different optimized versions of the same circuit.

            metrics: (list[str]): The names of the metrics to evaluate for each circuit.
                Accepts:
                    - "estimated_runtime"
                    - "qiskit_scheduler_runtime"
                    - "trad_depth"
                    - "multi_qubit_depth"
                    - "gate_aware_depth"
                    - "gate_aware_depth_w=[float]"

            run_local (bool): Run the test locally, i.e. without Qiskit API access.
                If true, DRTester object:
                    - cannot calculate the "qiskit_scheduler_runtime" metric
                    - requires a file path to saved InstructionDurations for the device to calculate
                        "estimated_runtime" metric
                    (Default: False)

            inst_dur_path (str): The path to an InstructionDurations text file for the
                target device (Default: None)
        """

        self.backend_name = device_name
        self.weight_map = weight_map
        self.compilers = compilers
        self.metrics = metrics
        self.results = {}

        if run_local:
            self.duration_map = translate_ibm_instruction_durations(inst_dur_path)
        else:
            self.backend = QiskitRuntimeService().backend(device_name)

            # Build custom duration map from specified backend
            self.duration_map = translate_ibm_instruction_durations(
                self.backend.instruction_durations
            )

            # Build schedule config for verifying homebrew runtime estimator
            inst_map = self.backend.instruction_schedule_map
            meas_map = self.backend.meas_map
            dt = self.backend.dt

            self.schedule_config = ScheduleConfig(inst_map, meas_map, dt)

    def save_ibm_instruction_durations(self, instruction_durations, save_dir="./"):
        """
        Save Qiskit instruction durations in a text file.
        Args:
            instruction_durations (InstructionDurations): An InstructionDurations object
                from a Qiskit backend.

            save_dir (str): The directory to save the text file in.
        """

        now = datetime.now()
        file_path = os.path.join(
            save_dir,
            f"{self.backend_name}_inst_dur_{now.date()}_{now.strftime('%H:%M:%S')}.txt",
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(instruction_durations))

    def get_metric_vals(self, results_dict, circ_path):
        """
        Get metric values for a single circuit.
        Args:
            results_dict (dict): The results dictionary storing metric values for all
                circuits. Indexes metric values by [circuit_name][compiler][metric].

            circ_path (str): The path to the circuit QASM file.
        """

        # Get results dict keys from circuit file path
        path_list = circ_path.split(os.sep)
        circ_name = os.path.splitext(path_list[-1])[0]
        compiler = path_list[4]

        # Prepare empty nested dictionaries to recieve values
        if circ_name not in results_dict.keys():
            results_dict[circ_name] = {}
        if compiler not in results_dict[circ_name].keys():
            results_dict[circ_name][compiler] = {}

        circ = Circuit.from_file(circ_path)

        # Calculate and record values
        for metric in self.metrics:
            if metric == "trad_depth":
                value = circ.depth
            elif metric == "multi_qubit_depth":
                value = circ.multi_qudit_depth
            elif metric == "gate_aware_depth":
                value = round(self._get_gate_aware_depth(circ, self.weight_map), 4)
            elif metric.startswith("gate_aware_depth_w="):
                var_weight = ast.literal_eval(metric.split("=")[1])
                weight_map = {
                    "ecr": 1.0,
                    "cz": 1.0,
                    "rz": 0.0,
                    "sx": var_weight,
                    "x": var_weight,
                }
                value = round(self._get_gate_aware_depth(circ, weight_map), 4)
            elif metric == "estimated_runtime":
                value = f"{self._estimate_runtime(circ, self.duration_map):e}"
            elif metric == "qiskit_scheduler_runtime":
                num_qiskit_scheduler_pulses = schedule_circuit(
                    bqskit_to_qiskit(circ), self.schedule_config
                ).duration
                value = num_qiskit_scheduler_pulses * self.schedule_config.dt
            else:
                value = None

            results_dict[circ_name][compiler][metric] = value

    def _get_gate_aware_depth(self, circuit, weight_map):
        """
        Calculate gate-aware depth for a single circuit.
        Args:
            circuit (Circuit): The circuit to get the gate-aware depth of.

            weight_map (dict[str: float]): Mapping from native gate set names to
                gate weights.
        """

        qudit_depths = np.zeros(circuit.num_qudits, dtype=float)
        for op in circuit:
            increment = weight_map[op.gate.qasm_name]
            new_depth = max(qudit_depths[list(op.location)]) + increment
            qudit_depths[list(op.location)] = new_depth
        return float(max(qudit_depths))

    def _estimate_runtime(self, circuit, duration_map):
        """
        Estimate the runtime of the the circuit.
        Args:
            circuit (Circuit): The circuit to estimate the runtime of.

            duration_map (dict): A dictionary storing the gate execution times for
                all gates available on the device. Indexed by
                [num_qudits][qudit_locations][gate_name].
        """

        qudit_runtimes = np.zeros(circuit.num_qudits, dtype=float)
        for op in circuit:
            duration = duration_map[op.num_qudits][op.location][op.gate.qasm_name]
            new_runtime = max(qudit_runtimes[list(op.location)]) + duration
            qudit_runtimes[list(op.location)] = new_runtime
        return float(max(qudit_runtimes))

    def get_data(self, qasm_dir):
        """
        Get metric values for all circuits in a directory.
        Args:
            qasm_directory (str): The directory containing the circuit QASM files.
        """

        for file_name in os.listdir(qasm_dir):
            if file_name.endswith(".qasm"):
                path = os.path.join(qasm_dir, file_name)
                with open(path, encoding="utf-8") as f:
                    now = datetime.now()
                    print(
                        f"Started gathering data for {file_name} at {now.strftime('%H:%M:%S')}"
                    )
                    self.get_metric_vals(self.results, path)

    def record(self, file_path=None):
        """
        Record all metric values to CSV file.
        Args:
            file_path (str): The path to save the CSV to.
        """

        # Set up new output CSV with config options
        if file_path is None:
            now = datetime.now()
            file = open(
                f"depth_v_runtime_{now.date()}_{now.strftime('%H:%M:%S')}.csv",
                "w",
                encoding="utf-8",
            )
        else:
            file = open(file_path, "w", encoding="utf-8")

        # Record test setup
        file.write("Test Configuration\n")
        file.write(f"compilers: {self.compilers}\n")
        file.write(f"backend: {self.backend_name}\n")

        csv_labels = "circuit, "
        for compiler in self.compilers:
            for metric in self.metrics:
                csv_labels += f"{compiler}_{metric}, "
        csv_labels += "\n"
        file.write(csv_labels)

        # Record data values to CSV
        for circ_name in self.results.keys():
            result_row = f"{circ_name}, "
            for compiler in self.compilers:
                for metric in self.metrics:
                    result_row += f"{self.results[circ_name][compiler][metric]}, "
            result_row += "\n"
            file.write(result_row)
