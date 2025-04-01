from io import StringIO
import os
from datetime import datetime
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.scheduler import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit

from bqskit.ext import bqskit_to_qiskit
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import RZGate


"""
Implements the DRTester class.
"""


class DRTester():
    """
    A DRTester automates the process of obtaining and recording metric
    values for all circuits compiled for a single device.
    """

    def __init__(self, device_name, compilers, metrics):
        """
        Construct a DRTester.
        Args:
            device_name (str): The name of the target device the circuits have been compiled
                for. Supports IBM devices only.

            compilers (list[str]): The names of compilers that have been used to produce
                different optimized versions of the same circuit.

            metrics: (list[str]): The names of the metrics to evaluate for each circuit.
                Accepts:
                    - "estimated_runtime"
                    - "qiskit_scheduler_runtime"
                    - "trad_depth"
                    - "multi_qubit_depth"
                    - "gate_aware_depth_w=[float]"
        """

        self.backend_name = device_name
        self.backend = QiskitRuntimeService().backend(device_name)

        # Build schedule config for verifying homebrew runtime estimator
        inst_map = self.backend.instruction_schedule_map
        meas_map = self.backend.meas_map
        dt = self.backend.dt

        self.schedule_config = ScheduleConfig(inst_map, meas_map, dt)
        
        # Build custom duration map from specified backend
        self.duration_map = self._translate_ibm_instruction_durations(self.backend.instruction_durations)

        self.compilers = compilers
        self.metrics = metrics
        self.results = {}


    def save_ibm_instruction_durations(self, instruction_durations, save_dir="./"):
        """
        Save Qiskit instruction durations in a text file.
        Args: 
            instruction_durations (InstructionDurations): An InstructionDurations object
                from a Qiskit backend.
            
            save_dir (str): The directory to save the text file in.
        """
        
        now = datetime.now()
        file_path = os.path.join(save_dir, f"{self.backend_name}_inst_dur_{now.date()}_{now.strftime('%H:%M:%S')}.txt")
        with open(file_path, "w") as f:
            f.write(str(instruction_durations))


    def _translate_ibm_instruction_durations(self, instruction_durations):
        """
        Translate a Qiskit InstructionDurations object to a duration map readable
        by _estimate_runtime.

        Args:
            instruction_durations (InstructionDurations): An InstructionDurations object
                from a Qiskit backend.
        """
        
        duration_map = {}
        
        # Write instruction durations to buffer for parsing
        buffer = StringIO(newline="\n")
        buffer.write(str(instruction_durations))
        
        # Convert each line to dictionary entry
        buffer.seek(0)
        for line in buffer.readlines():
            
            # Import data
            line = line.strip()
            gate_name, end = line.split("(")
            end = "(" + end
            location, end = end.split(":")
            _, time, _ = end.split(" ")
            location = eval(location)
            time = float(time)
            if gate_name in ['id','rx','rz', 'sx', 'x',]:
                num_qudits = 1
            elif gate_name in ['cz', 'ecr',]:
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
            elif metric.startswith("gate_aware_depth_w="):
                weight = eval(metric.split("=")[1])
                value = round(self._get_gate_aware_depth(circ, weight), 4)
            elif metric == "estimated_runtime": 
                value = f"{self._estimate_runtime(circ, self.duration_map):e}"
            elif metric == "qiskit_scheduler_runtime":
                num_qiskit_scheduler_pulses = schedule_circuit(bqskit_to_qiskit(circ), self.schedule_config).duration
                value = num_qiskit_scheduler_pulses * self.schedule_config.dt
            else:
                value = None

            results_dict[circ_name][compiler][metric] = value


    def _get_gate_aware_depth(self, circuit, weight):
        """
        Calculate gate-aware depth for a single circuit.

        Uses weights 0, w (variable), and 1 for single-qubit Rz gates, other 
        single-qubit gates, and 1 for multi-qubit gates, respectively.

        Args:
            circuit (Circuit): The circuit to get the gate-aware depth of.

            weight (float): The weight for non-Rz single-qubit gates.
        """

        qudit_depths = np.zeros(circuit.num_qudits, dtype=float)
        for op in circuit:
            if op.num_qudits == 1 and op.gate == RZGate():
                increment = 0
            elif op.num_qudits == 1 and op.gate != RZGate():
                increment = weight
            else:
                increment = 1
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
            if file_name.endswith('.qasm'):
                path = os.path.join(qasm_dir, file_name)
                with open(path) as f:
                    now = datetime.now()
                    print(f"Started gathering data for {file_name} at {now.strftime('%H:%M:%S')}")
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
            file = open(f"depth_v_runtime_{now.date()}_{now.strftime('%H:%M:%S')}.csv", "w")
        else:
            file = open(file_path, "w")

        # Record test setup
        file.write('Test Configuration\n')
        file.write(f'compilers: {self.compilers}\n')
        file.write(f'backend: {self.backend.backend_name}\n')    

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