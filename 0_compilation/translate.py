"""
Translates the compiled/optimized circuits to circuits which
are compatible with IBM devices using Qiskit.

To preserve differences in algorithm optimizations, Qiskit's
optimization level is set to 0 (no optimization).
"""

import os

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

from bqskit.ir import Circuit
from bqskit.ext import bqskit_to_qiskit
from bqskit.ext import qiskit_to_bqskit


device_name = ""
root_qasm_dir = r"./qasm/compiled/"

# Iterate over all .qasm files in the root directory
for root, subdirs, files in os.walk(root_qasm_dir):

    if len(files) > 0:
        for file in files:

            if not file.endswith(".qasm"):
                continue

            comp_file_path = os.path.join(root, file)
            root_path_list = root.split(os.sep)

            # Skip untranslatable circuits
            if root_path_list[4] == "eagle" and root_path_list[3] in [
                "tket",
            ]:
                print(f"{comp_file_path} will cause translation error, skipping.")
                continue

            # Obtain backend name from directory
            new_device_name = root_path_list[5]

            # Connect to new backend if necessary
            if new_device_name != device_name:
                device_name = new_device_name
                backend = QiskitRuntimeService().backend(device_name)

            print(f"Translating {comp_file_path}")

            # Load circuit via bqskit
            circ = Circuit.from_file(f"{comp_file_path}")
            q_circ = bqskit_to_qiskit(circ)

            # Translate circuit with qiskit
            pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
            trans_circ = pm.run(q_circ)

            # Build target path by replacing 'compiled'
            # with 'translated'
            root_path_list[2] = "translated"
            target_file_path = os.path.join(*root_path_list, file)

            # Save to qasm
            bqskit_out_circuit = qiskit_to_bqskit(trans_circ)
            bqskit_out_circuit.save(target_file_path)
