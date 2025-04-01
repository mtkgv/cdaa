import os

from drtester import DRTester


"""
Obtain the depth(s) and estimated runtime for circuits compiled
for all available IBM devices.
"""


# Configuration
compilers = ["sabre0330", "sqgm", "tket", "qiskit141",]
hardware = {
    "eagle": ["ibm_sherbrooke", "ibm_kyiv", "ibm_brisbane",],
    "heron": ["ibm_marrakesh", "ibm_kingston", "ibm_aachen",],
}

weights = [x/100.0 for x in range(1,101)]
metrics = ["multi_qubit_depth",] 
for weight in weights:
    metrics.append(f"gate_aware_depth_w={weight}")
metrics += ["trad_depth", "estimated_runtime",]

parent_qasm_dir = r"../0_compilation/qasm/translated/"

inst_dur_out_path = r"./data/instruction_durations/"


# Experiment Script
for architecture, device_names in hardware.items():
    print(f"+++{architecture}+++")

    # TKET-compiled circuits are unavailable for Eagle devices
    if architecture == "eagle":
        compilers.remove("tket")
    else:
        compilers.insert(2, "tket")

    for device_name in device_names:
        print(f"---{device_name}---")
        
        tester = DRTester(
            device_name=device_name,
            compilers=compilers,
            metrics=metrics,
        )

        tester.save_ibm_instruction_durations(
            instruction_durations=tester.backend.instruction_durations,
            save_dir=inst_dur_out_path,
        )

        for compiler in compilers:
            print(f"==={compiler}===")

            qasm_dir = os.path.join(parent_qasm_dir, rf"{compiler}/{architecture}/{device_name}/")
            tester.get_data(qasm_dir)

        csv_out_path = rf"./data/csv/{device_name}.csv"
        tester.record(csv_out_path)