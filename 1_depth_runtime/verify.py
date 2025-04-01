import os

from drtester import DRTester


"""
Obtain the estimated runtime and Qiskit pulse schedule duration for
circuits compiled for the IBM Sherbrooke.
"""


# Configuration
compilers = ["sabre0330", "sqgm", "qiskit141",]
hardware = {
    "eagle": ["ibm_sherbrooke",],
}
metrics = ["estimated_runtime", "qiskit_scheduler_runtime",]
parent_qasm_dir = r"../0_compilation/qasm/translated/"

inst_dur_out_path = r"./data/instruction_durations/"
csv_out_path = rf"./data/csv/verify.csv"


# Experiment Script
for architecture, device_names in hardware.items():
    print(f"+++{architecture}+++")

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

        tester.record(csv_out_path)
