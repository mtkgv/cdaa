"""
Obtains the depth(s) and estimated runtime for circuits compiled
for all available IBM devices.
"""

import os

from qiskit_ibm_runtime import QiskitRuntimeService

from utils import DRTester
from utils import translate_ibm_instruction_durations


def construct_weight_map(duration_maps):
    """
    Construct the average weight map for gate-aware depth from a list of
    device duration maps.

    Returns (dict[str: float]): Mapping from native gate set names to gate weights.

    Args:
        duration_maps (list[dict]): The list of device duration maps to take
            the average over.
    """

    if len(duration_maps) < 1:
        raise RuntimeError(
            f"At least 1 duration map required, got {len(duration_maps)}"
        )

    weight_map = {}

    # Sum total gate execution time and number of gates for every gate in the devices' native
    # gate set, over all devices and all locations
    for duration_map in duration_maps:
        for num_qudits in duration_map.keys():
            for location in duration_map[num_qudits].keys():
                for gate_name in duration_map[num_qudits][location].keys():

                    execution_time = duration_map[num_qudits][location][gate_name]

                    if gate_name not in weight_map.keys() and gate_name not in [
                        "measure",
                        "reset",
                    ]:
                        weight_map[gate_name] = [execution_time, 1]
                    elif gate_name not in ["measure", "reset"]:
                        weight_map[gate_name][0] += execution_time
                        weight_map[gate_name][1] += 1

    # Use total gate execution time and number of gates to calculate average gate time
    for gate_name in weight_map.keys():
        total_execution_time = weight_map[gate_name][0]
        total_gate_count = weight_map[gate_name][1]
        weight_map[gate_name] = total_execution_time / total_gate_count

    # Normalize average gate time by max gate time
    max_avg_gate_time = max(weight_map.values())

    for gate_name in weight_map.keys():
        weight_map[gate_name] /= max_avg_gate_time

    return weight_map


def construct_weight_map_local(inst_dur_dir, device_names):
    """
    Construct a weight map for gate-aware depth using the average gate times of
    the specified devices using locally saved InstructionDurations text files.

    Returns (dict[str: float]): Mapping from native gate set names to gate weights.

    Args:
        inst_dur_dir (str): The path to InstructionDuration text files.

        device_names (list[str]): A list of device names to filter the
            InstructionDuration text files with. Includes files containing device
            names on the list.
    """

    duration_maps = []

    # Get one copy of the duration map for each device
    for device_name in device_names:
        for file_name in os.listdir(inst_dur_dir):
            if device_name in file_name:
                full_file_path = os.path.join(inst_dur_dir, file_name)
                duration_map = translate_ibm_instruction_durations(full_file_path)
                duration_maps.append(duration_map)
                break

    weight_map = construct_weight_map(duration_maps)

    return weight_map


def construct_weight_map_API(device_names):
    """
    Construct a weight map for gate-aware depth using the average gate times of
    the specified devices using InstructionDurations obtained through IBM API
    access.

    Returns (dict[str: float]): Mapping from native gate set names to gate weights.

    Args:
        device_names (list[str]): The list of device names to obtain
            InstructionDurations from.
    """

    duration_maps = []

    for device_name in device_names:
        backend = QiskitRuntimeService().backend(device_name)
        inst_dur = backend.instruction_durations
        duration_map = translate_ibm_instruction_durations(inst_dur)
        duration_maps.append(duration_map)

    weight_map = construct_weight_map(duration_maps)

    return weight_map


# Configuration
compilers = [
    "sabre0330",
    "sqgm",
    "tket",
    "qiskit141",
]
hardware = {
    "eagle": [
        "ibm_sherbrooke",
        "ibm_kyiv",
        "ibm_brisbane",
    ],
    "heron": [
        "ibm_marrakesh",
        "ibm_kingston",
        "ibm_aachen",
    ],
}

weights = [x / 100.0 for x in range(1, 101)]
metrics = [
    "multi_qubit_depth",
]
for weight in weights:
    metrics.append(f"gate_aware_depth_w={weight}")
metrics += [
    "gate_aware_depth",
    "trad_depth",
    "estimated_runtime",
]

run_local = True
save_inst_durs = False
inst_dur_dir = r"./data/instruction_durations/"
parent_qasm_dir = r"../0_compilation/qasm/translated/"


# Experiment Script
for architecture, device_names in hardware.items():
    print(f"+++{architecture}+++")

    # TKET-compiled circuits are unavailable for Eagle devices
    if architecture == "eagle":
        compilers.remove("tket")
    else:
        compilers.insert(2, "tket")

    # Construct weight map using gate times from all devices of given architecture
    if run_local:
        weight_map = construct_weight_map_local(inst_dur_dir, device_names)
    else:
        weight_map = construct_weight_map_API(device_names)

    print(f"{architecture} weight map:\n{weight_map}")

    for device_name in device_names:
        print(f"---{device_name}---")

        # Path to device-specific InstructionDurations text file, if needed
        inst_dur_path = os.path.join(inst_dur_dir, device_name + "_inst_dur.txt")

        tester = DRTester(
            device_name=device_name,
            weight_map=weight_map,
            compilers=compilers,
            metrics=metrics,
            run_local=run_local,
            inst_dur_path=inst_dur_path,
        )

        if save_inst_durs:
            tester.save_ibm_instruction_durations(
                instruction_durations=tester.backend.instruction_durations,
                save_dir=inst_dur_dir,
            )

        for compiler in compilers:
            print(f"==={compiler}===")

            qasm_dir = os.path.join(
                parent_qasm_dir, rf"{compiler}/{architecture}/{device_name}/"
            )
            tester.get_data(qasm_dir)

        csv_out_path = rf"./data/csv/{device_name}.csv"
        tester.record(csv_out_path)
