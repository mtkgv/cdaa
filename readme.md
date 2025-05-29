# Circuit Depth Accuracy (Artifact)
This repository hosts data and source code for experiments in the paper *Is Circuit Depth Accurate for Comparing Quantum Circuit Runtimes?.*[^1]

The experimental code for the single-qubit gates matter (SQGM) algorithm[^2] was adapted from the [original SQGM repository](https://github.com/ebony72/sqgm) to work with a newer version of NumPy.


## Installation

The experiments require an "old" and "new" virtual environment to run, which can be installed using

```sh
python -m venv /path/to/oldvenv
activate .oldvenv/bin/activate
pip install -r old_requirements.txt
deactivate

python -m venv /path/to/newvenv
activate .newvenv/bin/activate
pip install -r new_requirements.txt
deactivate
```

The "old" environment is only used to compile circuits with SABRE and SQGM, since they are implemented using an older version of Qiskit. For all other steps, only the "new" environment is needed.


## Organization

The repository organizes files into three subdirectories for experimental reproduction and one for raw data archival.

```sh
(root)
├── 0_compilation
│   └── qasm
│       ├── compiled
│       ├── original
│       └── translated
├── 1_depth_runtime
│   └── data
│       ├── csv
│       └── instruction_durations
├── 2_analysis
│   └── out
└── data
```

The first three, `0_compilation`, `1_depth_runtime`, and `2_analysis`, act as workspaces in which to run one of the experiment's three phases. Each contains the Python code needed to run that phase and an output directory to store output data in. 

More specifically, the output directories' contents consist of:

- `0_compilation/qasm/`
    - `original`: Original QASM files for the 15 benchmark circuits
    - `compiled`: New QASM files after compiling them for target devices. Within this directory, they are stored according to the compiler used, architecture, and device using the path `/{compiler}/{architecture}/{device}/`
    - `translated`: The new QASM files after translating them to their target device's native gate set. Internally, it organizes files the same way as the `compiled` directory.

- `1_depth_runtime/data/`
    - `csv`: CSV files containing depths and runtimes measured for the translated circuits on their respective devices
    - `instruction_durations`: Text files recording the device gate times provided by IBM

- `2_analysis/imgs/`
    - `out`: Figure image files created from the CSV data

- `data`: A copy of the preceding directory structure, populated with data and results obtained during the experiment. This data may be may be copied to the experimental workspace to reproduce the paper's results.
    - *Example*: To reproduce the figures, copy the CSV files in `data/1_depth_runtime/data/csv/` to `1_depth_runtime/data/csv/` and then run the final experimental phase in `2_analysis`.


## Running the Experiment

Experimental phases may be run in sequence or individually, as long as the expected data files from the previous phase are present. Configuration settings for each phase can be changed within the Python scripts to modify the devices, depth metrics tested, etc.

### A. Compilation

1. Navigate to `0_compilation`

2. Compile circuits with SABRE and SQGM by activating the "old"  virtual environment and running the `compile` script.

    ```sh
    activate /path/to/oldvenv/bin/activate
    python compile.py
    ```

3. Compile circuits with TKET and Qiskit by switching to the "new" virtual environment and running the same `compile` script. **Hereafter, all scripts may be executed in the "new" virtual environment.**

    ```sh
    activate /path/to/newvenv/bin/activate
    python compile.py
    ```

4. Translate all previously compiled circuits to the target devices' native gate sets by running `python translate.py`. This step is required for circuits compiled with SABRE and SQGM, since they are routing algorithms and therefore do not rebase circuits.

### B. Depth & Runtime

1. Next, navigate to `1_depth_runtime`

2. Obtain the depths and runtimes of the compiled circuits on their target devices by running `python data.py`. Note that the script is set up to run in local mode, which averts the need for Qiskit API access but requires copying the saved InstructionDurations text files in `1_depth_runtime/data/instruction_durations`.

3. Obtain the estimated runtime and Qiskit pulse schedule duration for circuits compiled for the IBM Sherbrooke by running `python verify.py`

### C. Analysis

1. Finally, navigate to `2_analysis`

2. Plot the figures from the results section by running `python runfigs.py`

3. Create the (textual) numerical analysis of depth metrics by running `python runstats.py`


## License

Apache License 2.0


## References

[^1]: M. Tremba, J. Liu, and P. Hovland, "Is circuit depth accurate for comparing quantum circuit runtimes?," *arXiv preprint: [arXiv:2505.16908](https://doi.org/10.48550/arXiv.2505.16908)*, 2025.

[^2]: S. Li, K. D. Nguyen, Z. Clare and Y. Feng, "Single-qubit gates matter for optimising quantum circuit depth in qubit mapping," *2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD), San Francisco, CA, USA*, 2023, pp. 1-9, doi: [10.1109/ICCAD57390.2023.10323863](https://doi.org/10.1109/ICCAD57390.2023.10323863).