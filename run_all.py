import os
import subprocess
import time  # Add a small delay for better readability in the console output

# -----------------------------
# Helper function
# -----------------------------
def run_experiments(dataset_name: str, original_config_path: str, temp_config_path: str, predict_values: list[int]) -> None:
    """
    Run a set of experiments by updating `num_for_predict` in a temporary config file
    and executing the training/inference script for each requested horizon.

    Args:
        dataset_name: Name of the dataset (used only for console messages).
        original_config_path: Path to the base config file.
        temp_config_path: Path to the temporary config file (will be overwritten).
        predict_values: List of `num_for_predict` values to run.
    """
    script_to_run = "run.py"

    print(f"Starting experiments for {dataset_name}...", flush=True)
    print("-" * 40, flush=True)
    time.sleep(1)

    for value in predict_values:
        print(f"Running experiment with num_for_predict = {value}", flush=True)

        # Read the original config and update only the `num_for_predict` line
        with open(original_config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.strip().startswith("num_for_predict"):
                new_lines.append(f"num_for_predict = {value}\n")
            else:
                new_lines.append(line)

        with open(temp_config_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        command = ["python", script_to_run, "--config", temp_config_path]

        try:
            # Run the script and stream output live to the terminal (no capture_output)
            subprocess.run(command, check=True)
            print(f"\nExperiment for num_for_predict={value} finished successfully.", flush=True)

        except subprocess.CalledProcessError as e:
            print(f"!!! Experiment failed for num_for_predict={value} !!!", flush=True)
            # Since output is not captured, check the terminal output for details
            print(f"Returned exit code: {e.returncode}", flush=True)
            break

        finally:
            print("-" * 40, flush=True)
            time.sleep(1)

    # Clean up temp config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        print(f"Temporary file ({temp_config_path}) deleted.", flush=True)


# -----------------------------
# Experiments
# -----------------------------
run_experiments(
    dataset_name="PEMS03",
    original_config_path="./configurations/PEMS03.conf",
    temp_config_path="./configurations/PEMS03_temp.conf",
    predict_values=[12, 24, 48, 96],
)

run_experiments(
    dataset_name="PEMS04",
    original_config_path="./configurations/PEMS04.conf",
    temp_config_path="./configurations/PEMS04_temp.conf",
    predict_values=[12, 24, 48, 96],
)

run_experiments(
    dataset_name="PEMS07",
    original_config_path="./configurations/PEMS07.conf",
    temp_config_path="./configurations/PEMS07_temp.conf",
    predict_values=[48, 96],
)

run_experiments(
    dataset_name="PEMS08",
    original_config_path="./configurations/PEMS08.conf",
    temp_config_path="./configurations/PEMS08_temp.conf",
    predict_values=[12, 24, 48, 96],
)

print("All experiments completed.", flush=True)
