import subprocess
import json

def run_script(script_name, args):
    cmd = ["python", script_name] + args
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    with open('conf.py', 'r') as f:
        config_str = f.read().replace('config = ', '')
        config = eval(config_str)

    print("--- Starting Experiment ---")

    print("\n--- Step 1: Collecting Trajectories ---")
    run_script("datacollect.py", [])

    print("\n--- Step 2: Training Model ---")
    run_script("my_train.py", [])

    print("\n--- Step 3: Validating Model ---")
    print("\n--- Validating on training distribution ---")
    run_script("my_val.py", ["--corridor_length", str(config['env']['train_corridor_length'])])

    print("\n--- Validating on generalization task ---")
    run_script("my_val.py", ["--corridor_length", str(config['env']['test_corridor_length'])])

    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    main() 