import os
from src.data_loader import ScientificDataProcessor
from src.experiment import ExperimentRunner
from src.visualizer import plot_results
from src import config

def main():
    # 0. Check Requirements
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"[ERROR] '{config.RAW_DATA_PATH}' not found.")
        print(f"Please download the PeopleArt dataset and extract it into '{config.RAW_DATA_PATH}'")
        return

    # 1. Data Processing
    print("[*] Initializing Data Processor...")
    processor = ScientificDataProcessor()
    processor.prepare_data_split()

    # 2. Initialize Experiment
    runner = ExperimentRunner(processor)

    # 3. Execute Phases
    runner.run_phase_1_baseline()
    runner.run_phase_2_zero_shot()
    runner.run_phase_3_adaptation()
    runner.run_phase_4_style_analysis()
    runner.run_phase_5_divergence_hypothesis()

    # 4. Final Reporting
    final_results = runner.print_conclusion()
    plot_results(final_results)

if __name__ == "__main__":
    main()