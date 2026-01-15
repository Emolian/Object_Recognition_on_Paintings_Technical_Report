import matplotlib.pyplot as plt
from src import config


def plot_results(results_dict):
    """Generates the comparison chart for the report."""
    labels = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure(figsize=(10, 6))

    # Color coding: Gray for Baseline, Red for Zero-Shot, Green for Recovery
    colors = ['gray'] + ['red', 'green'] + ['blue'] * (len(values) - 3)

    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('mAP@50')
    plt.title('The Cross-Depiction Problem: Drop & Recovery')
    plt.ylim(0, 1.0)

    # Add Westlake Baseline line
    plt.axhline(y=config.WESTLAKE_2016_BASELINE, color='r', linestyle='--', label='Westlake (2016) Baseline')
    plt.legend()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom')

    output_path = 'replication_result_chart.png'
    plt.savefig(output_path)
    print(f"[*] Result chart saved to '{output_path}'")