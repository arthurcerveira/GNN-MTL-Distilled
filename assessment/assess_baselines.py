import json
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

from chemprop_callback import (
    chemprop_single_target_callback, chemprop_multi_target_callback, MaskedMSE
)
from bambu_callback import bambu_callback

DATASETS = {
    "TVT-Split": "pivoted_pXC50_over_1000_split.csv",
    "Lo-Split": "lo_pivoted_pXC50_over_1000_split_5000_clusters.csv",
    "Hi-Split": "hi_pivot_pXC50_over_1000.csv"
}
DATASET = "Lo-Split"
DATA_PATH = Path(__file__).parent / ".." / "data"


# Callbacks: Callable[[pd.DataFrame, List[str], bool], Dict[str, np.ndarray]]
callbacks = {
    f"{DATASET}-ST-Chemprop": chemprop_single_target_callback,
    f"{DATASET}-MT-Chemprop": chemprop_multi_target_callback,
    f"{DATASET}-Bambu": bambu_callback
}


if __name__ == "__main__":
    # Load the test dataset
    dataset = pd.read_csv(DATA_PATH / DATASETS[DATASET])
    test_dataset = dataset[dataset["split"] == "test"]

    # Define the targets
    targets = test_dataset.drop(columns=["SMILES", "split"]).columns
    if "cluster" in targets:
        targets = targets.drop("cluster")
    target_results = {
        label: dict() for label in callbacks.keys()
    }
    # Aggregations: mean and std for each metric across all targets
    aggregated_results = {
        label: dict() for label in callbacks.keys()
    }

    # Run the callbacks
    for label, callback in callbacks.items():
        predictions = callback(test_dataset, targets, verbose=True)

        # Compute metrics
        for target in targets:
            y_true = test_dataset[target].dropna().values
            y_pred = predictions[target]

            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            target_results[label][target] = {
                "MSE": mse,
                "MAE": mae,
                "R2": r2
            }

    # Aggregate the results
    for label in callbacks.keys():
        for metric in ["MSE", "MAE", "R2"]:
            aggregated_results[label][metric] = {
                "mean": np.mean([target_results[label][target][metric] for target in targets]),
                "std": np.std([target_results[label][target][metric] for target in targets])
            }
    
    # Print the aggregated results
    for label in callbacks.keys():
        print(f"\n{label}")
        for metric in ["MSE", "MAE", "R2"]:
            mean = aggregated_results[label][metric]["mean"]
            std = aggregated_results[label][metric]["std"]
            print(f"{metric}: {mean:.3f} ± {std:.3f}")

    # Save target and aggregated results
    with open("../results/target_baseline_assessment.json", "w") as f:
        json.dump(target_results, f, indent=4, ensure_ascii=False)
    print("\nResults saved to target_baseline_assessment.json")

    with open("../results/aggregated_baseline_assessment.json", "w") as f:
        json.dump(aggregated_results, f, indent=4, ensure_ascii=False)
    print("Results saved to aggregated_baseline_assessment.json")
