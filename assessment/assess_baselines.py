import json
import sys
from pathlib import Path
from functools import partial
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

from chemprop_callback import (
    chemprop_single_target_callback, 
    chemprop_multi_target_callback, 
    chemprop_clustered_multi_target_callback,
    MaskedMSE,
)
from bambu_callback import bambu_callback
from dnn_callback import (
    dnn_single_target_callback,
    dnn_clustered_multi_target_callback,
    # dnn_multi_target_callback,
)

DATASETS = {
    "TVT": "pivoted_pXC50_over_1000_split.csv",
    "Lo": "pivoted_pXC50_over_1000_split_lo.csv",
    "Hi": "pivoted_pXC50_over_1000_split_hi.csv"
}
DATASET = "TVT" if len(sys.argv) == 1 else sys.argv[1]
DATA_PATH = Path(__file__).parent / ".." / "data"
RESULTS_PATH = Path(__file__).parent / ".." / "results"


# KD: Knowledge Distillation
def knowledge_distillation(callback):
    def wrapper(*args, **kwargs):
        # Find trained_on key in kwargs
        trained_on = kwargs.get("trained_on")
        if trained_on is None:
            raise ValueError("trained_on key not found in kwargs")

        # Replace trained_on with trained_on-KD
        kwargs["trained_on"] = kwargs["trained_on"] + "-KD"
        return callback(*args, **kwargs)

    return wrapper


chemprop_clustered_multi_target_callback_kd = knowledge_distillation(chemprop_clustered_multi_target_callback)
chemprop_multi_target_callback_kd = knowledge_distillation(chemprop_multi_target_callback)


# Callbacks: Callable[[pd.DataFrame, List[str], bool], Dict[str, np.ndarray]]
callbacks = {
    f"{DATASET}-Clustered-MT-Chemprop-KD": chemprop_clustered_multi_target_callback_kd,
    f"{DATASET}-Clustered-MT-Chemprop": chemprop_clustered_multi_target_callback,
    f"{DATASET}-MT-Chemprop-KD": chemprop_multi_target_callback_kd,
    f"{DATASET}-MT-Chemprop": chemprop_multi_target_callback,
    f"{DATASET}-ST-Chemprop": chemprop_single_target_callback,
    f"{DATASET}-Clustered-MT-DNN": dnn_clustered_multi_target_callback,
    # f"{DATASET}-MT-DNN": dnn_multi_target_callback,
    f"{DATASET}-DNN": dnn_single_target_callback,
    f"{DATASET}-Bambu": bambu_callback,
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
        predictions = callback(test_dataset, targets, trained_on=DATASET, verbose=True)

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
    with open(RESULTS_PATH / f"target_baseline_assessment_{DATASET}.json", "w") as f:
        json.dump(target_results, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to target_baseline_assessment_{DATASET}.json")

    with open(RESULTS_PATH / f"aggregated_baseline_assessment_{DATASET}.json", "w") as f:
        json.dump(aggregated_results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to aggregated_baseline_assessment_{DATASET}.json")
