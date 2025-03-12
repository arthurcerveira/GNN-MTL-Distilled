import json

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

from chemprop_callback import (
    chemprop_single_target_callback, chemprop_multi_target_callback
)
import torch
import torch.nn.functional as F
from torch import Tensor

from chemprop import data, featurizers, models, nn


DATASETS = {
    "TVT-Split": "pivot_pXC50_over_1000_split.csv",
    "Lo-Split": "lo_pivot_pXC50_over_1000.csv",
    "Hi-Split": "hi_pivot_pXC50_over_1000.csv"
}
DATASET = "TVT-Split"


@nn.metrics.LossFunctionRegistry.register("masked_mse")
@nn.metrics.MetricRegistry.register("masked_mse")
class MaskedMSE(nn.metrics.ChempropMetric):
    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        lt_mask: Tensor | None = None,
        gt_mask: Tensor | None = None,
    ) -> None:
        """Update total loss by considering only valid targets per task."""
        # Auto-set mask where targets are NaN
        mask = ~torch.isnan(targets) if mask is None else mask  
        targets = torch.where(mask, targets, torch.zeros_like(targets))  # Replace NaN with 0 (ignored due to mask)

        weights = torch.ones_like(targets, dtype=torch.float) if weights is None else weights
        lt_mask = torch.zeros_like(targets, dtype=torch.bool) if lt_mask is None else lt_mask
        gt_mask = torch.zeros_like(targets, dtype=torch.bool) if gt_mask is None else gt_mask

        # Compute loss, apply weights and mask
        L = self._calc_unreduced_loss(preds, targets, mask)  
        L = L * weights * self.task_weights * mask  

        # Aggregate loss per task
        valid_counts = mask.sum(dim=0)  # Count valid values per task
        per_task_loss = L.sum(dim=0) / valid_counts.clamp(min=1)  # Avoid division by zero

        # Sum over tasks
        self.total_loss += per_task_loss.sum()
        self.num_samples += valid_counts.sum()

    def compute(self):
        return self.total_loss / self.num_samples if self.num_samples > 0 else torch.tensor(0.0)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Compute the element-wise loss while ignoring missing values based on mask."""
        loss = F.mse_loss(preds, targets, reduction="none")  # Compute MSE loss
        return loss * mask  # Zero out loss for missing targets



# Callbacks: Callable[[pd.DataFrame, List[str], bool], Dict[str, np.ndarray]]
callbacks = {
    f"{DATASET}-ST-Chemprop": chemprop_single_target_callback,
    f"{DATASET}-MT-Chemprop": chemprop_multi_target_callback
}


if __name__ == "__main__":
    # Load the test dataset
    dataset = pd.read_csv(f"../data/{DATASETS[DATASET]}")
    test_dataset = dataset[dataset["split"] == "test"]

    # Define the targets
    targets = test_dataset.drop(columns=["SMILES", "split"]).columns
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
    with open("target_baseline_assessment.json", "w") as f:
        json.dump(target_results, f, indent=4, ensure_ascii=False)
    print("\nResults saved to target_baseline_assessment.json")

    with open("aggregated_baseline_assessment.json", "w") as f:
        json.dump(aggregated_results, f, indent=4, ensure_ascii=False)
    print("Results saved to aggregated_baseline_assessment.json")
