import os
from pathlib import Path
import json
import sys

current_file_dir = Path(__file__).resolve().parent
chemprop_path = current_file_dir / "chemprop"
sys.path.append(str(chemprop_path))

import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from chemprop import data, featurizers, models, nn, conf


DATASETS = {
    "TVT": "pivoted_pXC50_over_1000_split.csv",
    "TVT-KD": "pivoted_pXC50_over_1000_split_distilled.csv",
    "Lo": "pivoted_pXC50_over_1000_split_lo.csv",
    "Lo-KD": "pivoted_pXC50_over_1000_split_lo_distilled.csv",
    "Hi": "pivoted_pXC50_over_1000_split_hi.csv",
    "Hi-KD": "pivoted_pXC50_over_1000_split_hi_distilled.csv",
}
DATASET = "Hi"

TRAIN_TARGET_SPECIFIC = False
TRAIN_ALL_MULTI_TARGET = False
TRAIN_CLUSTERED_MULTI_TARGET = True
RETRAIN = True

checkpoints_dir = current_file_dir / ".." / "checkpoints"
results_dir = current_file_dir / ".." / "results"
data_dir = current_file_dir / ".." / "data"

target_clusters = None
if TRAIN_CLUSTERED_MULTI_TARGET:
    label = DATASET.split('-')[0]
    with open(data_dir / f"target_clusters_correlation_{label}.json", "r") as f:
        target_clusters = json.load(f)

# Reference: https://chemprop.readthedocs.io/en/latest/training.html
input_path = data_dir / DATASETS[DATASET] # path to your data .csv file
activities_df = pd.read_csv(input_path)
num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'SMILES' # name of the column containing SMILES strings
targets = activities_df.columns
# Remove columns SMILES and split from the list of targets
targets = [
    target for target in targets if target not in [smiles_column, 'split', 'cluster']
]
total_targets = len(targets)


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


def data_pre_processing(activity_df, targets_columns, smiles_column, num_workers):
    df_input = activities_df[[smiles_column] + targets_columns + ["split"]].copy()
    
    # Drop rows with missing values in all targets columns
    df_input = df_input.dropna(subset=targets_columns, how="all")
    
    smis = df_input.loc[:, smiles_column].values
    ys = df_input.loc[:, targets_columns].values
    splits = df_input.loc[:, "split"].values
    print(f"Number of data points: {len(smis):,}")
    all_data = [
        data.MoleculeDatapoint.from_smi(smi, y) for smi, y in 
        tqdm(zip(smis, ys), total=len(smis), desc="Processing molecules")
    ]

    # Get indices for train, val, and test from splits column
    train_indices, val_indices, test_indices = (
        np.array(np.where(splits == "train")), 
        np.array(np.where(splits == "val")),
        np.array(np.where(splits == "test"))
    )

    # If there is no validation set, set the last 10% of the training set as the validation set
    if len(val_indices[0]) == 0:
        val_size = int(len(train_indices[0]) * 0.1)
        val_indices = np.array([train_indices[0][-val_size:]])
        train_indices = np.array([train_indices[0][:-val_size]])

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    try:
        train_dset = data.MoleculeDataset(train_data[0], featurizer)
        scaler = train_dset.normalize_targets()

        val_dset = data.MoleculeDataset(val_data[0], featurizer)
        val_dset.normalize_targets(scaler)

        test_dset = data.MoleculeDataset(test_data[0], featurizer)
    except Exception as e:
        print(f"Error normalizing targets: {e}")
        breakpoint()

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, batch_size=1024)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False, batch_size=1024)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False, batch_size=1024)

    return train_loader, val_loader, test_loader, scaler


def train_single_target(target, t_idx):
    print(f"({t_idx}/{total_targets}) Training model for {target}")
    
    checkpoint_path = checkpoints_dir / "target-specific" / DATASET / target
    if not RETRAIN and os.path.exists(checkpoint_path / "last.ckpt"):
        print(f"Model for {target} already trained. Skipping...")
        return

    train_loader, val_loader, test_loader, scaler = data_pre_processing(
        activities_df, [target], smiles_column, num_workers
    )

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)
    batch_norm = True

    metric_list = [nn.metrics.MSE(), nn.metrics.MAE(), nn.metrics.R2Score()] # Only the first metric is used for training and early stopping
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        checkpoint_path,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        enable_version_counter=False
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
    )

    print(f"Saving checkpoints to {checkpoint_path.resolve()}")
    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)
    return results


def train_multi_target(targets):
    checkpoint_path = checkpoints_dir / "multi-target" / DATASET / "MT-ALL"
    if not RETRAIN and os.path.exists(checkpoint_path / "last.ckpt"):
        print(f"Multi-target model already trained. Skipping...")
        return

    train_loader, val_loader, test_loader, scaler = data_pre_processing(
        activities_df, targets, smiles_column, num_workers
    )

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    # Adjusted FFN to handle multiple targets
    ffn = nn.RegressionFFN(
        output_transform=output_transform, n_tasks=len(targets), criterion=MaskedMSE()
    )

    batch_norm = True
    metric_list = [MaskedMSE()]  # Only the first metric is used for training and early stopping
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        checkpoint_path,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        enable_version_counter=False
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
    )

    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)

    # Save index-to-target mapping to checkpoint directory for later use
    index_to_target = {i: target for i, target in enumerate(targets)}
    with open(checkpoint_path / "index_to_target.json", "w") as f:
        json.dump(index_to_target, f, indent=4, ensure_ascii=False)

    return results


def train_clustered_multi_target(clustered_targets, cluster_idx):
    print(f"Training clustered multi-target model for {len(clustered_targets)} targets (cluster {cluster_idx})")

    checkpoint_path = checkpoints_dir / "clustered-multi-target" / DATASET / f"cluster-{cluster_idx}"
    if not RETRAIN and os.path.exists(checkpoint_path / "last.ckpt"):
        print(f"Clustered multi-target model for cluster {cluster_idx} already trained. Skipping...")
        return

    train_loader, val_loader, test_loader, scaler = data_pre_processing(
        activities_df, clustered_targets, smiles_column, num_workers
    )
    
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    ffn = nn.RegressionFFN(
        output_transform=output_transform, n_tasks=len(clustered_targets), criterion=MaskedMSE()
    )
    
    batch_norm = True
    metric_list = [MaskedMSE()]  # Only the first metric is used for training and early stopping
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        checkpoint_path,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        enable_version_counter=False
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
    )

    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)

    return results


if __name__ == "__main__":
    if TRAIN_TARGET_SPECIFIC:
        checkpoint_path = checkpoints_dir / "target-specific" / DATASET
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        print("Training target-specific models.")
        all_results = dict()

        for t_idx, target in enumerate(targets):
            results = train_single_target(target, t_idx)
            all_results[target] = results

        print("Training complete.")
        results_path = results_dir / "target-specific" / "chemprop-train.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print("Results saved.")

    if TRAIN_ALL_MULTI_TARGET:
        checkpoint_path = checkpoints_dir / "multi-target" / DATASET / "MT-ALL"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        print("Training model for all targets simultaneously")
        results = train_multi_target(targets)

        print("Training complete.")
        results_path = results_dir / "multi-target" / "chemprop-train.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print("Results saved.")

    if TRAIN_CLUSTERED_MULTI_TARGET:
        print("Training model for clustered multi-targets")
        checkpoint_path = checkpoints_dir / "clustered-multi-target" / DATASET
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        all_results = dict()
        cluster_to_targets = target_clusters["cluster_to_targets"]

        def train_cluster(cluster_idx, cluster_to_targets):
            clustered_targets = cluster_to_targets[cluster_idx]
            results = train_clustered_multi_target(clustered_targets, cluster_idx)
            return cluster_idx, results

        with Pool(processes=3) as pool:
            train_fn = partial(train_cluster, cluster_to_targets=cluster_to_targets)
            for cluster_idx, results in pool.imap_unordered(train_fn, cluster_to_targets.keys()):
                all_results[cluster_idx] = results

        # for cluster_idx in cluster_to_targets.keys():
        #     clustered_targets = cluster_to_targets[cluster_idx]
        #     results = train_clustered_multi_target(clustered_targets, cluster_idx)
        #     all_results[cluster_idx] = results

        print("Training complete.")

        results_path = results_dir / "clustered-multi-target" / "chemprop-train.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print("Results saved.")
