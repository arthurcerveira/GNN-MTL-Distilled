from pathlib import Path
import json
import sys

chemprop_path = Path(__file__).parent / '..' / "baselines" / "chemprop"
sys.path.append(str(chemprop_path))

import numpy as np
from lightning import pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from chemprop import data, featurizers, models, nn


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


def run_mpnn_on_smiles(smiles_input, mpnn):
    test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles_input]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
    test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=4)

    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=False,
            # enable_progress_bar=True,
            accelerator="cuda",
            devices=1,
        )
        test_preds = trainer.predict(mpnn, test_loader)
        test_preds = np.concatenate(test_preds, axis=0)  # Concatenate batches

    # Check if the model is single or multi-target
    tasks = test_preds.shape[1]
    if tasks == 1:
        # Single target model: return a 1D array (# of samples,)
        return test_preds[:, 0]
    
    # Multi-target model: return a 2D array (# of targets, # of samples)
    # Can be decomposed into multiple 1D arrays
    preds_tasks = np.array([test_preds[:, i] for i in range(tasks)])
    return preds_tasks


def chemprop_single_target_callback(test_dataset, targets, verbose=False):
    target_predictions = dict()

    for target in targets:
        smiles_input = test_dataset.dropna(subset=[target])["SMILES"].tolist()

        if verbose:
            print(f"Running ST-Chemprop on {len(smiles_input)} SMILES for target {target}...")
        
        mpnn = models.MPNN.load_from_checkpoint(f'../checkpoints/target-specific/{target}/last.ckpt')
        predictions = run_mpnn_on_smiles(smiles_input, mpnn)
        target_predictions[target] = predictions

    return target_predictions


def chemprop_multi_target_callback(test_dataset, targets, verbose=False):
    if verbose:
        print(f"Running MT-Chemprop on {len(test_dataset)} SMILES for {len(targets)} targets...")
    
    target_predictions = dict()
    with open("../checkpoints/multi-target/MT-ALL/index_to_target.json", "r") as f:
        index_to_target = json.load(f)
    
    smiles_input = test_dataset["SMILES"].tolist()
    mpnn = models.MPNN.load_from_checkpoint(f'../checkpoints/multi-target/MT-ALL/last.ckpt')
    predictions = run_mpnn_on_smiles(smiles_input, mpnn)

    target_to_index = {v: k for k, v in index_to_target.items()}
    for target in targets:
        target_mask = test_dataset[target].notna()
        target_idx = int(target_to_index[target])
        target_predictions[target] = predictions[target_idx][target_mask]

    return target_predictions
