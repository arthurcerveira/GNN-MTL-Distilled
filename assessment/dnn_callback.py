import sys
from pathlib import Path
import json

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

current_dir = Path(__file__).parent
DNN = current_dir / '..' / "baselines" / "dnn" / "src"
sys.path.append(str(DNN))

checkpoints_dir = current_dir / '..' / "checkpoints" / "dnn"
data_dir = current_dir / '..' / "data"

from ST import net
from MT import net_mt


def get_fingerprints(smiles, verbose=True):
    fp_generator = GetMorganGenerator(radius=2, fpSize=2048)

    # Convert SMILES to fingerprints
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    mols_iter = tqdm(mols, desc="Generating fingerprints") if verbose else mols

    all_fps = [fp_generator.GetFingerprint(mol) for mol in mols_iter]
    fps_array = np.array(all_fps)

    return fps_array


def dnn_single_target_callback(test_dataset, targets, trained_on="TVT", verbose=False):
    """
    Predicts values for multiple targets using trained DNN models.

    Args:
        test_dataset (pd.DataFrame): DataFrame with at least 'SMILES' and the target columns.
        targets (list of str): List of target column names.
        trained_on (str): Dataset/model name for checkpoint path.
        verbose (bool): Whether to show progress bars.

    Returns:
        dict: {target: np.ndarray of predictions (NaN for missing target values)}
    """
    # Model architecture parameters (should match training)
    kwargs = {
        "input_dim": 2048,
        "hidden1_dim": 1024,
        "hidden2_dim": 128,
        "output_dim": 1,
        "drop_rate": 0.5
    }

    smiles = test_dataset['SMILES'].values
    # Precompute fingerprints for all molecules (since SMILES is the same for all targets)
    fingerprints = get_fingerprints(smiles, verbose=verbose)
    fingerprints_tensor = torch.tensor(fingerprints, dtype=torch.float32)

    predictions_dict = dict()

    for target in tqdm(targets, desc="DNN targets") if verbose else targets:
        # Load the model for this target
        model = net.Net(**kwargs)
        checkpoint_path = (
            checkpoints_dir / "target-specific" / trained_on / target / "weights.pt"
        )
        model.load_state_dict(torch.load(
            checkpoint_path,
            map_location=torch.device('cuda'),
            weights_only=False
        ))
        model.eval()

        # Only predict for rows where the target is not NaN
        target_mask = test_dataset[target].notna().values
        fingerprints_masked = fingerprints_tensor[target_mask]

        # Predict
        with torch.no_grad():
            outputs = model(fingerprints_masked)
            target_predictions = outputs.squeeze().cpu().numpy()

        predictions_dict[target] = target_predictions

    return predictions_dict


def dnn_clustered_multi_target_callback(test_dataset, targets, trained_on="TVT", verbose=False):
    """
    Predicts values for multiple targets using clustered DNN models.

    Args:
        test_dataset (pd.DataFrame): DataFrame with at least 'SMILES' and the target columns.
        targets (list of str): List of target column names.
        trained_on (str): Dataset/model name for checkpoint path.
        verbose (bool): Whether to show progress bars.

    Returns:
        dict: {target: np.ndarray of predictions (NaN for missing target values)}
    """
    # Model architecture parameters (should match training)
    kwargs = {
        "input_dim": 2048,
        "hidden1_dim": 1024,
        "hidden2_dim": 128,
        "output_dim": 1,
        "drop_rate": 0.5
    }

    target_predictions = dict()

    label = trained_on.split('-')[0]
    with open(data_dir / f"target_clusters_correlation_{label}.json", "r") as f:
        target_clusters = json.load(f)

    # Precompute fingerprints for all molecules once
    smiles = test_dataset['SMILES'].values
    fingerprints = get_fingerprints(smiles, verbose=verbose)
    fingerprints_tensor = torch.tensor(fingerprints, dtype=torch.float32).to('cuda')

    for cluster_idx in target_clusters["cluster_to_targets"]:
        clustered_targets = target_clusters["cluster_to_targets"][cluster_idx]

        kwargs["num_tasks"] = len(clustered_targets)

        # Load the model for this cluster
        model = net_mt.Net(**kwargs)
        checkpoint_path = (
            checkpoints_dir / "clustered-multi-target" / trained_on / f"cluster-{cluster_idx}" / "weights.pt"
        )
        model.load_state_dict(torch.load(
            checkpoint_path,
            map_location=torch.device('cuda'),
            weights_only=False
        ))
        model.to("cuda").eval()

        # Only predict for rows where at least one target in the cluster is not NaN
        target_dataset = test_dataset.dropna(subset=clustered_targets, how="all")
        
        if verbose:
            print(
                f"Running clustered MT-DNN on {len(target_dataset)} SMILES "
                f"for {len(clustered_targets)} targets (cluster {cluster_idx})..."
            )

        # Get fingerprints for the filtered dataset
        target_mask = test_dataset.index.isin(target_dataset.index)
        fingerprints_masked = fingerprints_tensor[target_mask]

        # Predict
        with torch.no_grad():
            predictions = []
            for task_idx in range(kwargs["num_tasks"]):
                outputs = model(fingerprints_masked, task_idx)
                predictions.append(outputs.squeeze().cpu().numpy())

        # Store predictions for each target in the cluster
        for preds, target in zip(predictions, clustered_targets):
            target_specific_mask = target_dataset[target].notna()
            target_predictions[target] = preds[target_specific_mask]

    return target_predictions


def dnn_multi_target_callback(test_dataset, targets, trained_on="TVT", verbose=False):
    """
    Predicts values for multiple targets using clustered DNN models.

    Args:
        test_dataset (pd.DataFrame): DataFrame with at least 'SMILES' and the target columns.
        targets (list of str): List of target column names.
        trained_on (str): Dataset/model name for checkpoint path.
        verbose (bool): Whether to show progress bars.

    Returns:
        dict: {target: np.ndarray of predictions (NaN for missing target values)}
    """
    # Model architecture parameters (should match training)
    kwargs = {
        "input_dim": 2048,
        "hidden1_dim": 1024,
        "hidden2_dim": 128,
        "output_dim": 1,
        "drop_rate": 0.5
    }

    target_predictions = dict()

    # Precompute fingerprints for all molecules once
    smiles = test_dataset['SMILES'].values
    fingerprints = get_fingerprints(smiles, verbose=verbose)
    fingerprints_tensor = torch.tensor(fingerprints, dtype=torch.float32).to('cuda')

    kwargs["num_tasks"] = len(targets)

    # Load the model for this cluster
    model = net_mt.Net(**kwargs)
    checkpoint_path = (
        checkpoints_dir / "multi-target" / trained_on / "cluster-MT-ALL" / "weights.pt"
    )
    model.load_state_dict(torch.load(
        checkpoint_path,
        map_location=torch.device('cuda'),
        weights_only=False
    ))
    model.to("cuda").eval()

    # Only predict for rows where at least one target in the cluster is not NaN
    target_dataset = test_dataset.dropna(subset=targets, how="all")

    if verbose:
        print(
            f"Running clustered MT-DNN on {len(target_dataset)} SMILES "
            f"for {len(targets)} targets (cluster MT-ALL)..."
        )

    # Get fingerprints for the filtered dataset
    target_mask = test_dataset.index.isin(target_dataset.index)
    fingerprints_masked = fingerprints_tensor[target_mask]

    # Predict
    with torch.no_grad():
        predictions = []
        for task_idx in range(kwargs["num_tasks"]):
            outputs = model(fingerprints_masked, task_idx)
            predictions.append(outputs.squeeze().cpu().numpy())

    # Store predictions for each target in the cluster
    for preds, target in zip(predictions, targets):
        target_specific_mask = target_dataset[target].notna()
        target_predictions[target] = preds[target_specific_mask]

    return target_predictions
