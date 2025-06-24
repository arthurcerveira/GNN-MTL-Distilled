import sys
from pathlib import Path
from collections import namedtuple
import json

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler

current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "dnn" / "src" / "ST"))
sys.path.append(str(current_dir / "dnn" / "src" / "MT"))

from st_train import main as st_main
from mt_train import main as mt_main



DATASETS = {
    "TVT": "pivoted_pXC50_over_1000_split.csv",
    "TVT-KD": "pivoted_pXC50_over_1000_split_distilled.csv",
    "Lo": "pivoted_pXC50_over_1000_split_lo.csv",
    "Lo-KD": "pivoted_pXC50_over_1000_split_lo_distilled.csv",
    "Hi": "pivoted_pXC50_over_1000_split_hi.csv",
    "Hi-KD": "pivoted_pXC50_over_1000_split_hi_distilled.csv",
}
DATASET = "TVT" if len(sys.argv) == 1 else sys.argv[1]

TRAIN_TARGET_SPECIFIC = False
TRAIN_ALL_MULTI_TARGET = False
TRAIN_CLUSTERED_MULTI_TARGET = True
RETRAIN = False

DATA_PATH = current_dir / ".." / "data"
CHECKPOINTS_PATH = current_dir / ".." / "checkpoints" / "dnn"
VERBOSE = True

def get_fingerprints(dataset, targets):
    smiles = dataset["SMILES"].values
    fp_generator = GetMorganGenerator(radius=2, fpSize=2048)

    # Convert SMILES to fingerprints
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    mols_iter = tqdm(mols, desc="Generating fingerprints") if VERBOSE else mols

    all_fps = [fp_generator.GetFingerprint(mol) for mol in mols_iter]
    fps_array = np.array(all_fps)
    fps_array_df = pd.DataFrame(fps_array, index=dataset.index)

    train_fps = fps_array_df[dataset["split"] == "train"]
    val_fps = fps_array_df[dataset["split"] == "val"]
    test_fps = fps_array_df[dataset["split"] == "test"]

    if len(val_fps) == 0:
        train_fps, val_fps = train_test_split(train_fps, test_size=0.1, random_state=1907)

    dataset_labels = dataset.fillna(-1)

    train_labels = dataset_labels.loc[train_fps.index, targets].values
    val_labels = dataset_labels.loc[val_fps.index, targets].values
    test_labels = dataset_labels.loc[test_fps.index, targets].values

    if VERBOSE:
        print(f"Train FPS: {train_fps.shape}, Val FPS: {val_fps.shape}, Test FPS: {test_fps.shape}")
        print(f"Train Labels: {train_labels.shape}, Val Labels: {val_labels.shape}, Test Labels: {test_labels.shape}")

    return (
        train_fps.values, val_fps.values, test_fps.values,
        train_labels, val_labels, test_labels,
    )


def train_single_target(args, datasets):
    if VERBOSE:
        print(f"Training single-target model for {DATASET} dataset")
    checkpoints_path = CHECKPOINTS_PATH / "target-specific" / DATASET
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    args.checkpoint = str(checkpoints_path)

    st_main(args, datasets, RETRAIN)


def train_clustered_multi_target(args, datasets):
    if VERBOSE:
        print(f"Training clustered multi-target model for {DATASET} dataset")
    checkpoints_path = CHECKPOINTS_PATH / "clustered-multi-target" / DATASET
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    args.checkpoint = str(checkpoints_path)

    label = DATASET.split("-")[0]
    cluster_tasks_json = DATA_PATH / f"target_clusters_correlation_{label}.json"
    with open(cluster_tasks_json, 'r') as f:
        cluster_to_targets = json.load(f)['cluster_to_targets']

    args.cluster_to_targets = cluster_to_targets
    mt_main(args, datasets, RETRAIN)


def train_multi_target(args, datasets):
    if VERBOSE:
        print(f"Training multi-target model for {DATASET} dataset")
    checkpoints_path = CHECKPOINTS_PATH / "multi-target" / DATASET
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    args.checkpoint = str(checkpoints_path)

    # Train MT model for all targets simultaneously
    *_, targets = datasets
    args.cluster_to_targets = {
        "MT-ALL": list(targets),
    }

    mt_main(args, datasets, RETRAIN)


if __name__ == "__main__":
    args = namedtuple("Args", [
        "config", "batch_size", "epochs", "learning_rate", 
        "drop_rate", "seed", "patience", "trial", "ver", 
        "cluster", "cluster_first", "cluster_last", "gpu",
        "distance", "hidden1", "hidden2", "cluster_tasks_json"
    ])
    args.config = current_dir / "dnn" / "src" / "config.yaml"
    args.batch_size = 256
    args.epochs = 50
    args.learning_rate = 1e-3
    args.trial = 0
    args.ver = None
    args.cluster = None
    args.cluster_first = None
    args.cluster_last = None
    args.gpu = 0
    args.distance = 430
    args.hidden1 = 1024
    args.hidden2 = 128
    args.drop_rate = 0.5
    args.seed = 0
    args.patience = 5
    dataset = pd.read_csv(DATA_PATH / DATASETS[DATASET])
    targets = dataset.columns
    targets = [t for t in targets if t not in ("SMILES", "split", "cluster")]
    train_fps, val_fps, test_fps, train_labels, val_labels, test_labels = get_fingerprints(dataset, targets)
    datasets = (
        train_fps, train_labels,
        val_fps, val_labels,
        test_fps, test_labels,
        targets
    )

    if TRAIN_TARGET_SPECIFIC:
        train_single_target(args, datasets)
    elif TRAIN_ALL_MULTI_TARGET:
        train_multi_target(args, datasets)
    elif TRAIN_CLUSTERED_MULTI_TARGET:
        train_clustered_multi_target(args, datasets)

