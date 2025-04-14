from pathlib import Path
import sys
import os
import concurrent.futures

current_file_dir = Path(__file__).resolve().parent
bambu_path = current_file_dir / "bambu-v2"
sys.path.append(str(bambu_path))

# Comment line "@wraps(super_class._get_param_names)" on flaml/default/estimator.py
from bambu.preprocess import preprocess
from bambu.train import train
from bambu.predict import predict

import pandas as pd
from rdkit import Chem
# Disable rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

DATASETS = {
    "TVT": "pivoted_pXC50_over_1000_split.csv",
    "TVT-KD": "pivoted_pXC50_over_1000_split_distilled.csv",
    "Lo": "pivoted_pXC50_over_1000_split_lo.csv",
    "Hi": "pivoted_pXC50_over_1000_split_hi.csv",
}
DATASET = "Lo"
RETRAIN = False

checkpoints_dir = current_file_dir / ".." / "checkpoints"
results_dir = current_file_dir / ".." / "results"
data_dir = current_file_dir / ".." / "data"

checkpoints_path = checkpoints_dir / "bambu" / DATASET
checkpoints_path.mkdir(parents=True, exist_ok=True)

input_path = data_dir / DATASETS[DATASET]
activities_df = pd.read_csv(input_path)
activities_df = activities_df[activities_df["split"].isin(["train", "val"])]
activities_df["InChI"] = activities_df["SMILES"].parallel_apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)))

ESTIMATORS = ["decision_tree", "gradient_boosting", "svm", 'rf', 'extra_tree']
targets = activities_df.drop(columns=['SMILES', 'InChI', 'split']).columns.tolist()


def train_model(target):
    df_input = activities_df[['InChI', target]].rename(columns={target: 'activity'})
    df_input = df_input.dropna(subset=['activity'])
    
    preprocessed_df = preprocess(
        df_input,
        output_file=None,
        output_preprocessor_file=checkpoints_path / "Morgan.pkl",
        feature_type="morgan-2048"
    )
    print(f"Training model for {target} with {len(preprocessed_df)} samples")
    preprocessed_df["activity"] = df_input["activity"].reset_index(drop=True)
    
    print(f"Checkpoint path: {(checkpoints_path / f'{target}.pkl').resolve()}")
    train(preprocessed_df, checkpoints_path / f"{target}.pkl", estimators=ESTIMATORS, task="regression")


print("\nTotal targets:", len(targets))
if not RETRAIN:
    targets = [target for target in targets if not (checkpoints_path / f"{target}.pkl").exists()]

print("Targets to train:", len(targets))
for target in targets:
    train_model(target)
