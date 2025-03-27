from pathlib import Path
import sys
import os
import concurrent.futures

bambu_path = Path(".") / "bambu-v2"
# os.chdir(bambu_path)
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

N_PROC = 4  # Set the maximum number of parallel processes
RETRAIN = False

input_path = "../data/pivoted_pXC50_over_1000_split.csv"
activities_df = pd.read_csv(input_path)
activities_df = activities_df[activities_df["split"].isin(["train", "val"])]
activities_df["InChI"] = activities_df["SMILES"].parallel_apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)))

ESTIMATORS = ["decision_tree", "gradient_boosting", "svm", 'rf', 'extra_tree']
targets = activities_df.drop(columns=['SMILES', 'InChI', 'split']).columns.tolist()

def train_model(target):
    print(f"Training model for {target}")
    df_input = activities_df[['InChI', target]].rename(columns={target: 'activity'})
    df_input = df_input.dropna(subset=['activity'])
    
    preprocessed_df = preprocess(df_input, output_file=None, output_preprocessor_file="../checkpoints/bambu/Morgan.pkl", feature_type="morgan-2048")
    preprocessed_df["activity"] = df_input["activity"].reset_index(drop=True)
    
    train(preprocessed_df, f"../checkpoints/bambu/{target}.pkl", estimators=ESTIMATORS, task="regression")


print("\nTotal targets:", len(targets))
if not RETRAIN:
    targets = [target for target in targets if not Path(f"../checkpoints/bambu/{target}.pkl").exists()]

print("Targets to train:", len(targets))
targets = ['ACHE', 'ADORA1', 'ADORA2A', 'ALDH1A1', 'ALOX15', 'ALOX15B', 'APOBEC3F', 'APOBEC3G', 'ATAD5', 'ATXN2', 'BAZ2B', 'BRCA1', 'CBX1', 'CGA', 'CNR1', 'CNR2', 'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4', 'DRD2', 'EGFR', 'EHMT2', 'ERG', 'ESR1', 'EYA2', 'F10', 'F2', 'FEN1', 'GAA', 'GBA', 'GLA', 'GLP1R', 'GLS', 'GMNN', 'GNAS', 'GSK3B', 'HPGD', 'HSD17B10', 'HSF1', 'HTR1A', 'HTT', 'IDH1', 'IMPA1', 'KAT2A', 'KCNH2', 'KDM4A', 'KDM4E', 'KDR', 'KMT2A', 'L3MBTL1', 'LMNA', 'MAPK1', 'MAPK14', 'MAPT', 'MBNL1', 'MTOR', 'NFE2L2', 'NPC1', 'NPSR1', 'OPRD1', 'OPRM1', 'PIN1', 'PKM', 'PLK1', 'POLB', 'POLH', 'POLI', 'POLK', 'PTH1R', 'RAB9A', 'RECQL', 'RGS4', 'RORC', 'RXFP1', 'SLC6A3', 'SLC6A4', 'SMAD3', 'SMN1', 'SMN2', 'SMPD1', 'SNCA', 'TARDBP', 'TDP1', 'THRB', 'TP53', 'TSHR', 'TXNRD1', 'USP2', 'VDR', 'WRN']

# with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROC) as executor:
#     executor.map(train_model, targets)
for target in targets:
    train_model(target)
