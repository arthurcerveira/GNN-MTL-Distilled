from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pickle
import sys
from pathlib import Path

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

current_dir = Path(__file__).parent
BAMBU = current_dir / '..' / "baselines" / "bambu-v2"
sys.path.append(str(BAMBU))

checkpoints_dir = current_dir / '..' / "checkpoints" / "bambu"

import bambu


def smiles_to_fingerprints(smiles, preprocessor):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fingerprints = preprocessor.vectorized_compute_features(mols)

    return fingerprints


def bambu_callback(test_dataset, targets, trained_on="TVT", verbose=False):
    smiles = test_dataset['SMILES'].values
    preprocessor_file = checkpoints_dir / trained_on / "Morgan.pkl"
    with open(preprocessor_file, 'rb') as preprocessor_reader:
        preprocessor = pickle.loads(preprocessor_reader.read())
    
    fingerprints = smiles_to_fingerprints(smiles, preprocessor)
    fingerprints_df = pd.DataFrame(
        fingerprints, columns=preprocessor.features, index=test_dataset.index
    )

    target_predictions = dict()
    targets_iter = tqdm(targets) if verbose else targets
    for target in targets_iter:
        if verbose:
            targets_iter.set_description(f"Running Bambu on target {target}...")

        with open(checkpoints_dir / trained_on / f'{target}.pkl', 'rb') as model_reader:
            model = pickle.loads(model_reader.read())
        
        target_mask = test_dataset[target].notna()
        target_df = fingerprints_df[target_mask]
        
        predictions = model.predict(target_df)
        target_predictions[target] = predictions
    
    return target_predictions
