from rdkit import Chem
import pandas as pd


def model_predict_wrapper(model, preprocessor, mols):
    mol_features = preprocessor.vectorized_compute_features(mols)
    df_features = pd.DataFrame(mol_features, columns=preprocessor.features)

    predicted_activity = model.predict(df_features)

    return predicted_activity


mols_from_smiles = lambda smiles: [Chem.MolFromSmiles(smi) for smi in smiles]