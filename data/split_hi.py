import pandas as pd
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem, DataStructs
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
import lohi_splitter as lohi
import networkx as nx

OVER_N = 1000

pivoted_assays_split = pd.read_csv(
    f"./pivoted_pXC50_over_{OVER_N}_split.csv"
).drop(columns=["split"])  # .head(10_000)

smiles = pivoted_assays_split["SMILES"].to_list()
num_mols = len(smiles)

similarity_matrix = load_npz(f"similarity_matrix_{num_mols}_mols.npz")

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from tqdm import tqdm


def split_dataset_mis(smiles, similarity_matrix):
    """
    Splits the dataset into train/test sets using a Maximal Independent Set (MIS)-inspired greedy approach.

    Parameters:
        smiles (list): List of molecular SMILES strings (instances).
        similarity_matrix (csr_matrix): Sparse similarity matrix where only values > 0.4 are stored.

    Returns:
        train_set (set): SMILES strings assigned to training.
        test_set (set): SMILES strings assigned to testing.
    """
    num_instances = len(smiles)

    # Step 1: Construct a graph from the sparse similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(num_instances))

    # Add edges where similarity > 0.4
    row_idx, col_idx = similarity_matrix.nonzero()
    for i, j in tqdm(zip(row_idx, col_idx), total=len(row_idx), desc="Adding edges"):
        if i < j:  # Avoid duplicate edges
            G.add_edge(i, j)

    # Step 2: Sort nodes by degree (higher first)
    nodes_sorted = sorted(G.nodes, key=lambda n: G.degree[n], reverse=True)

    # Step 3: Initialize train and test sets
    train_set = set([nodes_sorted[0]])
    test_set = set()

    # # Step 4: Greedy selection ensuring strict separation
    for i, node in enumerate(tqdm(nodes_sorted[1:], desc="Assigning nodes")):  # Start from the second node
        # Check if this node has similarity > 0.4 with any instance in the train set
        is_similar_to_train = any(similarity_matrix[node, train_node] > 0.4 for train_node in train_set)
        is_dissimilar_to_test = any(similarity_matrix[node, test_node] <= 0.4 for test_node in test_set)

        if is_similar_to_train and (is_dissimilar_to_test or len(test_set) == 0):
            train_set.add(node)  # Assign to training set if it's similar to any training instance
        else:
            test_set.add(node)   # Otherwise, assign to test set

        if i % 10_000 == 0:
            print(f"Train set: {len(train_set):,} instances")
            print(f"Test set: {len(test_set):,} instances")

    # Convert indices to SMILES
    train_smiles = {smiles[i] for i in train_set}
    test_smiles = {smiles[i] for i in test_set}

    return train_smiles, test_smiles

print("Performing Hi splitting using MIS-inspired approach...")

train, test = split_dataset_mis(smiles, similarity_matrix)
print(f"Train set: {len(train):,} instances")
print(f"Test set: {len(test):,} instances")

# Save the split
import pandas as pd

train_df = pd.DataFrame(list(train), columns=["SMILES"])
test_df = pd.DataFrame(list(test), columns=["SMILES"])

train_df["split"] = "train"
test_df["split"] = "test"

hi_split_assays = pd.concat([train_df, test_df], ignore_index=True)
hi_split_assays.to_csv(f"pivoted_pXC50_over_{OVER_N}_split_hi.csv", index=False)

import matplotlib.pyplot as plt

def get_similar_mols(lhs, rhs, return_idx=False):
    fp_generator = GetMorganGenerator(radius=2, fpSize=1024)

    # Convert SMILES to fingerprints (vectorized using list comprehension)
    lhs_fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(smile)) for smile in lhs]
    rhs_fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(smile)) for smile in rhs]

    # Compute similarities in bulk
    nearest_sim = np.zeros(len(lhs))
    nearest_idx = np.zeros(len(lhs), dtype=int)

    for i, lhs_fp in tqdm(enumerate(lhs_fps), total=len(lhs), desc="Computing similarities"):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(lhs_fp, rhs_fps))  # Convert to NumPy for efficiency
        nearest_idx[i] = sims.argmax()
        nearest_sim[i] = sims[nearest_idx[i]]

    return (nearest_sim, nearest_idx) if return_idx else nearest_sim


nearest_sim, nearest_idx = get_similar_mols(test, train, return_idx=True)
plt.hist(nearest_sim, bins=50)
plt.axvline(x=0.4, color = 'r', ls='--')
plt.title('Maximal similarity to train')
plt.savefig('maximal_similarity.png')