from pathlib import Path
import pandas as pd
import numpy as np
import lohi_splitter as lohi
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem, DataStructs
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz


SPLIT_LO = True
SPLIT_HI = False

OVER_N = 1000

pivoted_assays_split = pd.read_csv(
    f"data/pivoted_pXC50_over_{OVER_N}_split.csv"
).drop(columns=["split"])  # .head(10_000)

smiles = pivoted_assays_split["SMILES"].to_list()


def get_similar_mols(lhs, rhs, return_idx=False):
    fp_generator = GetMorganGenerator(radius=2, fpSize=1024)

    # Convert SMILES to fingerprints (vectorized using list comprehension)
    lhs_fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(smile)) for smile in lhs]
    rhs_fps = [fp_generator.GetFingerprint(Chem.MolFromSmiles(smile)) for smile in rhs]

    # Compute similarities in bulk
    nearest_sim = np.zeros(len(lhs))
    nearest_idx = np.zeros(len(lhs), dtype=int)

    for i, lhs_fp in enumerate(lhs_fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(lhs_fp, rhs_fps))  # Convert to NumPy for efficiency
        nearest_idx[i] = sims.argmax()
        nearest_sim[i] = sims[nearest_idx[i]]

    return (nearest_sim, nearest_idx) if return_idx else nearest_sim


def set_cluster_columns(data, cluster_smiles, train_smiles, smiles_column="smiles"):
    data = data.copy()
    data["cluster"] = -1
    is_train = data[smiles_column].isin(train_smiles)
    data.loc[is_train, ["cluster"]] = 0

    for i, cluster in enumerate(cluster_smiles):
        is_cluster = data[smiles_column].isin(cluster)
        data.loc[is_cluster, ["cluster"]] = i + 1

    is_in_cluster = data["cluster"] != -1
    return data[is_in_cluster]


def construct_similarity_matrix(smiles, threshold, verbose=False):
    """
    Construct a similarity matrix for a given list of SMILES.
    """
    fp_generator = GetMorganGenerator(radius=2, fpSize=1024)

    # Convert SMILES to fingerprints
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    mols_iter = tqdm(mols, desc="Generating fingerprints") if verbose else mols

    all_fps = [fp_generator.GetFingerprint(mol) for mol in mols_iter]
    num_mols = len(all_fps)

    # Store only values above threshold to optimize memory usage
    rows, cols, data = [], [], []
    all_fps_iter = tqdm(all_fps, desc="Computing pairwise similarities") if verbose else all_fps

    for i, fp in enumerate(all_fps_iter):
        sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
        valid_idx = np.where(np.array(sims) > threshold)[0]

        rows.extend([i] * len(valid_idx))
        cols.extend(valid_idx)
        data.extend(sims[j] for j in valid_idx)

    # CSR: Compressed Sparse Row format
    similarity_matrix = csr_matrix((data, (rows, cols)), shape=(num_mols, num_mols))
    return similarity_matrix


def select_distinct_clusters(smiles, threshold, min_cluster_size, max_clusters, values, std_threshold, verbose=True):
    """
    A greedy algorithm to select independent clusters from datasets.
    """
    global pivoted_assays_split
    clusters = []
    num_mols = len(smiles)

    if Path(f"similarity_matrix_{num_mols}_mols.npz").exists():
        print("Loading similarity matrix...")
        similarity_matrix = load_npz(f"similarity_matrix_{num_mols}_mols.npz")
    else:
        similarity_matrix = construct_similarity_matrix(smiles, threshold, verbose=verbose)
        # Save similarity matrix for future use
        save_npz(f"similarity_matrix_{num_mols}_mols.npz", similarity_matrix)

    while len(clusters) < max_clusters:
        if verbose:
            print(f"Clusters: {len(clusters)}/{max_clusters}")

        # Compute neighbor counts and compute std deviations efficiently
        total_neighbours = np.array(similarity_matrix.getnnz(axis=1))
        stds = np.array([values[similarity_matrix[i].indices].std() if total_neighbours[i] > 0 else 0 
                         for i in range(num_mols)])

        # Find the most distant cluster considering cluster size and std deviation
        valid_neighbours_idx = np.where(total_neighbours > min_cluster_size)[0]
        valid_stds_idx = np.where(stds > std_threshold)[0]
        valid_idx = np.intersect1d(valid_neighbours_idx, valid_stds_idx)

        central_idx = None
        least_neighbours = max(total_neighbours)
        for idx in valid_idx:
            n_neighbours = total_neighbours[idx]
            if n_neighbours >= least_neighbours:
                continue
            least_neighbours, central_idx = n_neighbours, idx

        if (central_idx is None) or (valid_idx.size == 0):
            break

        # Get cluster members
        is_neighbour = similarity_matrix[central_idx].indices

        # Remove central_idx from cluster and append to the end
        is_neighbour = is_neighbour[is_neighbour != central_idx]
        cluster_smiles = np.append(smiles[is_neighbour], smiles[central_idx])
        clusters.append(cluster_smiles.tolist())
        
        # Remove neighbors of neighbors from the rest of smiles
        nearest_sim = get_similar_mols(smiles, cluster_smiles)
        keep_mask = np.where(nearest_sim < threshold)[0]  # Use NumPy for efficiency

        smiles, values = smiles[keep_mask], values[keep_mask] 
        similarity_matrix = similarity_matrix[np.ix_(keep_mask, keep_mask)]  # Update similarity matrix
        num_mols = len(smiles)

        # Preview results
        if len(clusters) % 10 == 0 and len(clusters) > 0:
            train_smiles = list(smiles)
            
            # Move one molecule from each test cluster to the training set
            leave_one_clusters = []
            for cluster in clusters:
                train_smiles.append(cluster[-1])
                leave_one_clusters.append(cluster[:-1])

            lo_split_assays = set_cluster_columns(pivoted_assays_split, leave_one_clusters, train_smiles, smiles_column="SMILES")

            # cluster 0 means train
            lo_split_assays["lo_split"] = lo_split_assays["cluster"].apply(lambda x: "train" if x == 0 else "test")
            print(lo_split_assays.shape)
            print(pd.DataFrame({
                "#": lo_split_assays["lo_split"].value_counts().apply(lambda x: f"{x:,}"),
                "%": lo_split_assays["lo_split"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}")
            }, index=["train", "test"]))

            train = lo_split_assays[lo_split_assays["lo_split"] == "train"]
            test = lo_split_assays[lo_split_assays["lo_split"] == "test"]
            print("Minimum number of assays in each split:")
            print(pd.Series({
                "train": train.notnull().sum().min(),
                "test": test.notnull().sum().min()
            }))

            lo_split_assays.to_csv(f"lo_intermediate/{len(clusters)}.csv", index=False)

    return clusters, smiles


def lo_train_test_split(
    smiles, threshold, min_cluster_size, max_clusters, values, std_threshold
):
    """
    Lo splitter. Refer to tutorial 02_lo_split.ipynb and the paper by Simon Steshin titled "Lo-Hi: Practical ML Drug Discovery Benchmark", 2023.

    Parameters:
        smiles -- list of smiles
        threshold --  molecules with similarity larger than this number are considered similar
        min_cluster_size -- number of molecules per cluster
        max_clusters -- maximum number of selected clusters. The remaining molecules go to the training set.
        values -- values of the smiles
        std_threshold -- Lower bound of the acceptable standard deviation for a cluster. It should be greater than measurement noise.
                         If you're using ChEMBL-like data, set it to 0.60 for logKi and 0.70 for logIC50.
                         Set it lower if you have a high-quality dataset. Refer to the paper, Appendix B.

    Returns:
        clusters -- list of lists of smiles.
        train_smiles -- list of train smiles
    """
    if not isinstance(smiles, np.ndarray):
        smiles = np.array(smiles)
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    cluster_smiles, train_smiles = select_distinct_clusters(
        smiles, threshold, min_cluster_size, max_clusters, values, std_threshold
    )
    train_smiles = list(train_smiles)
    # Move one molecule from each test cluster to the training set
    leave_one_clusters = []
    for cluster in cluster_smiles:
        train_smiles.append(cluster[-1])
        leave_one_clusters.append(cluster[:-1])

    return leave_one_clusters, train_smiles


def split_lo(smiles, values):
    """
    Reference: https://github.com/SteshinSS/lohi_splitter/blob/main/tutorial/02_lo_split.ipynb
    """
    # Similarity threshold for clustering molecules.
    # Molecules are considered similar if their ECFP4 Tanimoto Similarity is larger than this threshold.
    threshold = 0.4

    ## The minimum number of molecules required in a cluster.
    min_cluster_size = 5

    # Maximum number of clusters to be created. Any additional molecules are added to the training set.
    max_clusters = 5000

    # Minimum standard deviation of values within a cluster.
    # This ensures that clusters with too little variability are filtered out.
    # For further details, refer to Appendix B of the paper.
    std_threshold = 0.60

    cluster_smiles, train_smiles = lo_train_test_split(smiles=smiles, 
                                                       threshold=threshold, 
                                                       min_cluster_size=min_cluster_size, 
                                                       max_clusters=max_clusters, 
                                                       values=values, 
                                                       std_threshold=std_threshold)
    return cluster_smiles, train_smiles


def solve_linear_problem(m, max_mip_gap=0.1, verbose=True, **optimize_kwargs):
    """
    Solves MIP linear model with default parameters.
    """
    m.max_mip_gap = max_mip_gap
    m.threads = -1
    m.emphasis = 2
    m.verbose = verbose
    m.optimize(**optimize_kwargs)
    return m


def coarse_hi_split(smiles, threshold=0.4):
    """
    Reference: https://github.com/SteshinSS/lohi_splitter/blob/main/tutorial/01_hi_split_coarsening.ipynb
    """
    # Define a threshold for similarity. Molecules with similarity > 0.4 are considered similar.
    similarity_threshold = 0.4

    # Set fractions for the train and test sets.
    # Increase their sum to discard fewer molecules. Decrease it to speed up computations.
    train_min_frac = 0.70
    test_min_frac = 0.10

    # Threshold for graph clustering.
    # Increase it to discard fewer molecules. Decrease it to speed up computations.
    coarsening_threshold = 0.4

    # How close we should be to the theoretical optimum to terminate the optimization.
    # Should be in [0, 1].
    # Decrease it to discard fewer molecules. Increase it to speed up computations.
    max_mip_gap = 0.01

    partition = lohi.hi_train_test_split(smiles=smiles,
                                        similarity_threshold=similarity_threshold,
                                        train_min_frac=train_min_frac,
                                        test_min_frac=test_min_frac,
                                        coarsening_threshold=coarsening_threshold,
                                        max_mip_gap=max_mip_gap)


if __name__ == "__main__":
    if SPLIT_LO:
        # Set all columns dtype to float
        values = pivoted_assays_split.drop(columns="SMILES").astype(float).mean(axis=1, skipna=True).to_list()

        cluster_smiles, train_smiles = split_lo(smiles, values)
        lo_split_assays = set_cluster_columns(pivoted_assays_split, cluster_smiles, train_smiles, smiles_column="SMILES")

        # cluster 0 means train
        lo_split_assays["lo_split"] = lo_split_assays["cluster"].apply(lambda x: "train" if x == 0 else "test")
        print(lo_split_assays.shape)
        print(lo_split_assays["lo_split"].value_counts())

        lo_split_assays.to_csv(f"data/lo_pivoted_pXC50_over_{OVER_N}_split.csv", index=False)

    # if SPLIT_HI:
    #     # Overwrite the lohi solve_linear_problem function to set optimization parameters
    #     lohi.solve_linear_problem = solve_linear_problem
    
    #     coarse_hi_split(smiles, threshold=0.4)
