# %%
# !pip install duckdb pandarallel ipywidgets

# %%
import duckdb

excape_path = './data/Full-ExCAPE.tsv'

con = duckdb.connect(database=':memory:', read_only=False)
con.execute(f'CREATE TABLE excape AS SELECT * FROM read_csv_auto(\'{excape_path}\')')

con.execute('SELECT * FROM excape LIMIT 10').fetchdf()

# %%
print(f'Number of rows: {con.execute("SELECT COUNT(*) FROM excape").fetchdf().values[0][0]:,}')

unique_genes = con.execute('SELECT DISTINCT Gene_Symbol FROM excape').fetchdf()

print(f'Number of unique genes: {unique_genes.shape[0]:,}')
print(unique_genes.sample(5))

# %%
# Save pXC50 activity dataset
from smiles_utils import clean_smile
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(progress_bar=False, nb_workers=6)


idx = 0
min_assays_threshold = 1000
pXC50_activity_dataset = pd.DataFrame(columns=["SMILES", "pXC50", "Gene", "Gene_idx"])

for gene in tqdm(unique_genes["Gene_Symbol"]):
    gene_rows = con.execute(f'SELECT * FROM excape WHERE Gene_Symbol = \'{gene}\'').fetchdf()

    # Clean and remove duplicates
    gene_rows = gene_rows.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"])
    assay_pXC50 = gene_rows[["SMILES", "pXC50"]].copy()
    assay_pXC50["pXC50"] = assay_pXC50["pXC50"].astype(float)
    assay_pXC50 = assay_pXC50.dropna(subset=["pXC50"])

    # First skip: not enough assays to clean the SMILES
    if len(assay_pXC50) < min_assays_threshold:
        continue

    assay_pXC50["SMILES"] = assay_pXC50["SMILES"].parallel_apply(clean_smile)
    assay_pXC50 = assay_pXC50.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"])
    assay_pXC50["pXC50"] = assay_pXC50["pXC50"].clip(0, 10)
    
    # Second skip: cleaning the SMILES filtered out invalid strings
    if len(assay_pXC50) < min_assays_threshold:
        continue

    assay_pXC50["Gene"] = gene
    assay_pXC50["Gene_idx"] = idx
    pXC50_activity_dataset = pd.concat([pXC50_activity_dataset, assay_pXC50], ignore_index=True)
    idx += 1

pXC50_activity_dataset.to_csv(f"./data/pXC50_activity_dataset_over_{min_assays_threshold}.csv", index=False)
print(f"Number of genes with at least {min_assays_threshold} assays: {idx:,}")
print(f"Number of assays: {len(pXC50_activity_dataset):,}")
print(f"Number of unique SMILES: {pXC50_activity_dataset['SMILES'].nunique():,}")
print(f"Number of unique genes: {pXC50_activity_dataset['Gene'].nunique():,}")
sampled_assays = pXC50_activity_dataset.sample(5)
print(sampled_assays)
