import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR = "./bindingdb_data/processed"
DRUG_GRAPH_PATH = "./bindingdb_data/processed/drug_graphs/drug_graphs.pkl"
ENZYME_GRAPH_PATH = "./bindingdb_data/processed/enzyme_graphs/enzyme_graphs.pkl"
OUTPUT_DIR = "./bindingdb_data/final_dataset"

TASK = "regression"       # "regression" or "binary"
SPLIT_MODE = "target"     # "target", "drug", or "random"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 5: Assemble Final Dataset (memory-efficient)               ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Task:       {TASK}")
    print(f"  Split mode: {SPLIT_MODE}")
    print(f"  Ratios:     {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")

    # ------------------------------------------------------------------
    # 5A: Load binding data
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5A: Loading binding data")
    print("=" * 70)

    if TASK == "binary":
        bp = os.path.join(INPUT_DIR, "bindingdb_human_binary.csv")
    else:
        bp = os.path.join(INPUT_DIR, "bindingdb_human_clean.csv")

    df = pd.read_csv(bp)
    print(f"Loaded {len(df):,} rows")

    # ------------------------------------------------------------------
    # 5B: Load enzyme graphs (small — ~484 MB, fits in RAM easily)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5B: Loading enzyme graphs")
    print("=" * 70)

    with open(ENZYME_GRAPH_PATH, 'rb') as f:
        enzyme_graphs = pickle.load(f)
    print(f"Loaded {len(enzyme_graphs):,} enzyme graphs ({os.path.getsize(ENZYME_GRAPH_PATH)/1e6:.0f} MB on disk)")

    # Filter binding data to only enzymes we have graphs for
    available_enzymes = set(enzyme_graphs.keys())
    before = len(df)
    df = df[df['uniprot_id'].isin(available_enzymes)].copy()
    print(f"Rows with available enzyme graph: {before:,} -> {len(df):,}")

    # ------------------------------------------------------------------
    # 5C: Find which drug SMILES we actually need
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5C: Identifying needed drug graphs")
    print("=" * 70)

    needed_smiles = set(df['smiles'].unique())
    print(f"Unique drugs in binding data: {len(needed_smiles):,}")
    print(f"(We only need to load these, not all 912K)")

    # ------------------------------------------------------------------
    # 5D: Load drug graphs — ONLY the ones we need
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5D: Loading drug graphs (filtered — this is the slow part)")
    print("=" * 70)
    print(f"Drug graph file: {DRUG_GRAPH_PATH} ({os.path.getsize(DRUG_GRAPH_PATH)/1e9:.1f} GB)")
    print("Loading full pickle (unavoidable, but we discard unneeded entries after)...")

    with open(DRUG_GRAPH_PATH, 'rb') as f:
        all_drug_graphs = pickle.load(f)
    print(f"Loaded {len(all_drug_graphs):,} total drug graphs")

    # Keep only what we need
    drug_graphs = {s: all_drug_graphs[s] for s in needed_smiles if s in all_drug_graphs}
    del all_drug_graphs  # free ~8 GB
    print(f"Kept {len(drug_graphs):,} needed drug graphs")
    
    import gc
    gc.collect()

    # Filter binding data to drugs we have graphs for
    available_drugs = set(drug_graphs.keys())
    before = len(df)
    df = df[df['smiles'].isin(available_drugs)].copy()
    print(f"Rows with both drug + enzyme graph: {before:,} -> {len(df):,}")

    # ------------------------------------------------------------------
    # 5E: Deduplicate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5E: Deduplicating drug-target pairs")
    print("=" * 70)

    before = len(df)
    agg = {'pActivity': 'median'}
    if 'label_binary' in df.columns:
        agg['label_binary'] = 'first'
    if 'Target Name' in df.columns:
        agg['Target Name'] = 'first'
    if 'binding_type' in df.columns:
        agg['binding_type'] = 'first'

    df = df.groupby(['smiles', 'uniprot_id']).agg(agg).reset_index()
    print(f"Deduplicated: {before:,} -> {len(df):,} unique drug-target pairs")

    n_drugs = df['smiles'].nunique()
    n_targets = df['uniprot_id'].nunique()
    print(f"  Unique drugs:   {n_drugs:,}")
    print(f"  Unique targets: {n_targets:,}")
    print(f"  Density:        {len(df)/(n_drugs*n_targets)*100:.4f}%")

    # ------------------------------------------------------------------
    # 5F: Split
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"5F: Splitting ({SPLIT_MODE} split)")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    if SPLIT_MODE == "random":
        train_df, temp = train_test_split(df, test_size=VAL_RATIO+TEST_RATIO, random_state=RANDOM_SEED)
        val_df, test_df = train_test_split(temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=RANDOM_SEED)

    elif SPLIT_MODE == "target":
        targets = df['uniprot_id'].unique()
        np.random.shuffle(targets)
        n_tr = int(len(targets) * TRAIN_RATIO)
        n_va = int(len(targets) * VAL_RATIO)
        train_t = set(targets[:n_tr])
        val_t = set(targets[n_tr:n_tr+n_va])
        test_t = set(targets[n_tr+n_va:])
        train_df = df[df['uniprot_id'].isin(train_t)]
        val_df = df[df['uniprot_id'].isin(val_t)]
        test_df = df[df['uniprot_id'].isin(test_t)]
        print(f"  Train targets: {len(train_t):,}, Val: {len(val_t):,}, Test: {len(test_t):,}")
        assert len(train_t & test_t) == 0, "Target leak!"

    elif SPLIT_MODE == "drug":
        drugs = df['smiles'].unique()
        np.random.shuffle(drugs)
        n_tr = int(len(drugs) * TRAIN_RATIO)
        n_va = int(len(drugs) * VAL_RATIO)
        train_d = set(drugs[:n_tr])
        val_d = set(drugs[n_tr:n_tr+n_va])
        test_d = set(drugs[n_tr+n_va:])
        train_df = df[df['smiles'].isin(train_d)]
        val_df = df[df['smiles'].isin(val_d)]
        test_df = df[df['smiles'].isin(test_d)]
        print(f"  Train drugs: {len(train_d):,}, Val: {len(val_d):,}, Test: {len(test_d):,}")
        assert len(train_d & test_d) == 0, "Drug leak!"

    print(f"\n  Train: {len(train_df):>10,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):>10,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):>10,} ({len(test_df)/len(df)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 5G: Save in batches
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5G: Saving final dataset")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    BATCH_SIZE = 50_000

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_df = split_df.reset_index(drop=True)
        n_batches = (len(split_df) + BATCH_SIZE - 1) // BATCH_SIZE
        
        all_samples = []
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(split_df))
            batch = split_df.iloc[start:end]

            for _, row in batch.iterrows():
                sample = {
                    'smiles': row['smiles'],
                    'uniprot_id': row['uniprot_id'],
                    'drug_graph': drug_graphs[row['smiles']],
                    'enzyme_graph': enzyme_graphs[row['uniprot_id']],
                    'pActivity': float(row['pActivity']),
                }
                if 'label_binary' in row and pd.notna(row.get('label_binary')):
                    sample['label_binary'] = int(row['label_binary'])
                all_samples.append(sample)

            print(f"  {split_name}: processed batch {batch_idx+1}/{n_batches} ({end:,} / {len(split_df):,})")

        path = os.path.join(OUTPUT_DIR, f"{split_name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [SAVED] {path} ({len(all_samples):,} samples, {size_mb:.1f} MB)\n")

    # Save CSVs for inspection
    for name, sdf in [('train', train_df), ('val', val_df), ('test', test_df)]:
        sdf.to_csv(os.path.join(OUTPUT_DIR, f"{name}_pairs.csv"), index=False)

    # Save metadata
    meta = {
        'task': TASK,
        'split_mode': SPLIT_MODE,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'n_unique_drugs': n_drugs,
        'n_unique_targets': n_targets,
        'drug_node_feature_dim': list(drug_graphs.values())[0]['node_features'].shape[1],
        'enzyme_node_feature_dim': list(enzyme_graphs.values())[0]['node_features'].shape[1],
        'contact_distance': 8.0,
        'random_seed': RANDOM_SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), 'wb') as f:
        pickle.dump(meta, f)

    print("=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)
    for k, v in meta.items():
        print(f"  {k:30s}: {v}")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 5 COMPLETE — DATASET READY FOR TRAINING!                  ║")
    print("╠" + "═" * 68 + "╣")
    print("║   Output: ./bindingdb_data/final_dataset/                         ║")
    print("║                                                                    ║")
    print("║   Each sample contains:                                            ║")
    print("║     sample['drug_graph']    — molecular graph (nodes, edges)       ║")
    print("║     sample['enzyme_graph']  — residue contact graph                ║")
    print("║     sample['pActivity']     — binding affinity label               ║")
    print("╚" + "═" * 68 + "╝")