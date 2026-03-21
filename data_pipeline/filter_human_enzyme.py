import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
DOWNLOAD_DIR = "./bindingdb_data"
OUTPUT_DIR = "./bindingdb_data/processed"
CHUNK_SIZE = 100_000  # rows per chunk — uses ~200MB RAM per chunk

# Auto-detect TSV
TSV_PATH = None
abs_dir = os.path.abspath(DOWNLOAD_DIR)
if os.path.exists(abs_dir):
    for f in os.listdir(abs_dir):
        if f.endswith('.tsv'):
            TSV_PATH = os.path.join(abs_dir, f)
            break

if TSV_PATH is None:
    raise FileNotFoundError(f"No TSV file found in {abs_dir}. Run step1 first.")

# Only load the columns we actually need — this is the key speedup
# Loading 11 columns instead of 640 is ~60x less data to parse
COLS_WE_NEED = [
    'Ligand SMILES',
    'Target Name',
    'Target Source Organism According to Curator or DataSource',
    'Ki (nM)',
    'IC50 (nM)',
    'Kd (nM)',
    'UniProt (SwissProt) Primary ID of Target Chain 1',
    'PubChem CID',
    'ChEMBL ID of Ligand',
    'Ligand InChI Key',
    'BindingDB Reactant_set_id',
]


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 2: Filter BindingDB (CHUNKED — fast version)               ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nInput: {TSV_PATH}")
    print(f"File size: {os.path.getsize(TSV_PATH) / 1e9:.2f} GB")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    
    # --- Check which columns exist ---
    print("\nReading header...")
    header = pd.read_csv(TSV_PATH, sep='\t', nrows=0)
    valid_cols = [c for c in COLS_WE_NEED if c in header.columns]
    missing = [c for c in COLS_WE_NEED if c not in header.columns]
    
    print(f"Loading {len(valid_cols)} columns: {valid_cols}")
    if missing:
        print(f"[WARNING] Missing (non-critical): {missing}")
    
    # --- Process in chunks ---
    print(f"\nProcessing file in chunks of {CHUNK_SIZE:,} rows...")
    
    chunks_kept = []
    total_rows = 0
    human_rows = 0
    
    reader = pd.read_csv(
        TSV_PATH,
        sep='\t',
        usecols=valid_cols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines='skip',
    )
    
    org_col = 'Target Source Organism According to Curator or DataSource'
    uniprot_col = 'UniProt (SwissProt) Primary ID of Target Chain 1'
    
    for chunk_num, chunk in enumerate(reader):
        total_rows += len(chunk)
        
        # Filter: human only
        if org_col in chunk.columns:
            mask = chunk[org_col].str.lower().isin(['homo sapiens', 'human'])
            chunk = chunk[mask]
        
        if len(chunk) == 0:
            if chunk_num % 10 == 0:
                print(f"  Chunk {chunk_num}: {total_rows:,} rows scanned, {human_rows:,} human kept")
            continue
        
        human_rows += len(chunk)
        chunks_kept.append(chunk)
        
        if chunk_num % 10 == 0:
            print(f"  Chunk {chunk_num}: {total_rows:,} rows scanned, {human_rows:,} human kept")
    
    print(f"\n[OK] Finished reading file.")
    print(f"  Total rows scanned: {total_rows:,}")
    print(f"  Human rows kept:    {human_rows:,}")
    
    # --- Combine ---
    print("\nCombining filtered chunks...")
    df = pd.concat(chunks_kept, ignore_index=True)
    del chunks_kept  # free memory
    print(f"Combined: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.0f} MB in RAM")
    
    # --- Clean binding measurements ---
    print("\n" + "=" * 70)
    print("Cleaning binding measurements...")
    print("=" * 70)
    
    df['binding_value_nM'] = np.nan
    df['binding_type'] = ''
    
    # Priority: Ki > Kd > IC50 (last one written wins)
    for col, btype in [('IC50 (nM)', 'IC50'), ('Kd (nM)', 'Kd'), ('Ki (nM)', 'Ki')]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            mask = vals.notna() & (vals > 0)
            df.loc[mask, 'binding_value_nM'] = vals[mask]
            df.loc[mask, 'binding_type'] = btype
            print(f"  {col}: {mask.sum():,} valid numeric values")
    
    before = len(df)
    df = df[df['binding_value_nM'].notna()].copy()
    print(f"\nRows with valid binding: {before:,} -> {len(df):,}")
    
    # pActivity = -log10(value_in_M) = 9 - log10(value_in_nM)
    df['pActivity'] = 9 - np.log10(df['binding_value_nM'])
    df = df[(df['pActivity'] >= 0) & (df['pActivity'] <= 15)].copy()
    print(f"After outlier removal:  {len(df):,}")
    
    print(f"\nBinding type distribution:")
    for bt, count in df['binding_type'].value_counts().items():
        print(f"  {bt:10s}: {count:>10,}")
    
    print(f"\npActivity stats:")
    print(f"  Mean:   {df['pActivity'].mean():.2f}")
    print(f"  Median: {df['pActivity'].median():.2f}")
    print(f"  Std:    {df['pActivity'].std():.2f}")
    print(f"  Range:  [{df['pActivity'].min():.2f}, {df['pActivity'].max():.2f}]")
    
    # --- Validate SMILES and UniProt ---
    print("\n" + "=" * 70)
    print("Validating identifiers...")
    print("=" * 70)
    
    smiles_col = 'Ligand SMILES'
    before = len(df)
    df = df[df[smiles_col].notna() & (df[smiles_col].str.len() > 0)].copy()
    print(f"With valid SMILES:  {before:,} -> {len(df):,}")
    
    if uniprot_col in df.columns:
        before = len(df)
        df = df[df[uniprot_col].notna() & (df[uniprot_col].str.len() > 0)].copy()
        df['uniprot_id'] = df[uniprot_col].str.split(',').str[0].str.strip()
        print(f"With valid UniProt: {before:,} -> {len(df):,}")
    
    df = df.rename(columns={smiles_col: 'smiles'})
    
    # --- Binary labels ---
    print("\n" + "=" * 70)
    print("Creating labels...")
    print("=" * 70)
    
    df['label_binary'] = np.nan
    df.loc[df['pActivity'] >= 6, 'label_binary'] = 1   # binder (<1 uM)
    df.loc[df['pActivity'] < 5, 'label_binary'] = 0    # non-binder (>10 uM)
    
    binders = (df['label_binary'] == 1).sum()
    non_binders = (df['label_binary'] == 0).sum()
    ambiguous = df['label_binary'].isna().sum()
    
    print(f"  Binders (pAct >= 6):     {binders:>10,}")
    print(f"  Non-binders (pAct < 5):  {non_binders:>10,}")
    print(f"  Ambiguous (5-6):         {ambiguous:>10,}")
    
    n_drugs = df['smiles'].nunique()
    n_targets = df['uniprot_id'].nunique() if 'uniprot_id' in df.columns else 'N/A'
    
    print(f"\n  Total rows:      {len(df):>10,}")
    print(f"  Unique drugs:    {n_drugs:>10,}")
    print(f"  Unique targets:  {n_targets:>10}")
    
    # --- Save ---
    print("\n" + "=" * 70)
    print("Saving...")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    out_cols = ['smiles', 'uniprot_id', 'Target Name',
                'binding_value_nM', 'binding_type', 'pActivity', 'label_binary']
    optional = ['BindingDB Reactant_set_id', 'PubChem CID',
                'ChEMBL ID of Ligand', 'Ligand InChI Key']
    for c in optional:
        if c in df.columns:
            out_cols.append(c)
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols]
    
    path = os.path.join(OUTPUT_DIR, "bindingdb_human_clean.csv")
    df_out.to_csv(path, index=False)
    print(f"[SAVED] {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    
    df_bin = df_out[df_out['label_binary'].notna()]
    path = os.path.join(OUTPUT_DIR, "bindingdb_human_binary.csv")
    df_bin.to_csv(path, index=False)
    print(f"[SAVED] {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    
    drugs = df_out[['smiles']].drop_duplicates()
    path = os.path.join(OUTPUT_DIR, "unique_drugs.csv")
    drugs.to_csv(path, index=False)
    print(f"[SAVED] {path} ({len(drugs):,} compounds)")
    
    if 'uniprot_id' in df_out.columns:
        targets = df_out[['uniprot_id', 'Target Name']].drop_duplicates(subset='uniprot_id')
        path = os.path.join(OUTPUT_DIR, "unique_targets.csv")
        targets.to_csv(path, index=False)
        print(f"[SAVED] {path} ({len(targets):,} proteins)")
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 2 COMPLETE                                                ║")
    print("╠" + "═" * 68 + "╣")
    print("║   Next: Run step3_build_drug_graphs.py                            ║")
    print("╚" + "═" * 68 + "╝")