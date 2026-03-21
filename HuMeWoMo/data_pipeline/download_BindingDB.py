import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIG - Change these paths if needed
# =============================================================================
DOWNLOAD_DIR = "./bindingdb_data"
ZIP_FILE = os.path.join(DOWNLOAD_DIR, "BindingDB_All_tsv.zip")
TSV_FILE = None  # Will be set after unzipping

# BindingDB download URL (check https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp for latest)
# The filename changes with each release. Update if needed.
BINDINGDB_URL = "https://www.bindingdb.org/bind/downloads/BindingDB_All_2D_202411_tsv.zip"

# =============================================================================
# STEP 1A: Download
# =============================================================================
def download_bindingdb():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    if os.path.exists(ZIP_FILE) and os.path.getsize(ZIP_FILE) > 1_000_000:
        print(f"[INFO] ZIP file already exists at {ZIP_FILE}, skipping download.")
        return
    
    print("=" * 70)
    print("STEP 1A: Downloading BindingDB")
    print("=" * 70)
    print(f"URL: {BINDINGDB_URL}")
    print(f"Saving to: {ZIP_FILE}")
    print()
    print("NOTE: This file is ~2GB. If the automated download fails,")
    print("please download manually from:")
    print("  https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp")
    print("Look for 'BindingDB_All_2D_yyyymm_tsv.zip' and save it to:")
    print(f"  {os.path.abspath(ZIP_FILE)}")
    print()
    
    try:
        response = requests.get(BINDINGDB_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(ZIP_FILE, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"\n[OK] Download complete: {os.path.getsize(ZIP_FILE) / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"\n[ERROR] Automated download failed: {e}")
        print()
        print("Please download manually:")
        print("  1. Go to: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp")
        print("  2. Download 'BindingDB_All_2D_yyyymm_tsv.zip'")
        print(f"  3. Save it as: {os.path.abspath(ZIP_FILE)}")
        print("  4. Re-run this script")
        return False
    
    return True


# =============================================================================
# STEP 1B: Unzip
# =============================================================================
def unzip_bindingdb():
    print()
    print("=" * 70)
    print("STEP 1B: Unzipping BindingDB")
    print("=" * 70)
    
    # Find the ZIP file — it might have a different name than expected
    zip_path = ZIP_FILE
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
        # Search for any BindingDB zip in the download dir
        for f in os.listdir(DOWNLOAD_DIR):
            if f.endswith('.zip') and 'BindingDB' in f:
                candidate = os.path.join(DOWNLOAD_DIR, f)
                if os.path.getsize(candidate) > 1_000_000:
                    zip_path = candidate
                    print(f"[INFO] Found ZIP file: {zip_path}")
                    break
    
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
        print(f"[ERROR] ZIP file not found or too small")
        print("Please ensure the download completed successfully.")
        return None
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        tsv_files = [f for f in z.namelist() if f.endswith('.tsv')]
        print(f"Found TSV files in archive: {tsv_files}")
        
        # Extract all
        z.extractall(DOWNLOAD_DIR)
        print(f"[OK] Extracted to {DOWNLOAD_DIR}")
    
    # Find the extracted TSV
    abs_download_dir = os.path.abspath(DOWNLOAD_DIR)
    for f in os.listdir(abs_download_dir):
        if f.endswith('.tsv'):
            tsv_path = os.path.join(abs_download_dir, f)
            size_gb = os.path.getsize(tsv_path) / 1e9
            print(f"[OK] TSV file: {tsv_path} ({size_gb:.2f} GB)")
            return tsv_path
    
    print("[ERROR] No TSV file found after extraction")
    return None


# =============================================================================
# STEP 1C: Load and Explore
# =============================================================================
def explore_bindingdb(tsv_path):
    print()
    print("=" * 70)
    print("STEP 1C: Loading and Exploring BindingDB")
    print("=" * 70)
    
    # Load only the first 100K rows to explore quickly
    print("Loading first 100,000 rows to explore column structure...")
    df_sample = pd.read_csv(tsv_path, sep='\t', nrows=100_000, low_memory=False)
    
    print(f"\nShape (sample): {df_sample.shape}")
    print(f"Columns ({len(df_sample.columns)} total):")
    print("-" * 50)
    
    # Print all columns with their non-null counts
    for i, col in enumerate(df_sample.columns):
        non_null = df_sample[col].notna().sum()
        print(f"  {i:3d}. {col:60s} ({non_null:,} non-null)")
    
    # Identify the key columns we need
    print()
    print("=" * 70)
    print("KEY COLUMNS FOR OUR TASK")
    print("=" * 70)
    
    # These are the standard BindingDB column names — they may vary slightly by release
    key_columns = {
        'Drug SMILES': [
            'Ligand SMILES', 'SMILES', 'BindingDB Ligand Name',
            'Ligand InChI', 'Ligand InChI Key'
        ],
        'Target Info': [
            'Target Name', 'Target Source Organism According to Curator or DataSource',
            'UniProt (SwissProt) Primary ID of Target Chain',
            'UniProt (SwissProt) Secondary ID(s) of Target Chain',
            'UniProt (TrEMBL) Primary ID of Target Chain',
        ],
        'Binding Measurements': [
            'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
        ],
        'IDs': [
            'BindingDB Reactant_set_id', 'Ligand HET ID in PDB',
            'PDB ID(s) of Target Chain', 'PubChem CID', 'ChEMBL ID of Ligand',
        ]
    }
    
    all_cols = set(df_sample.columns)
    
    for category, expected_cols in key_columns.items():
        print(f"\n  {category}:")
        for col_name in expected_cols:
            # Try exact match or partial match
            if col_name in all_cols:
                non_null = df_sample[col_name].notna().sum()
                sample_val = df_sample[col_name].dropna().iloc[0] if non_null > 0 else "N/A"
                # Truncate long values
                sample_str = str(sample_val)[:80]
                print(f"    ✓ {col_name:55s} | {non_null:>6,} non-null | e.g. {sample_str}")
            else:
                # Try partial match
                matches = [c for c in all_cols if col_name.lower() in c.lower()]
                if matches:
                    for m in matches[:2]:
                        non_null = df_sample[m].notna().sum()
                        print(f"    ~ {m:55s} | {non_null:>6,} non-null (partial match)")
                else:
                    print(f"    ✗ {col_name:55s} | NOT FOUND")
    
    # Show organism distribution
    org_col_candidates = [c for c in all_cols if 'organism' in c.lower() or 'source' in c.lower()]
    if org_col_candidates:
        org_col = org_col_candidates[0]
        print()
        print("=" * 70)
        print(f"TOP 15 ORGANISMS (from column: '{org_col}')")
        print("=" * 70)
        org_counts = df_sample[org_col].value_counts().head(15)
        for org, count in org_counts.items():
            pct = count / len(df_sample) * 100
            print(f"  {org:45s} {count:>7,} ({pct:5.1f}%)")
    
    # Show measurement type distribution
    print()
    print("=" * 70)
    print("BINDING MEASUREMENT AVAILABILITY (in sample)")
    print("=" * 70)
    for mtype in ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']:
        if mtype in all_cols:
            non_null = df_sample[mtype].notna().sum()
            pct = non_null / len(df_sample) * 100
            print(f"  {mtype:20s}: {non_null:>7,} values ({pct:5.1f}%)")
    
    # Count total rows in full file (without loading everything)
    print()
    print("=" * 70)
    print("COUNTING TOTAL ROWS IN FULL FILE (this may take a minute)...")
    print("=" * 70)
    # Use absolute path to avoid working-directory issues
    abs_tsv_path = os.path.abspath(tsv_path)
    total_rows = 0
    try:
        with open(abs_tsv_path, 'r', encoding='utf-8', errors='replace') as f:
            for _ in f:
                total_rows += 1
        total_rows -= 1  # subtract header
        print(f"  Total rows in full BindingDB: {total_rows:,}")
    except FileNotFoundError:
        print(f"  [WARNING] Could not count rows — file not found at: {abs_tsv_path}")
        print(f"  This is non-critical. The data loaded fine above.")
    
    # Save column names for reference
    cols_file = os.path.join(DOWNLOAD_DIR, "column_names.txt")
    with open(cols_file, 'w') as f:
        for i, col in enumerate(df_sample.columns):
            f.write(f"{i}\t{col}\n")
    print(f"\n[OK] Column names saved to {cols_file}")
    print(f"[OK] TSV path for next step: {tsv_path}")
    
    return tsv_path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 1: Download and Explore BindingDB                          ║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Download
    success = download_bindingdb()
    
    # Unzip
    tsv_path = unzip_bindingdb()
    
    if tsv_path:
        # Explore
        explore_bindingdb(tsv_path)
        
        print()
        print("╔" + "═" * 68 + "╗")
        print("║   STEP 1 COMPLETE                                                ║")
        print("╠" + "═" * 68 + "╣")
        print("║   Next: Run step2_filter_human_enzymes.py                         ║")
        print("╚" + "═" * 68 + "╝")
    else:
        print()
        print("[!] Could not find TSV file. Please download manually and re-run.")