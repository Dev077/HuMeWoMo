import os
import time
import pickle
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

# =============================================================================
# CONFIG
# =============================================================================
USE_PYG = True   # Set True to store as PyTorch Geometric Data objects
                  # Set False to store as numpy dicts
                  # If True and torch-geometric is not installed, this will crash intentionally

INPUT_DIR = "./bindingdb_data/processed"
OUTPUT_DIR = "./bindingdb_data/processed/enzyme_graphs"
PDB_DIR = "./bindingdb_data/alphafold_structures"
CONTACT_DISTANCE = 8.0  # Angstroms
REQUEST_DELAY = 0.1     # seconds between AlphaFold API requests

if USE_PYG:
    import torch
    from torch_geometric.data import Data


# =============================================================================
# AMINO ACID FEATURES
# =============================================================================
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBICITY = {
    'A': 0.62, 'C': 0.29, 'D': 0.00, 'E': 0.00, 'F': 0.78,
    'G': 0.49, 'H': 0.14, 'I': 1.00, 'K': 0.00, 'L': 0.92,
    'M': 0.64, 'N': 0.00, 'P': 0.31, 'Q': 0.00, 'R': 0.00,
    'S': 0.18, 'T': 0.24, 'V': 0.86, 'W': 0.41, 'Y': 0.36,
}
MOLECULAR_WEIGHT = {
    'A': 0.44, 'C': 0.59, 'D': 0.65, 'E': 0.72, 'F': 0.81,
    'G': 0.37, 'H': 0.76, 'I': 0.64, 'K': 0.72, 'L': 0.64,
    'M': 0.73, 'N': 0.65, 'P': 0.56, 'Q': 0.72, 'R': 0.85,
    'S': 0.52, 'T': 0.58, 'V': 0.57, 'W': 1.00, 'Y': 0.89,
}
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}


def one_hot_aa(aa):
    encoding = [0] * 21
    if aa in AMINO_ACIDS:
        encoding[AMINO_ACIDS.index(aa)] = 1
    else:
        encoding[20] = 1
    return encoding


def get_residue_features(aa_letter):
    features = one_hot_aa(aa_letter)
    features.append(HYDROPHOBICITY.get(aa_letter, 0.5))
    features.append(MOLECULAR_WEIGHT.get(aa_letter, 0.5))
    features.append(CHARGE.get(aa_letter, 0))
    return features


def get_residue_feature_dim():
    return 24


# =============================================================================
# ALPHAFOLD STRUCTURE DOWNLOAD
# =============================================================================
def download_alphafold_structure(uniprot_id, output_dir):
    pdb_path = os.path.join(output_dir, f"AF-{uniprot_id}-F1-model_v6.pdb")

    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 100:
        return pdb_path

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(pdb_path, 'w') as f:
                f.write(response.text)
            return pdb_path
        else:
            return None
    except Exception:
        return None


# =============================================================================
# PDB → RESIDUE CONTACT GRAPH
# =============================================================================
def pdb_to_residue_graph(pdb_path, contact_distance=8.0):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception:
        return None

    model = structure[0]

    residues = []
    ca_coords = []
    sequence = []

    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            if 'CA' not in residue:
                continue

            ca = residue['CA']
            resname = residue.get_resname()
            try:
                aa_letter = protein_letters_3to1.get(resname, 'X')
            except:
                aa_letter = 'X'

            residues.append(residue)
            ca_coords.append(ca.get_vector().get_array())
            sequence.append(aa_letter)

    if len(residues) < 5:
        return None

    ca_coords = np.array(ca_coords)
    num_residues = len(residues)

    node_features = np.array(
        [get_residue_features(aa) for aa in sequence],
        dtype=np.float32
    )

    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    contacts = np.where(
        (dist_matrix < contact_distance) & (dist_matrix > 0.01)
    )

    edge_index = np.array([contacts[0], contacts[1]], dtype=np.int64)
    edge_distances = dist_matrix[contacts].astype(np.float32).reshape(-1, 1)

    if len(edge_distances) > 0:
        edge_distances_norm = edge_distances / contact_distance
    else:
        edge_distances_norm = edge_distances

    if USE_PYG:
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_distances_norm, dtype=torch.float),
            sequence=''.join(sequence),
            num_residues=num_residues,
        )
    else:
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_distances_norm,
            'sequence': ''.join(sequence),
            'num_residues': num_residues,
            'num_contacts': edge_index.shape[1],
        }


# =============================================================================
# MAIN PROCESSING
# =============================================================================
def build_all_enzyme_graphs():
    print("=" * 70)
    print("STEP 4: Building Enzyme Residue Contact Graphs")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDB_DIR, exist_ok=True)

    targets_path = os.path.join(INPUT_DIR, "unique_targets.csv")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Target list not found at {targets_path}. Run step2 first.")

    targets = pd.read_csv(targets_path)
    print(f"Loaded {len(targets):,} unique enzyme targets")
    print(f"Format: {'PyTorch Geometric Data' if USE_PYG else 'numpy dicts'}")
    print(f"Contact distance cutoff: {CONTACT_DISTANCE} Å")
    print(f"Residue feature dim: {get_residue_feature_dim()}")
    print()

    enzyme_graphs = {}
    download_failed = []
    parse_failed = []

    for idx, row in tqdm(targets.iterrows(), total=len(targets), desc="Processing enzymes"):
        uniprot_id = str(row['uniprot_id']).strip()

        if not uniprot_id or uniprot_id == 'nan' or len(uniprot_id) < 4:
            download_failed.append((uniprot_id, "invalid ID"))
            continue

        pdb_path = download_alphafold_structure(uniprot_id, PDB_DIR)
        if pdb_path is None:
            download_failed.append((uniprot_id, "download failed"))
            continue

        graph = pdb_to_residue_graph(pdb_path, CONTACT_DISTANCE)
        if graph is None:
            parse_failed.append((uniprot_id, "parse failed"))
            continue

        enzyme_graphs[uniprot_id] = graph
        time.sleep(REQUEST_DELAY)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Successfully processed: {len(enzyme_graphs):,} / {len(targets):,}")
    print(f"  Download failures:      {len(download_failed):,}")
    print(f"  Parse failures:         {len(parse_failed):,}")

    if enzyme_graphs:
        if USE_PYG:
            residue_counts = [g.num_residues for g in enzyme_graphs.values()]
            contact_counts = [g.edge_index.shape[1] for g in enzyme_graphs.values()]
        else:
            residue_counts = [g['num_residues'] for g in enzyme_graphs.values()]
            contact_counts = [g['num_contacts'] for g in enzyme_graphs.values()]

        print(f"\nGraph statistics:")
        print(f"  Residues per enzyme: mean={np.mean(residue_counts):.0f}, "
              f"median={np.median(residue_counts):.0f}, "
              f"min={np.min(residue_counts)}, max={np.max(residue_counts)}")
        print(f"  Contacts per enzyme: mean={np.mean(contact_counts):.0f}, "
              f"median={np.median(contact_counts):.0f}")

    # Save single pickle
    pkl_path = os.path.join(OUTPUT_DIR, "enzyme_graphs.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(enzyme_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(pkl_path) / 1e6
    print(f"\n[SAVED] {pkl_path} ({size_mb:.1f} MB)")

    if download_failed or parse_failed:
        failures = (
            [(uid, reason, 'download') for uid, reason in download_failed] +
            [(uid, reason, 'parse') for uid, reason in parse_failed]
        )
        fail_df = pd.DataFrame(failures, columns=['uniprot_id', 'reason', 'stage'])
        fail_path = os.path.join(OUTPUT_DIR, "failed_targets.csv")
        fail_df.to_csv(fail_path, index=False)
        print(f"[SAVED] Failed targets: {fail_path}")

    return enzyme_graphs


# =============================================================================
# EXAMPLE
# =============================================================================
def show_example():
    print()
    print("=" * 70)
    print("EXAMPLE: Building a contact graph for human CDK2 (P24941)")
    print("=" * 70)

    os.makedirs(PDB_DIR, exist_ok=True)
    uniprot_id = "P24941"

    print(f"Downloading AlphaFold structure for {uniprot_id}...")
    pdb_path = download_alphafold_structure(uniprot_id, PDB_DIR)

    if pdb_path is None:
        print("[ERROR] Could not download. Check your internet connection.")
        return

    print(f"PDB file: {pdb_path}")
    graph = pdb_to_residue_graph(pdb_path, CONTACT_DISTANCE)

    if graph is None:
        print("[ERROR] Failed to build graph")
        return

    if USE_PYG:
        print(f"\nCDK2 Contact Graph (PyG Data):")
        print(f"  Residues (nodes):   {graph.num_residues}")
        print(f"  Contacts (edges):   {graph.edge_index.shape[1]}")
        print(f"  Node feature shape: {graph.x.shape}")
        print(f"  Edge index shape:   {graph.edge_index.shape}")
        print(f"  Edge feature shape: {graph.edge_attr.shape}")
        print(f"  Sequence (first 50): {graph.sequence[:50]}...")
    else:
        print(f"\nCDK2 Contact Graph (numpy dict):")
        print(f"  Residues (nodes):   {graph['num_residues']}")
        print(f"  Contacts (edges):   {graph['num_contacts']}")
        print(f"  Node feature shape: {graph['node_features'].shape}")
        print(f"  Edge index shape:   {graph['edge_index'].shape}")
        print(f"  Edge feature shape: {graph['edge_features'].shape}")
        print(f"  Sequence (first 50): {graph['sequence'][:50]}...")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 4: Build Enzyme Residue Contact Graphs                     ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  USE_PYG = {USE_PYG}")
    print()

    show_example()
    enzyme_graphs = build_all_enzyme_graphs()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 4 COMPLETE                                                ║")
    print("╠" + "═" * 68 + "╣")
    print("║   Next: Run step5_build_dataset.py                                ║")
    print("╚" + "═" * 68 + "╝")