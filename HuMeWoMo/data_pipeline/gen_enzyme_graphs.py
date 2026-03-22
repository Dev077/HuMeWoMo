import os
import time
import pickle
import requests
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1

# =============================================================================
# CONFIG
# =============================================================================
USE_PYG = True   # Store as PyTorch Geometric Data objects if possible
INPUT_PATH = "data/enzymes.txt"
OUTPUT_PATH = "data/enzymes.pkl"
PDB_DIR = "data/alphafold_structures"
CONTACT_DISTANCE = 8.0  
REQUEST_DELAY = 0.1     

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("[WARNING] torch_geometric not found. Falling back to numpy dicts.")
    USE_PYG = False

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

def get_residue_features(aa_letter):
    encoding = [0] * 21
    if aa_letter in AMINO_ACIDS:
        encoding[AMINO_ACIDS.index(aa_letter)] = 1
    else:
        encoding[20] = 1
    encoding.append(HYDROPHOBICITY.get(aa_letter, 0.5))
    encoding.append(MOLECULAR_WEIGHT.get(aa_letter, 0.5))
    encoding.append(CHARGE.get(aa_letter, 0))
    return encoding

# =============================================================================
# CORE LOGIC
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
        return None
    except Exception:
        return None

def pdb_to_residue_graph(pdb_path, contact_distance=8.0):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception:
        return None
    model = structure[0]
    residues, ca_coords, sequence = [], [], []
    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True) or 'CA' not in residue:
                continue
            ca = residue['CA']
            resname = residue.get_resname()
            aa_letter = protein_letters_3to1.get(resname, 'X')
            residues.append(residue)
            ca_coords.append(ca.get_vector().get_array())
            sequence.append(aa_letter)
    if len(residues) < 5:
        return None
    ca_coords = np.array(ca_coords)
    node_features = np.array([get_residue_features(aa) for aa in sequence], dtype=np.float32)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    contacts = np.where((dist_matrix < contact_distance) & (dist_matrix > 0.01))
    edge_index = np.array([contacts[0], contacts[1]], dtype=np.int64)
    edge_distances = dist_matrix[contacts].astype(np.float32).reshape(-1, 1)
    edge_distances_norm = edge_distances / contact_distance if len(edge_distances) > 0 else edge_distances

    if USE_PYG:
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_distances_norm, dtype=torch.float),
            sequence=''.join(sequence),
            num_residues=len(residues),
        )
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_distances_norm,
        'sequence': ''.join(sequence),
        'num_residues': len(residues),
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    os.makedirs(PDB_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run gen_enzyme_list.py first.")
        exit(1)
        
    with open(INPUT_PATH, 'r') as f:
        uniprot_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Starting batch processing of {len(uniprot_ids)} enzymes...")
    enzyme_graphs = {}
    failed = []

    for uid in tqdm(uniprot_ids, desc="Generating Graphs"):
        pdb_path = download_alphafold_structure(uid, PDB_DIR)
        if pdb_path:
            graph = pdb_to_residue_graph(pdb_path, CONTACT_DISTANCE)
            if graph:
                enzyme_graphs[uid] = graph
            else:
                failed.append((uid, "Parse error"))
        else:
            failed.append((uid, "Download error"))
        time.sleep(REQUEST_DELAY)

    print(f"\nProcessing complete:")
    print(f"  Success: {len(enzyme_graphs)}")
    print(f"  Failed:  {len(failed)}")

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(enzyme_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved graphs to {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")
