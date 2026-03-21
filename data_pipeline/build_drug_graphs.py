import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    print("[OK] RDKit loaded")
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")

try:
    import torch
    from torch_geometric.data import Data
    print("[OK] PyTorch Geometric loaded")
    USE_PYG = True
except ImportError:
    print("[INFO] PyTorch Geometric not found. Will save as numpy dicts instead.")
    print("       Install with: pip install torch torch-geometric")
    USE_PYG = False


# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR = "./bindingdb_data/processed"
OUTPUT_DIR = "./bindingdb_data/processed/drug_graphs"


# =============================================================================
# ATOM (NODE) FEATURIZER
# =============================================================================
# Standard atom features used in most molecular GNN papers
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'Other']
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

def one_hot(value, choices):
    """One-hot encode a value given a list of choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for 'other'
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # 'other' category
    return encoding


def get_atom_features(atom):
    """
    Extract features for a single atom.
    Returns a feature vector of dimension 33.
    """
    features = []
    
    # Atom type (one-hot, 14 dim)
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES[:-1]:
        symbol = 'Other'
    features.extend(one_hot(symbol, ATOM_TYPES))
    
    # Degree (one-hot, 0-6, 7 dim)
    features.extend(one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
    
    # Formal charge (1 dim)
    features.append(atom.GetFormalCharge())
    
    # Number of Hs (one-hot, 0-4, 5 dim)
    features.extend(one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
    
    # Hybridization (one-hot, 6 dim)
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
    
    # Is aromatic (1 dim)
    features.append(1 if atom.GetIsAromatic() else 0)
    
    # Is in ring (1 dim)
    features.append(1 if atom.IsInRing() else 0)
    
    return features


def get_atom_feature_dim():
    """Return the dimension of atom feature vectors."""
    # 14 (atom type) + 7 (degree) + 1 (charge) + 5 (Hs) + 6 (hybrid) + 1 (aromatic) + 1 (ring) = 35
    return 14 + 7 + 1 + 5 + 6 + 1 + 1  # = 35


# =============================================================================
# BOND (EDGE) FEATURIZER
# =============================================================================
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

def get_bond_features(bond):
    """
    Extract features for a single bond.
    Returns a feature vector of dimension 10.
    """
    features = []
    
    # Bond type (one-hot, 5 dim)
    features.extend(one_hot(bond.GetBondType(), BOND_TYPES))
    
    # Is conjugated (1 dim)
    features.append(1 if bond.GetIsConjugated() else 0)
    
    # Is in ring (1 dim)
    features.append(1 if bond.IsInRing() else 0)
    
    # Bond stereo (one-hot, 4 dim)
    features.extend(one_hot(bond.GetStereo(), BOND_STEREO_TYPES))
    
    return features


def get_bond_feature_dim():
    """Return the dimension of bond feature vectors."""
    return 5 + 1 + 1 + 4  # = 11


# =============================================================================
# SMILES → GRAPH CONVERSION
# =============================================================================
def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a molecular graph.
    
    Returns:
        dict with keys:
            'node_features': np.array of shape (num_atoms, atom_feature_dim)
            'edge_index':    np.array of shape (2, num_bonds*2) — bidirectional
            'edge_features': np.array of shape (num_bonds*2, bond_feature_dim)
            'num_atoms':     int
            'num_bonds':     int
        
        Or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens info (but don't add explicit H atoms)
    # mol = Chem.AddHs(mol)  # uncomment if you want explicit hydrogens as nodes
    
    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    if len(atom_features) == 0:
        return None
    
    node_features = np.array(atom_features, dtype=np.float32)
    
    # Edge index and features (bidirectional)
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        
        # Add both directions (undirected graph)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(bf)
        edge_features.append(bf)
    
    if len(edge_indices) == 0:
        # Single atom molecule — no edges
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feat = np.zeros((0, get_bond_feature_dim()), dtype=np.float32)
    else:
        edge_index = np.array(edge_indices, dtype=np.int64).T  # shape: (2, num_edges)
        edge_feat = np.array(edge_features, dtype=np.float32)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_feat,
        'num_atoms': node_features.shape[0],
        'num_bonds': len(mol.GetBonds()),
    }


# =============================================================================
# MAIN PROCESSING
# =============================================================================
def build_all_drug_graphs():
    print("=" * 70)
    print("STEP 3: Building Drug Molecular Graphs")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load unique drugs
    drugs_path = os.path.join(INPUT_DIR, "unique_drugs.csv")
    if not os.path.exists(drugs_path):
        raise FileNotFoundError(f"Drug list not found at {drugs_path}. Run step2 first.")
    
    drugs = pd.read_csv(drugs_path)
    print(f"Loaded {len(drugs):,} unique drug SMILES")
    
    # Process each SMILES
    drug_graphs = {}  # smiles → graph dict
    failed = []
    
    print(f"\nConverting SMILES to molecular graphs...")
    print(f"  Atom feature dim: {get_atom_feature_dim()}")
    print(f"  Bond feature dim: {get_bond_feature_dim()}")
    print()
    
    for idx, row in tqdm(drugs.iterrows(), total=len(drugs), desc="Building drug graphs"):
        smiles = row['smiles']
        
        if not isinstance(smiles, str) or len(smiles) == 0:
            failed.append((idx, smiles, "empty"))
            continue
        
        graph = smiles_to_graph(smiles)
        
        if graph is None:
            failed.append((idx, smiles, "invalid"))
            continue
        
        drug_graphs[smiles] = graph
    
    print(f"\n[OK] Successfully converted: {len(drug_graphs):,} / {len(drugs):,}")
    print(f"[!!] Failed: {len(failed):,}")
    
    if failed:
        print(f"\nSample failures:")
        for idx, smi, reason in failed[:5]:
            smi_short = str(smi)[:60] if smi else "None"
            print(f"  Row {idx}: {smi_short}... ({reason})")
    
    # Stats
    atom_counts = [g['num_atoms'] for g in drug_graphs.values()]
    bond_counts = [g['num_bonds'] for g in drug_graphs.values()]
    print(f"\nGraph statistics:")
    print(f"  Atoms per molecule: mean={np.mean(atom_counts):.1f}, "
          f"median={np.median(atom_counts):.0f}, "
          f"min={np.min(atom_counts)}, max={np.max(atom_counts)}")
    print(f"  Bonds per molecule: mean={np.mean(bond_counts):.1f}, "
          f"median={np.median(bond_counts):.0f}, "
          f"min={np.min(bond_counts)}, max={np.max(bond_counts)}")
    
    # Save as pickle (universal format)
    pkl_path = os.path.join(OUTPUT_DIR, "drug_graphs.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(drug_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(pkl_path) / 1e6
    print(f"\n[SAVED] {pkl_path} ({size_mb:.1f} MB)")
    
    # Also save as PyTorch Geometric Data objects if available
    if USE_PYG:
        print("\nConverting to PyTorch Geometric format...")
        pyg_graphs = {}
        for smiles, g in tqdm(drug_graphs.items(), desc="Converting to PyG"):
            data = Data(
                x=torch.tensor(g['node_features'], dtype=torch.float),
                edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(g['edge_features'], dtype=torch.float),
            )
            pyg_graphs[smiles] = data
        
        pyg_path = os.path.join(OUTPUT_DIR, "drug_graphs_pyg.pkl")
        with open(pyg_path, 'wb') as f:
            pickle.dump(pyg_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(pyg_path) / 1e6
        print(f"[SAVED] {pyg_path} ({size_mb:.1f} MB)")
    
    # Save the failed SMILES for reference
    if failed:
        failed_path = os.path.join(OUTPUT_DIR, "failed_smiles.csv")
        pd.DataFrame(failed, columns=['idx', 'smiles', 'reason']).to_csv(
            failed_path, index=False
        )
        print(f"[SAVED] Failed SMILES list: {failed_path}")
    
    return drug_graphs


# =============================================================================
# EXAMPLE: Show what a single drug graph looks like
# =============================================================================
def show_example():
    print()
    print("=" * 70)
    print("EXAMPLE: Aspirin molecular graph")
    print("=" * 70)
    
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    graph = smiles_to_graph(aspirin_smiles)
    
    print(f"SMILES: {aspirin_smiles}")
    print(f"Atoms (nodes): {graph['num_atoms']}")
    print(f"Bonds: {graph['num_bonds']}")
    print(f"Edges (bidirectional): {graph['edge_index'].shape[1]}")
    print(f"Node feature matrix shape: {graph['node_features'].shape}")
    print(f"Edge index shape: {graph['edge_index'].shape}")
    print(f"Edge feature matrix shape: {graph['edge_features'].shape}")
    
    print(f"\nFirst atom features (atom 0 = Carbon):")
    print(f"  {graph['node_features'][0]}")
    
    print(f"\nEdge index (first 6 edges):")
    print(f"  Source: {graph['edge_index'][0, :6]}")
    print(f"  Target: {graph['edge_index'][1, :6]}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 3: Build Drug Molecular Graphs                             ║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Show example first
    show_example()
    
    # Build all graphs
    drug_graphs = build_all_drug_graphs()
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 3 COMPLETE                                                ║")
    print("╠" + "═" * 68 + "╣")
    print("║   Output: ./bindingdb_data/processed/drug_graphs/                 ║")
    print("║   Next: Run step4_build_enzyme_graphs.py                          ║")
    print("╚" + "═" * 68 + "╝")