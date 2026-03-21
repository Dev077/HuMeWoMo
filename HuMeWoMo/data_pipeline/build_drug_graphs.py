import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# =============================================================================
# CONFIG
# =============================================================================
USE_PYG = True   # Set True to store as PyTorch Geometric Data objects
                  # Set False to store as numpy dicts
                  # If True and torch-geometric is not installed, this will crash intentionally

INPUT_DIR = "./bindingdb_data/processed"
OUTPUT_DIR = "./bindingdb_data/processed/drug_graphs"

if USE_PYG:
    import torch
    from torch_geometric.data import Data


# =============================================================================
# ATOM (NODE) FEATURIZER
# =============================================================================
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'Other']
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding


def get_atom_features(atom):
    features = []
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES[:-1]:
        symbol = 'Other'
    features.extend(one_hot(symbol, ATOM_TYPES))
    features.extend(one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
    features.append(atom.GetFormalCharge())
    features.extend(one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features


def get_atom_feature_dim():
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
    features = []
    features.extend(one_hot(bond.GetBondType(), BOND_TYPES))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    features.extend(one_hot(bond.GetStereo(), BOND_STEREO_TYPES))
    return features


def get_bond_feature_dim():
    return 5 + 1 + 1 + 4  # = 11


# =============================================================================
# SMILES → GRAPH CONVERSION
# =============================================================================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    if len(atom_features) == 0:
        return None

    node_features = np.array(atom_features, dtype=np.float32)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(bf)
        edge_features.append(bf)

    if len(edge_indices) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feat = np.zeros((0, get_bond_feature_dim()), dtype=np.float32)
    else:
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_feat = np.array(edge_features, dtype=np.float32)

    if USE_PYG:
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_feat, dtype=torch.float),
        )
    else:
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

    drugs_path = os.path.join(INPUT_DIR, "unique_drugs.csv")
    if not os.path.exists(drugs_path):
        raise FileNotFoundError(f"Drug list not found at {drugs_path}. Run step2 first.")

    drugs = pd.read_csv(drugs_path)
    print(f"Loaded {len(drugs):,} unique drug SMILES")
    print(f"Format: {'PyTorch Geometric Data' if USE_PYG else 'numpy dicts'}")
    print(f"Atom feature dim: {get_atom_feature_dim()}")
    print(f"Bond feature dim: {get_bond_feature_dim()}")
    print()

    drug_graphs = {}
    failed = []

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
    if USE_PYG:
        atom_counts = [g.x.shape[0] for g in drug_graphs.values()]
        bond_counts = [g.edge_index.shape[1] // 2 for g in drug_graphs.values()]
    else:
        atom_counts = [g['num_atoms'] for g in drug_graphs.values()]
        bond_counts = [g['num_bonds'] for g in drug_graphs.values()]

    print(f"\nGraph statistics:")
    print(f"  Atoms per molecule: mean={np.mean(atom_counts):.1f}, "
          f"median={np.median(atom_counts):.0f}, "
          f"min={np.min(atom_counts)}, max={np.max(atom_counts)}")
    print(f"  Bonds per molecule: mean={np.mean(bond_counts):.1f}, "
          f"median={np.median(bond_counts):.0f}, "
          f"min={np.min(bond_counts)}, max={np.max(bond_counts)}")

    # Save single pickle
    pkl_path = os.path.join(OUTPUT_DIR, "drug_graphs.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(drug_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(pkl_path) / 1e6
    print(f"\n[SAVED] {pkl_path} ({size_mb:.1f} MB)")

    if failed:
        failed_path = os.path.join(OUTPUT_DIR, "failed_smiles.csv")
        pd.DataFrame(failed, columns=['idx', 'smiles', 'reason']).to_csv(failed_path, index=False)
        print(f"[SAVED] Failed SMILES list: {failed_path}")

    return drug_graphs


# =============================================================================
# EXAMPLE
# =============================================================================
def show_example():
    print()
    print("=" * 70)
    print("EXAMPLE: Aspirin molecular graph")
    print("=" * 70)

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    graph = smiles_to_graph(aspirin_smiles)

    print(f"SMILES: {aspirin_smiles}")
    if USE_PYG:
        print(f"Type: PyTorch Geometric Data")
        print(f"Atoms (nodes): {graph.x.shape[0]}")
        print(f"Edges (bidirectional): {graph.edge_index.shape[1]}")
        print(f"Node feature shape: {graph.x.shape}")
        print(f"Edge index shape: {graph.edge_index.shape}")
        print(f"Edge feature shape: {graph.edge_attr.shape}")
        print(f"\nFirst atom features:\n  {graph.x[0]}")
    else:
        print(f"Type: numpy dict")
        print(f"Atoms (nodes): {graph['num_atoms']}")
        print(f"Bonds: {graph['num_bonds']}")
        print(f"Node feature shape: {graph['node_features'].shape}")
        print(f"Edge index shape: {graph['edge_index'].shape}")
        print(f"Edge feature shape: {graph['edge_features'].shape}")
        print(f"\nFirst atom features:\n  {graph['node_features'][0]}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 3: Build Drug Molecular Graphs                             ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  USE_PYG = {USE_PYG}")
    print()

    show_example()
    drug_graphs = build_all_drug_graphs()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   STEP 3 COMPLETE                                                ║")
    print("╠" + "═" * 68 + "╣")
    print("╚" + "═" * 68 + "╝")
    print("║   Next: Run step4_build_enzyme_graphs.py                          ║")