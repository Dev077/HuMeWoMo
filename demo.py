import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, Batch

# Path setup
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from HuMeWoMo.models.homo_binding_model import HomoBindingModel
from HuMeWoMo.datasets.homo_binding_dataset import edges_to_nodes

# =============================================================================
# DRUG FEATURIZATION
# =============================================================================
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'Other']
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
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
    if symbol not in ATOM_TYPES[:-1]: symbol = 'Other'
    features.extend(one_hot(symbol, ATOM_TYPES))
    features.extend(one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
    features.append(atom.GetFormalCharge())
    features.extend(one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
    features.extend(one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    return features

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

def get_bond_features(bond):
    features = []
    features.extend(one_hot(bond.GetBondType(), BOND_TYPES))
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    features.extend(one_hot(bond.GetStereo(), BOND_STEREO_TYPES))
    return features

def smiles_to_homo_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bf, bf])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    homo_x, homo_edge_index = edges_to_nodes(x, edge_index, edge_attr, edge_feat_dim=11)
    return Data(x=homo_x, edge_index=homo_edge_index)

# =============================================================================
# INFERENCE LOGIC (NEW 8/4 MODEL)
# =============================================================================
def run_demo():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE} (Model: 8/4 Architecture)")

    # 1. Load Model
    model = HomoBindingModel(
        drug_in_dim=50, enzyme_in_dim=27, hidden_dim=128,
        num_drug_layers=8, num_enzyme_layers=8, num_decoder_layers=4
    ).to(DEVICE)
    
    model_path = "best_homo_model_8_4.pt" # New path for new model
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    else:
        print("[WARNING] Model weights not found. Running with random initialization.")
    
    model.eval()

    # 2. Pre-computation
    ENZYME_PKL = "data/human1_enzymes.pkl"
    ENZYME_EMB_PATH = "data/human1_embeddings.pkl" # Separate file for 8/4 embs
    
    if not os.path.exists(ENZYME_EMB_PATH):
        if not os.path.exists(ENZYME_PKL):
            raise FileNotFoundError(f"{ENZYME_PKL} not found.")
        print(f"Pre-computing 8/4 enzyme embeddings...")
        with open(ENZYME_PKL, 'rb') as f:
            enzyme_graphs = pickle.load(f)
        enzyme_embeddings = {}
        with torch.no_grad():
            for uid, graph in tqdm(enzyme_graphs.items(), desc="Encoding"):
                if not isinstance(graph, Data):
                    graph = Data(x=torch.tensor(graph['node_features'], dtype=torch.float),
                                 edge_index=torch.tensor(graph['edge_index'], dtype=torch.long))
                graph = graph.to(DEVICE)
                k, v = model.enzyme_encoder(graph.x, graph.edge_index)
                enzyme_embeddings[uid] = {'k': k.cpu(), 'v': v.cpu(), 'num_nodes': k.shape[0]}
        with open(ENZYME_EMB_PATH, 'wb') as f:
            pickle.dump(enzyme_embeddings, f)
    else:
        print(f"Loading pre-computed enzyme embeddings from {ENZYME_EMB_PATH}...")
        with open(ENZYME_EMB_PATH, 'rb') as f:
            enzyme_embeddings = pickle.load(f)

    # 3. Predict
    SMILES = "CC(=O)Oc1ccccc1C(=O)O"
    drug_graph = smiles_to_homo_graph(SMILES).to(DEVICE)
    with torch.no_grad():
        d_k, d_v = model.drug_encoder(drug_graph.x, drug_graph.edge_index)
    
    results = {}
    uids = list(enzyme_embeddings.keys())
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(uids), batch_size), desc="Predicting"):
            batch_uids = uids[i:i+batch_size]
            e_k_list, e_v_list, e_batch_list = [], [], []
            for j, uid in enumerate(batch_uids):
                emb = enzyme_embeddings[uid]
                e_k_list.append(emb['k']); e_v_list.append(emb['v'])
                e_batch_list.append(torch.full((emb['num_nodes'],), j, dtype=torch.long))
            e_k_batch, e_v_batch = torch.cat(e_k_list).to(DEVICE), torch.cat(e_v_list).to(DEVICE)
            e_batch_vec = torch.cat(e_batch_list).to(DEVICE)
            
            curr_bs = len(batch_uids)
            d_k_batch, d_v_batch = d_k.repeat(curr_bs, 1), d_v.repeat(curr_bs, 1)
            d_batch_vec = torch.arange(curr_bs, device=DEVICE).repeat_interleave(d_k.shape[0])
            
            preds = model.decoder(d_k_batch, d_v_batch, e_k_batch, e_v_batch, d_batch_vec, e_batch_vec)
            for j, uid in enumerate(batch_uids):
                results[uid] = float(preds[j])

    top_targets = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop targets for {SMILES}:")
    for uid, score in top_targets: print(f"  {uid}: {score:.4f}")

if __name__ == "__main__":
    run_demo()
