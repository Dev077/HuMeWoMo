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
# DRUG FEATURIZATION (Copied from build_drug_graphs.py)
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
    
    # Transform to homo graph
    homo_x, homo_edge_index = edges_to_nodes(x, edge_index, edge_attr, edge_feat_dim=11)
    return Data(x=homo_x, edge_index=homo_edge_index)

if __name__ == "__main__":
    # =============================================================================
    # INFERENCE LOGIC
    # =============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Load Model
    # Architecture must match the saved weights (3 layers, 3 layers, 3 layers)
    model = HomoBindingModel(
        drug_in_dim=50, enzyme_in_dim=27, hidden_dim=128,
        num_drug_layers=3, num_enzyme_layers=3, num_decoder_layers=3
    ).to(DEVICE)
    
    model_path = "best_homo_model.pt"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Check if it's a checkpoint dict or raw state_dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model.eval()

    # 2. Handle Enzyme Pre-computation
    ENZYME_PKL = "data/human1_enzymes.pkl"
    ENZYME_EMB_PATH = "data/human1_embeddings.pkl"
    
    if not os.path.exists(ENZYME_EMB_PATH):
        print(f"Pre-computing enzyme embeddings from {ENZYME_PKL}...")
        if not os.path.exists(ENZYME_PKL):
            raise FileNotFoundError(f"Error: {ENZYME_PKL} not found. Run gen_enzyme_graphs.py first.")

        with open(ENZYME_PKL, 'rb') as f:
            enzyme_graphs = pickle.load(f)
            
        enzyme_embeddings = {}
        with torch.no_grad():
            for uid, graph in tqdm(enzyme_graphs.items(), desc="Encoding Enzymes"):
                if not isinstance(graph, Data):
                    x = torch.tensor(graph['node_features'], dtype=torch.float)
                    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                    edge_attr = torch.tensor(graph.get('edge_features', []), dtype=torch.float)
                else:
                    x = graph.x
                    edge_index = graph.edge_index
                    edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None

                # Transform to homo graph (residues + contacts as nodes)
                # Enzyme edge_feat_dim is 1 (normalized distance)
                homo_x, homo_edge_index = edges_to_nodes(x, edge_index, edge_attr, edge_feat_dim=1)
                
                homo_x = homo_x.to(DEVICE)
                homo_edge_index = homo_edge_index.to(DEVICE)
                
                k, v = model.enzyme_encoder(homo_x, homo_edge_index)
                enzyme_embeddings[uid] = {
                    'k': k.cpu(), 'v': v.cpu(), 
                    'num_nodes': k.shape[0]
                }
        
        with open(ENZYME_EMB_PATH, 'wb') as f:
            pickle.dump(enzyme_embeddings, f)
        print(f"Saved {len(enzyme_embeddings)} embeddings to {ENZYME_EMB_PATH}")
    else:
        print(f"Loading pre-computed enzyme embeddings from {ENZYME_EMB_PATH}...")
        with open(ENZYME_EMB_PATH, 'rb') as f:
            enzyme_embeddings = pickle.load(f)

    # 3. Encode Drug
    SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    print(f"Encoding drug: {SMILES}")
    drug_graph = smiles_to_homo_graph(SMILES).to(DEVICE)
    with torch.no_grad():
        d_k, d_v = model.drug_encoder(drug_graph.x, drug_graph.edge_index)
    
    # 4. Batch Prediction
    print("Running batch affinity prediction...")
    results = {}
    uids = list(enzyme_embeddings.keys())
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(uids), batch_size), desc="Predicting"):
            batch_uids = uids[i:i+batch_size]
            
            # Prepare enzyme batch
            e_k_list, e_v_list, e_batch_list = [], [], []
            for j, uid in enumerate(batch_uids):
                emb = enzyme_embeddings[uid]
                e_k_list.append(emb['k'])
                e_v_list.append(emb['v'])
                e_batch_list.append(torch.full((emb['num_nodes'],), j, dtype=torch.long))
            
            e_k_batch = torch.cat(e_k_list).to(DEVICE)
            e_v_batch = torch.cat(e_v_list).to(DEVICE)
            e_batch_vec = torch.cat(e_batch_list).to(DEVICE)
            
            # Expand drug to match batch size
            # drug_k/v are [num_drug_nodes, dim]
            # decoder expects [total_drug_nodes_in_batch, dim]
            curr_batch_size = len(batch_uids)
            d_k_batch = d_k.repeat(curr_batch_size, 1)
            d_v_batch = d_v.repeat(curr_batch_size, 1)
            # drug_batch vector: [0,0,0, 1,1,1, ...]
            num_d_nodes = d_k.shape[0]
            d_batch_vec = torch.arange(curr_batch_size, device=DEVICE).repeat_interleave(num_d_nodes)
            
            preds = model.decoder(
                d_k_batch, d_v_batch,
                e_k_batch, e_v_batch,
                d_batch_vec, e_batch_vec
            )
            
            for j, uid in enumerate(batch_uids):
                results[uid] = float(preds[j])

    # 5. Output
    print(f"\nPredictions for {SMILES}:")
    # Show top 5
    top_targets = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
    for uid, score in top_targets:
        print(f"  {uid}: {score:.4f}")
    
    # Save results
    with open("data/affinities.pkl", 'wb') as f:
        pickle.dump(results, f)
    print("\nFull affinities saved to data/affinities.pkl")

