"""
Homo Drug-Enzyme Binding Dataset for PyTorch
==============================================
An alternative dataset where bond edges are promoted to nodes,
creating a single homogeneous graph where both atoms and bonds
are nodes connected to each other.

Standard graph:
    Nodes = atoms (features: 37-dim)
    Edges = bonds (features: 11-dim)

Homo graph:
    Nodes = atoms + bonds (features: max(37,11) + 2 for node type)
    Edges = bond-node <-> atom-node connections (no edge features needed,
            the information is now IN the bond nodes)

This allows the GNN to learn explicit bond representations through
message passing, rather than bonds being passive edge attributes.

The same transformation is applied to enzyme graphs:
    Standard: residues are nodes, contacts are edges
    Homo: residues + contacts are all nodes

Requirements:
    pip install torch torch-geometric

Usage:
    from homo_dataset import HomoDrugEnzymeDataset, get_homo_dataloaders

    train_loader, val_loader, test_loader = get_homo_dataloaders(
        data_dir="./bindingdb_data/final_dataset",
        batch_size=64,
    )
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

# Import shared classes from the standard dataset module
from binding_dataset import DrugEnzymePair, DrugEnzymeCollater, CombinedBatch


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "./bindingdb_data/final_dataset"
TASK = "regression"


# =============================================================================
# GRAPH TRANSFORMATION: promote edges to nodes
# =============================================================================
def edges_to_nodes(x, edge_index, edge_attr):
    """
    Transform a graph by promoting edges to nodes.

    Input (standard graph):
        x:          (num_atoms, atom_feat_dim)     — atom node features
        edge_index: (2, num_edges)                 — directed edges
        edge_attr:  (num_edges, edge_feat_dim)     — bond/contact features

    Output (homo graph):
        homo_x:          (num_atoms + num_unique_edges, unified_feat_dim)
        homo_edge_index: (2, num_connections)

    The unified feature vector for each node is:
        [original_features (padded to max_dim), node_type_atom, node_type_bond]

    For an atom node:
        [atom_feat_0, ..., atom_feat_36, 0, ..., 0, 1, 0]
         |--- atom features (37) ---|  |pad|     |type|

    For a bond node:
        [0, ..., 0, bond_feat_0, ..., bond_feat_10, 0, 1]
         |--pad--|  |--- bond features (11) ---|    |type|

    Edges in the homo graph connect each bond-node to its two atom-nodes
    (bidirectionally), so each original bond creates 4 directed edges:
        bond_node -> source_atom
        bond_node -> target_atom
        source_atom -> bond_node
        target_atom -> bond_node
    """
    num_atoms = x.shape[0]
    atom_feat_dim = x.shape[1]
    num_edges = edge_index.shape[1]

    if edge_attr is not None and edge_attr.shape[0] > 0:
        edge_feat_dim = edge_attr.shape[1]
    else:
        # No edges — return atoms only with type indicator
        homo_x = torch.cat([
            x,
            torch.ones(num_atoms, 1, device=x.device),    # is_atom = 1
            torch.zeros(num_atoms, 1, device=x.device),   # is_bond = 0
        ], dim=1)
        homo_edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
        return homo_x, homo_edge_index

    # --- Deduplicate edges to get unique bonds ---
    # edge_index has bidirectional edges: (i->j) and (j->i) for each bond
    # We only want one bond-node per undirected bond
    # Take edges where source < target to get unique undirected bonds
    src = edge_index[0]
    tgt = edge_index[1]
    unique_mask = src < tgt
    unique_src = src[unique_mask]
    unique_tgt = tgt[unique_mask]
    unique_edge_attr = edge_attr[unique_mask]
    num_bonds = unique_src.shape[0]

    # --- Build unified node features ---
    # Total feature dim = atom_feat_dim + edge_feat_dim + 2 (node type indicators)
    unified_dim = atom_feat_dim + edge_feat_dim + 2

    # Atom nodes: [atom_features, zeros_for_bond_feats, 1, 0]
    atom_nodes = torch.cat([
        x,                                                      # (num_atoms, atom_feat_dim)
        torch.zeros(num_atoms, edge_feat_dim, device=x.device), # padding
        torch.ones(num_atoms, 1, device=x.device),              # is_atom = 1
        torch.zeros(num_atoms, 1, device=x.device),             # is_bond = 0
    ], dim=1)

    # Bond nodes: [zeros_for_atom_feats, bond_features, 0, 1]
    bond_nodes = torch.cat([
        torch.zeros(num_bonds, atom_feat_dim, device=x.device), # padding
        unique_edge_attr,                                        # (num_bonds, edge_feat_dim)
        torch.zeros(num_bonds, 1, device=x.device),             # is_atom = 0
        torch.ones(num_bonds, 1, device=x.device),              # is_bond = 1
    ], dim=1)

    # Concatenate: first num_atoms nodes are atoms, next num_bonds are bonds
    homo_x = torch.cat([atom_nodes, bond_nodes], dim=0)  # (num_atoms + num_bonds, unified_dim)

    # --- Build edges: connect each bond-node to its two atom-nodes ---
    # Bond node indices start at num_atoms
    bond_node_ids = torch.arange(num_bonds, device=x.device) + num_atoms

    # Each bond connects to source atom and target atom (bidirectional)
    homo_src = torch.cat([
        bond_node_ids,   # bond -> source_atom
        bond_node_ids,   # bond -> target_atom
        unique_src,      # source_atom -> bond
        unique_tgt,      # target_atom -> bond
    ])
    homo_tgt = torch.cat([
        unique_src,      # bond -> source_atom
        unique_tgt,      # bond -> target_atom
        bond_node_ids,   # source_atom -> bond
        bond_node_ids,   # target_atom -> bond
    ])

    homo_edge_index = torch.stack([homo_src, homo_tgt], dim=0)

    return homo_x, homo_edge_index


# =============================================================================
# DATASET
# =============================================================================
class HomoDrugEnzymeDataset(Dataset):
    """
    Same as DrugEnzymeDataset but transforms each graph so that
    edges become nodes. Both atom-nodes and bond-nodes live in
    a single homogeneous graph.

    Args:
        pkl_path:    Path to split pickle (train.pkl, val.pkl, test.pkl)
        task:        "regression" or "binary"
        max_samples: Optional cap for debugging
    """
    def __init__(self, pkl_path, task="regression", max_samples=None):
        print(f"Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.samples = pickle.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.task = task
        print(f"  Loaded {len(self.samples):,} samples (task={task}, homo=True)")

    def __len__(self):
        return len(self.samples)

    def _to_pyg(self, graph):
        """Ensure graph is a PyG Data object."""
        if isinstance(graph, Data):
            return graph
        return Data(
            x=torch.tensor(graph['node_features'], dtype=torch.float),
            edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph['edge_features'], dtype=torch.float),
        )

    def __getitem__(self, idx):
        s = self.samples[idx]

        drug = self._to_pyg(s['drug_graph'])
        enzyme = self._to_pyg(s['enzyme_graph'])

        # Transform: promote edges to nodes
        drug_homo_x, drug_homo_edge_index = edges_to_nodes(
            drug.x, drug.edge_index, drug.edge_attr
        )
        enzyme_homo_x, enzyme_homo_edge_index = edges_to_nodes(
            enzyme.x, enzyme.edge_index, enzyme.edge_attr
        )

        # New PyG Data objects with no edge_attr (info is in the nodes now)
        drug_homo = Data(x=drug_homo_x, edge_index=drug_homo_edge_index)
        enzyme_homo = Data(x=enzyme_homo_x, edge_index=enzyme_homo_edge_index)

        # Label
        if self.task == "binary":
            y = float(s.get('label_binary', 0))
        else:
            y = float(s['pActivity'])

        return DrugEnzymePair(drug_homo, enzyme_homo, y)


# =============================================================================
# DATALOADER FACTORY
# =============================================================================
def get_homo_dataloaders(data_dir=DATA_DIR, batch_size=64, task=TASK,
                         num_workers=0, max_samples=None):
    """
    Create train/val/test DataLoaders using the Homo transformation.
    Same interface as get_dataloaders but uses HomoDrugEnzymeDataset.
    """
    collater = DrugEnzymeCollater()

    loaders = {}
    for split in ['train', 'val', 'test']:
        pkl_path = os.path.join(data_dir, f"{split}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing {pkl_path}. Run step5 first.")

        dataset = HomoDrugEnzymeDataset(pkl_path, task=task, max_samples=max_samples)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collater,
            drop_last=(split == 'train'),
            pin_memory=torch.cuda.is_available(),
        )
        loaders[split] = loader

    return loaders['train'], loaders['val'], loaders['test']


# =============================================================================
# TEST / DEMO
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   Homo Drug-Enzyme Binding Dataset — Test                         ║")
    print("╚" + "═" * 68 + "╝")

    # ---- Show transformation on a single example ----
    print("\n" + "=" * 70)
    print("TRANSFORMATION EXAMPLE:")
    print("=" * 70)

    # Fake small molecule: 4 atoms, 3 bonds
    x = torch.randn(4, 37)
    edge_index = torch.tensor([[0,1, 1,2, 2,3],
                                [1,0, 2,1, 3,2]], dtype=torch.long)
    edge_attr = torch.randn(6, 11)

    print(f"\nBefore (standard graph):")
    print(f"  Nodes (atoms):  {x.shape[0]}")
    print(f"  Edges (bonds):  {edge_index.shape[1]} directed = {edge_index.shape[1]//2} undirected")
    print(f"  Node feat dim:  {x.shape[1]}")
    print(f"  Edge feat dim:  {edge_attr.shape[1]}")

    homo_x, homo_edge_index = edges_to_nodes(x, edge_index, edge_attr)

    n_atom_nodes = x.shape[0]
    n_bond_nodes = homo_x.shape[0] - n_atom_nodes

    print(f"\nAfter (homo graph):")
    print(f"  Total nodes:    {homo_x.shape[0]} ({n_atom_nodes} atom-nodes + {n_bond_nodes} bond-nodes)")
    print(f"  Total edges:    {homo_edge_index.shape[1]} directed")
    print(f"  Node feat dim:  {homo_x.shape[1]} (= {x.shape[1]} + {edge_attr.shape[1]} + 2 type bits)")
    print(f"  Edge features:  None (information is in bond nodes now)")

    print(f"\n  Atom node [0] type bits: ...{homo_x[0, -2:]}")
    print(f"  Bond node [{n_atom_nodes}] type bits: ...{homo_x[n_atom_nodes, -2:]}")

    # ---- Test with real data ----
    print("\n" + "=" * 70)
    print("REAL DATA TEST:")
    print("=" * 70)

    train_loader, val_loader, test_loader = get_homo_dataloaders(
        data_dir=DATA_DIR,
        batch_size=4,
        task=TASK,
        max_samples=100,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    batch = next(iter(train_loader))

    print(f"\n  Batch size:              {batch.batch_size}")
    print(f"\n  Drug graphs (homo):")
    print(f"    Node features:         {batch.drug_x.shape}")
    print(f"    Edge index:            {batch.drug_edge_index.shape}")
    print(f"    Batch vector:          {batch.drug_batch.shape}")
    print(f"    Unique graphs:         {batch.drug_batch.unique().shape[0]}")
    print(f"    Has edge_attr:         {batch.drug_edge_attr is not None}")

    print(f"\n  Enzyme graphs (homo):")
    print(f"    Node features:         {batch.enzyme_x.shape}")
    print(f"    Edge index:            {batch.enzyme_edge_index.shape}")
    print(f"    Batch vector:          {batch.enzyme_batch.shape}")
    print(f"    Unique graphs:         {batch.enzyme_batch.unique().shape[0]}")

    print(f"\n  Labels (y):              {batch.y}")

    # Verify node type indicators
    drug_type_bits = batch.drug_x[:, -2:]
    is_atom = (drug_type_bits[:, 0] == 1).sum().item()
    is_bond = (drug_type_bits[:, 1] == 1).sum().item()
    print(f"\n  Drug node breakdown:")
    print(f"    Atom nodes:            {is_atom}")
    print(f"    Bond nodes:            {is_bond}")
    print(f"    Total:                 {is_atom + is_bond}")

    enzyme_type_bits = batch.enzyme_x[:, -2:]
    is_residue = (enzyme_type_bits[:, 0] == 1).sum().item()
    is_contact = (enzyme_type_bits[:, 1] == 1).sum().item()
    print(f"\n  Enzyme node breakdown:")
    print(f"    Residue nodes:         {is_residue}")
    print(f"    Contact nodes:         {is_contact}")
    print(f"    Total:                 {is_residue + is_contact}")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   Homo dataset test passed!                                       ║")
    print("╚" + "═" * 68 + "╝")