import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "./bindingdb_data/final_dataset"
TASK = "regression"  # "regression" (predict pActivity) or "binary" (bind/no-bind)


# =============================================================================
# CUSTOM DATA OBJECT: holds both graphs + label in one object
# =============================================================================
class DrugEnzymePair:
    """
    A single (drug, enzyme, label) sample.
    Stores two PyG Data objects and a scalar label.
    """
    def __init__(self, drug_graph, enzyme_graph, y):
        self.drug = drug_graph      # PyG Data: x, edge_index, edge_attr
        self.enzyme = enzyme_graph  # PyG Data: x, edge_index, edge_attr
        self.y = y                  # float (pActivity) or int (0/1)


# =============================================================================
# CUSTOM COLLATER: batches drug and enzyme graphs separately
# =============================================================================
class DrugEnzymeCollater:
    """
    Collates a list of DrugEnzymePair into a batch.

    The key challenge: PyG's Batch.from_data_list only handles one graph per sample.
    We have TWO graphs per sample (drug + enzyme), so we batch them separately
    and combine into a single namespace.

    Output batch has:
        batch.drug_x            — (total_drug_atoms, drug_feat_dim)
        batch.drug_edge_index   — (2, total_drug_edges)
        batch.drug_edge_attr    — (total_drug_edges, drug_edge_feat_dim)
        batch.drug_batch        — (total_drug_atoms,) mapping atoms to samples

        batch.enzyme_x          — (total_enzyme_residues, enzyme_feat_dim)
        batch.enzyme_edge_index — (2, total_enzyme_contacts)
        batch.enzyme_edge_attr  — (total_enzyme_contacts, 1)
        batch.enzyme_batch      — (total_enzyme_residues,) mapping residues to samples

        batch.y                 — (batch_size,) labels
    """
    def __call__(self, samples):
        drug_graphs = [s.drug for s in samples]
        enzyme_graphs = [s.enzyme for s in samples]
        labels = torch.tensor([s.y for s in samples], dtype=torch.float)

        drug_batch = Batch.from_data_list(drug_graphs)
        enzyme_batch = Batch.from_data_list(enzyme_graphs)

        return CombinedBatch(drug_batch, enzyme_batch, labels)


class CombinedBatch:
    """
    Holds batched drug graphs, batched enzyme graphs, and labels
    in a single object with a clean attribute interface.
    """
    def __init__(self, drug_batch, enzyme_batch, y):
        # Drug graph batch
        self.drug_x = drug_batch.x
        self.drug_edge_index = drug_batch.edge_index
        self.drug_edge_attr = drug_batch.edge_attr
        self.drug_batch = drug_batch.batch

        # Enzyme graph batch
        self.enzyme_x = enzyme_batch.x
        self.enzyme_edge_index = enzyme_batch.edge_index
        self.enzyme_edge_attr = enzyme_batch.edge_attr
        self.enzyme_batch = enzyme_batch.batch

        # Labels
        self.y = y

    def to(self, device):
        """Move all tensors to device."""
        self.drug_x = self.drug_x.to(device)
        self.drug_edge_index = self.drug_edge_index.to(device)
        if self.drug_edge_attr is not None:
            self.drug_edge_attr = self.drug_edge_attr.to(device)
        self.drug_batch = self.drug_batch.to(device)

        self.enzyme_x = self.enzyme_x.to(device)
        self.enzyme_edge_index = self.enzyme_edge_index.to(device)
        if self.enzyme_edge_attr is not None:
            self.enzyme_edge_attr = self.enzyme_edge_attr.to(device)
        self.enzyme_batch = self.enzyme_batch.to(device)

        self.y = self.y.to(device)
        return self

    @property
    def batch_size(self):
        return self.y.shape[0]


# =============================================================================
# DATASET
# =============================================================================
class DrugEnzymeDataset(Dataset):
    """
    PyTorch Dataset for drug-enzyme binding prediction.

    Args:
        pkl_path:  Path to a split pickle file (train.pkl, val.pkl, test.pkl)
        task:      "regression" (pActivity) or "binary" (label_binary)
        max_samples: Optional cap on dataset size (for debugging/quick runs)
    """
    def __init__(self, pkl_path, task="regression", max_samples=None):
        print(f"Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.samples = pickle.load(f)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.task = task
        print(f"  Loaded {len(self.samples):,} samples (task={task})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        drug_graph = s['drug_graph']
        enzyme_graph = s['enzyme_graph']

        # Ensure both are PyG Data objects
        if not isinstance(drug_graph, Data):
            drug_graph = Data(
                x=torch.tensor(drug_graph['node_features'], dtype=torch.float),
                edge_index=torch.tensor(drug_graph['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(drug_graph['edge_features'], dtype=torch.float),
            )
        if not isinstance(enzyme_graph, Data):
            enzyme_graph = Data(
                x=torch.tensor(enzyme_graph['node_features'], dtype=torch.float),
                edge_index=torch.tensor(enzyme_graph['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(enzyme_graph['edge_features'], dtype=torch.float),
            )

        # Label
        if self.task == "binary":
            y = float(s.get('label_binary', 0))
        else:
            y = float(s['pActivity'])

        return DrugEnzymePair(drug_graph, enzyme_graph, y)


# =============================================================================
# DATALOADER FACTORY
# =============================================================================
def get_dataloaders(data_dir=DATA_DIR, batch_size=64, task=TASK,
                    num_workers=0, max_samples=None):
    """
    Create train/val/test DataLoaders.

    Args:
        data_dir:     Path to final_dataset directory
        batch_size:   Batch size
        task:         "regression" or "binary"
        num_workers:  DataLoader workers (0 = main process)
        max_samples:  Optional cap per split (for debugging)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    collater = DrugEnzymeCollater()

    loaders = {}
    for split in ['train', 'val', 'test']:
        pkl_path = os.path.join(data_dir, f"{split}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing {pkl_path}. Run step5 first.")

        dataset = DrugEnzymeDataset(pkl_path, task=task, max_samples=max_samples)

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
    print("║   Drug-Enzyme Binding Dataset — Test                              ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Load with small cap for quick test
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=4,
        task=TASK,
        max_samples=100,  # only load 100 samples per split for testing
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Inspect one batch
    print("\n" + "=" * 70)
    print("SAMPLE BATCH:")
    print("=" * 70)

    batch = next(iter(train_loader))

    print(f"\n  Batch size:              {batch.batch_size}")
    print(f"\n  Drug graphs:")
    print(f"    Node features:         {batch.drug_x.shape}")
    print(f"    Edge index:            {batch.drug_edge_index.shape}")
    print(f"    Edge features:         {batch.drug_edge_attr.shape}")
    print(f"    Batch vector:          {batch.drug_batch.shape}")
    print(f"    Unique graphs:         {batch.drug_batch.unique().shape[0]}")

    print(f"\n  Enzyme graphs:")
    print(f"    Node features:         {batch.enzyme_x.shape}")
    print(f"    Edge index:            {batch.enzyme_edge_index.shape}")
    print(f"    Edge features:         {batch.enzyme_edge_attr.shape}")
    print(f"    Batch vector:          {batch.enzyme_batch.shape}")
    print(f"    Unique graphs:         {batch.enzyme_batch.unique().shape[0]}")

    print(f"\n  Labels (y):              {batch.y}")
    print(f"  Label shape:             {batch.y.shape}")
    print(f"  Label dtype:             {batch.y.dtype}")

    # Test device transfer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = batch.to(device)
    print(f"\n  Moved to device:         {device}")
    print(f"  drug_x device:           {batch.drug_x.device}")
    print(f"  enzyme_x device:         {batch.enzyme_x.device}")
    print(f"  y device:                {batch.y.device}")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   Dataset test passed! Ready for model training.                  ║")
    print("╚" + "═" * 68 + "╝")