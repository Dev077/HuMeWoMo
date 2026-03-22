import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class HomoGraphEncoder(nn.Module):
    """
    Transformer-based encoder for homogeneous graphs.
    Returns a pooled vector per graph using both mean and max pooling (dual-pooling).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=8, heads=4):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                edge_dim=None,
                dropout=0.1
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.lin_in(x)
        
        for conv, norm in zip(self.convs, self.norms):
            # Attention + Residual + Norm
            h = conv(x, edge_index)
            x = norm(x + h)
        
        # 1. Project to out_channels
        x = self.lin_out(x)
        
        # 2. Dual-Pooling (GAP + GMP) -> [Batch, out_channels * 2]
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        return torch.cat([x_mean, x_max], dim=1)

class InteractionDecoder(nn.Module):
    """
    Simplified Interaction Decoder.
    Performs a multiplicative interaction on pooled representations before the final MLP.
    """
    def __init__(self, drug_dim, enzyme_dim, hidden_dim, num_layers=4):
        super().__init__()
        
        # Encoders now return 2x hidden_dim due to dual-pooling
        self.drug_map = nn.Linear(drug_dim * 2, hidden_dim)
        self.enzyme_map = nn.Linear(enzyme_dim * 2, hidden_dim)
        
        layers = []
        curr_dim = hidden_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, curr_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.mlp = nn.Sequential(*layers)
        self.final_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures pActivity > 0 and provides smoother gradients than ReLU
        )

    def forward(self, d_pooled, e_pooled):
        # 1. Map to common hidden space
        d_h = self.drug_map(d_pooled)
        e_h = self.enzyme_map(e_pooled)
        
        # 2. Multiplicative Interaction (Hadamard product)
        interaction = d_h * e_h
        
        # 3. Final MLP
        out = self.mlp(interaction)
        return self.final_head(out).squeeze(-1)

class SimpleHomoBindingModel(nn.Module):
    """
    A simplified model orchestrating Drug/Enzyme encoders and an interaction decoder.
    """
    def __init__(self, 
                 drug_in_dim=50, 
                 enzyme_in_dim=27, 
                 hidden_dim=256, 
                 n_heads=4,
                 num_drug_layers=8,
                 num_enzyme_layers=8,
                 num_decoder_layers=4):
        super().__init__()
        self.drug_encoder = HomoGraphEncoder(drug_in_dim, hidden_dim, hidden_dim, num_layers=num_drug_layers, heads=n_heads)
        self.enzyme_encoder = HomoGraphEncoder(enzyme_in_dim, hidden_dim, hidden_dim, num_layers=num_enzyme_layers, heads=n_heads)
        self.decoder = InteractionDecoder(hidden_dim, hidden_dim, hidden_dim, num_layers=num_decoder_layers)

    def forward(self, batch):
        # 1. Encode and Pool Graphs
        d_pooled = self.drug_encoder(batch.drug_x, batch.drug_edge_index, batch.drug_batch)
        e_pooled = self.enzyme_encoder(batch.enzyme_x, batch.enzyme_edge_index, batch.enzyme_batch)
        
        # 2. Interaction and Prediction
        prediction = self.decoder(d_pooled, e_pooled)
        
        return prediction

if __name__ == "__main__":
    import os
    import sys
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)

    print("Running Dimension Verification Test...")
    DATA_PATH = "./bindingdb_data/final_dataset"
    
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Using dummy data for shape test.")
        # Dummy batch simulation
        class DummyBatch:
            def __init__(self):
                self.drug_x = torch.randn(10, 48)
                self.drug_edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
                self.drug_batch = torch.zeros(10, dtype=torch.long)
                self.enzyme_x = torch.randn(20, 27)
                self.enzyme_edge_index = torch.tensor([[0,1],[1,0]], dtype=torch.long)
                self.enzyme_batch = torch.zeros(20, dtype=torch.long)
                self.y = torch.tensor([1.0])
        batch = DummyBatch()
    else:
        try:
            from ..datasets.homo_binding_dataset import HomoDrugEnzymeDataset
            from ..datasets.binding_dataset import DrugEnzymeCollater
            
            test_pkl = os.path.join(DATA_PATH, "test.pkl")
            dataset = HomoDrugEnzymeDataset(test_pkl, max_samples=4)
            collater = DrugEnzymeCollater()
            loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collater)
            batch = next(iter(loader))
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    print("\n" + "="*30)
    print("DATA SHAPES")
    print("="*30)
    print(f"Drug Node Features:   {batch.drug_x.shape}")
    print(f"Enzyme Node Features: {batch.enzyme_x.shape}")
    
    drug_in = batch.drug_x.shape[1]
    enzyme_in = batch.enzyme_x.shape[1]
    
    # Update defaults if they were wrong
    model = SimpleHomoBindingModel(
        drug_in_dim=drug_in,
        enzyme_in_dim=enzyme_in,
        hidden_dim=128
    )
    
    print("\n" + "="*30)
    print("MODEL PARAMETERS")
    print("="*30)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(batch)
    print(f"Output Shape: {out.shape}")
    print(f"Sample Preds: {out}")
    print("="*30)
