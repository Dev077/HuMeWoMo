import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

class HomoGraphEncoder(nn.Module):
    """
    Transformer-based encoder for homogeneous graphs (where edges are nodes).
    Each node (atom/bond or residue/contact) queries its neighbors to update its state.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        for _ in range(num_layers):
            # 1. Multi-head Attention (Graph Convolution)
            self.convs.append(TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                edge_dim=None
            ))
            self.norms1.append(nn.LayerNorm(hidden_channels))

            # 2. Feed-Forward Network (FC layers)
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.ReLU(),
                nn.Linear(hidden_channels * 2, hidden_channels)
            ))
            self.norms2.append(nn.LayerNorm(hidden_channels))
        
        # Final projections to produce KV pairs for the decoder
        self.lin_out_k = nn.Linear(hidden_channels, out_channels)
        self.lin_out_v = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin_in(x)
        
        for conv, norm1, ffn, norm2 in zip(self.convs, self.norms1, self.ffns, self.norms2):
            # Attention + Residual + Norm
            h = conv(x, edge_index)
            x = norm1(x + h)
            
            # Feed-Forward + Residual + Norm
            h = ffn(x)
            x = norm2(x + h)
        
        # Generate the final KV representation for each node
        k = self.lin_out_k(x)
        v = self.lin_out_v(x)
        return k, v

class DecoderBlock(nn.Module):
    """
    A single decoder block that generates queries from latents, 
    scans both molecules, and updates the latents.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        # Explicit query generators for each molecule
        self.q_proj_drug = nn.Linear(embed_dim, embed_dim)
        self.q_proj_enzyme = nn.Linear(embed_dim, embed_dim)
        
        self.drug_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enzyme_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, latents, d_k, d_v, d_mask, e_k, e_v, e_mask):
        # 1. Pre-Norm and Query Generation
        # We generate the "lookups" based on our current latent state
        norm_latents = self.ln1(latents)
        q_d = self.q_proj_drug(norm_latents)
        q_e = self.q_proj_enzyme(norm_latents)
        
        # 2. Parallel Cross-Attention
        # Each query scans its respective molecule
        attn_d, _ = self.drug_attn(query=q_d, key=d_k, value=d_v, key_padding_mask=~d_mask)
        attn_e, _ = self.enzyme_attn(query=q_e, key=e_k, value=e_v, key_padding_mask=~e_mask)
        
        # 3. Update Latents (First Residual)
        latents = latents + attn_d + attn_e
        
        # 4. Feed-Forward (Second Residual with Pre-Norm)
        h = self.ffn(self.ln2(latents))
        latents = latents + h
        
        return latents

class CrossAttentionDecoder(nn.Module):
    """
    Latent-Refinement Decoder. 
    Starts with learned latent embeddings and refines them through multiple blocks.
    """
    def __init__(self, embed_dim, num_latents=16, num_layers=3, num_heads=4):
        super().__init__()
        # The "learned embedding for the first iteration"
        self.latent_embeddings = nn.Parameter(torch.randn(1, num_latents, embed_dim))
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.final_head = nn.Sequential(
            nn.Linear(embed_dim * num_latents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Softplus() # Ensures pActivity is always > 0
        )

    def forward(self, drug_k, drug_v, enzyme_k, enzyme_v, drug_batch, enzyme_batch):
        # Flattened -> Dense
        d_k_dense, d_mask = to_dense_batch(drug_k, drug_batch)
        d_v_dense, _      = to_dense_batch(drug_v, drug_batch)
        e_k_dense, e_mask = to_dense_batch(enzyme_k, enzyme_batch)
        e_v_dense, _      = to_dense_batch(enzyme_v, enzyme_batch)

        batch_size = d_k_dense.size(0)
        
        # Initial latents: [Batch, Num_Latents, Dim]
        latents = self.latent_embeddings.expand(batch_size, -1, -1)
        
        # Refine latents through blocks
        for layer in self.layers:
            latents = layer(latents, d_k_dense, d_v_dense, d_mask, e_k_dense, e_v_dense, e_mask)
        
        # Final output
        flat_latents = latents.reshape(batch_size, -1)
        return self.final_head(flat_latents).squeeze(-1)

class HomoBindingModel(nn.Module):
    """
    The full model orchestrating Drug/Enzyme encoders and the Cross-Attention decoder.
    """
    def __init__(self, drug_in_dim=50, enzyme_in_dim=27, hidden_dim=128, n_heads=4):
        super().__init__()
        self.drug_encoder = HomoGraphEncoder(drug_in_dim, hidden_dim, hidden_dim, heads=n_heads)
        self.enzyme_encoder = HomoGraphEncoder(enzyme_in_dim, hidden_dim, hidden_dim, heads=n_heads)
        self.decoder = CrossAttentionDecoder(hidden_dim, num_heads=n_heads)

    def forward(self, batch):
        # 1. Encode Drug Graph
        d_k, d_v = self.drug_encoder(batch.drug_x, batch.drug_edge_index)
        
        # 2. Encode Enzyme Graph
        e_k, e_v = self.enzyme_encoder(batch.enzyme_x, batch.enzyme_edge_index)
        
        # 3. Cross-Attention and Prediction
        prediction = self.decoder(
            d_k, d_v, 
            e_k, e_v, 
            batch.drug_batch, 
            batch.enzyme_batch
        )
        
        return prediction

if __name__ == "__main__":
    import os
    from ..datasets.homo_binding_dataset import get_homo_dataloaders

    print("Checking for dataset...")
    DATA_PATH = "./bindingdb_data/final_dataset"
    
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Initializing model with dummy batch instead.")
        class DummyBatch:
            def __init__(self):
                self.drug_x = torch.randn(20, 50)
                self.drug_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                self.drug_batch = torch.zeros(20, dtype=torch.long)
                self.enzyme_x = torch.randn(100, 40)
                self.enzyme_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                self.enzyme_batch = torch.zeros(100, dtype=torch.long)
                self.y = torch.tensor([7.0])
        batch = DummyBatch()
    else:
        print("Loading real test sample...")
        try:
            from ..datasets.homo_binding_dataset import HomoDrugEnzymeDataset
            from ..datasets.binding_dataset import DrugEnzymeCollater
            
            test_pkl = os.path.join(DATA_PATH, "test.pkl")
            dataset = HomoDrugEnzymeDataset(test_pkl, max_samples=10)
            collater = DrugEnzymeCollater()
            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, collate_fn=collater
            )
            batch = next(iter(test_loader))
        except Exception as e:
            print(f"Error loading real data: {e}. Falling back to dummy.")
            class DummyBatch:
                def __init__(self):
                    self.drug_x = torch.randn(20, 50); self.drug_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long); self.drug_batch = torch.zeros(20, dtype=torch.long)
                    self.enzyme_x = torch.randn(100, 40); self.enzyme_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long); self.enzyme_batch = torch.zeros(100, dtype=torch.long)
                    self.y = torch.tensor([7.0])
            batch = DummyBatch()

    # Initialize model
    model = HomoBindingModel(
        drug_in_dim=50, 
        enzyme_in_dim=27, 
        hidden_dim=64, 
        n_heads=4
    )
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(batch)
    
    print("\n" + "="*50)
    print("INFERENCE TEST (Batch Size 1)")
    print("="*50)
    print(f"Input Drug Nodes:   {batch.drug_x.shape[0]}")
    print(f"Input Enzyme Nodes: {batch.enzyme_x.shape[0]}")
    print(f"Target pActivity:   {batch.y.item():.4f}")
    print(f"Model Prediction:   {output.item():.4f}")
    print("="*50)
