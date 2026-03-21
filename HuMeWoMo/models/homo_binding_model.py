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
    A single decoder block that queries both encoders in parallel and merges the results.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        # Parallel attention: each has its own internal query/key/value projections
        self.drug_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enzyme_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, queries, d_k, d_v, d_mask, e_k, e_v, e_mask):
        # 1. Parallel Cross-Attention
        # The latents query both molecules "at the same time"
        attn_d, _ = self.drug_attn(query=queries, key=d_k, value=d_v, key_padding_mask=~d_mask)
        attn_e, _ = self.enzyme_attn(query=queries, key=e_k, value=e_v, key_padding_mask=~e_mask)
        
        # 2. Combine results and apply first residual + norm
        # This fuses drug-specific and enzyme-specific context into the latents
        queries = self.norm1(queries + attn_d + attn_e)
        
        # 3. Feed-Forward and second residual + norm
        h = self.ffn(queries)
        queries = self.norm2(queries + h)
        
        return queries

class CrossAttentionDecoder(nn.Module):
    """
    Learned Latent Decoder. 
    A set of 'binding queries' attend to both drug and enzyme repeatedly.
    """
    def __init__(self, embed_dim, num_queries=16, num_layers=3, num_heads=4):
        super().__init__()
        # Learned "binding sites" or latent representations of the interaction
        self.latent_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.final_head = nn.Sequential(
            nn.Linear(embed_dim * num_queries, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, drug_k, drug_v, enzyme_k, enzyme_v, drug_batch, enzyme_batch):
        # Flattened -> Dense
        d_k_dense, d_mask = to_dense_batch(drug_k, drug_batch)
        d_v_dense, _      = to_dense_batch(drug_v, drug_batch)
        e_k_dense, e_mask = to_dense_batch(enzyme_k, enzyme_batch)
        e_v_dense, _      = to_dense_batch(enzyme_v, enzyme_batch)

        batch_size = d_k_dense.size(0)
        
        # Expand latent queries to batch size
        # queries: [Batch, Num_Queries, Dim]
        queries = self.latent_queries.expand(batch_size, -1, -1)
        
        # Apply decoder blocks
        for layer in self.layers:
            queries = layer(queries, d_k_dense, d_v_dense, d_mask, e_k_dense, e_v_dense, e_mask)
        
        # Flatten all queries for the final prediction head
        # Alternatively, could pool them: queries.mean(dim=1)
        flat_queries = queries.reshape(batch_size, -1)
        return self.final_head(flat_queries).squeeze(-1)

class HomoBindingModel(nn.Module):
    """
    The full model orchestrating Drug/Enzyme encoders and the Cross-Attention decoder.
    """
    def __init__(self, drug_in_dim=50, enzyme_in_dim=40, hidden_dim=128, n_heads=4):
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
    # Quick sanity check with dummy dimensions
    # drug: 50 features (37 atom + 11 bond + 2 type)
    # enzyme: 40 features (approx)
    model = HomoBindingModel(drug_in_dim=50, enzyme_in_dim=40, hidden_dim=64)
    print(model)
    print(f"\nModel initialized successfully with {sum(p.numel() for p in model.parameters()):,} parameters.")
