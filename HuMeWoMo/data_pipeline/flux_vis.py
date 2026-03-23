import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import cobra
from matplotlib.colors import LinearSegmentedColormap

def visualize_flux_delta(model_path, baseline_csv, inhibited_csv, output_png, top_n=25):
    """
    Identifies top N affected reactions and plots a single graph showing the DELTA.
    Blue edges = Increase in flux
    Red edges = Decrease in flux
    """
    if not os.path.exists(baseline_csv) or not os.path.exists(inhibited_csv):
        print("Error: Missing flux CSVs.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Load data
    print("Loading model and fluxes...")
    model = cobra.io.read_sbml_model(model_path)
    df_base = pd.read_csv(baseline_csv, index_col=0)
    df_inhib = pd.read_csv(inhibited_csv, index_col=0)

    # Calculate signed delta
    val_col_base = df_base.columns[0]
    val_col_inhib = df_inhib.columns[0]
    signed_delta = df_inhib[val_col_inhib] - df_base[val_col_base]
    abs_delta = signed_delta.abs()
    
    # Get top N most changed reactions
    top_rxn_ids = abs_delta.sort_values(ascending=False).head(top_n).index.tolist()
    print(f"Top {top_n} reactions selected for delta visualization.")

    # 2. Build the graph
    G = nx.MultiDiGraph()
    for rxn_id in top_rxn_ids:
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            d_val = signed_delta[rxn_id]
            
            # Skip if no change
            if abs(d_val) < 1e-9: continue

            subs = [m for m in rxn.metabolites if rxn.get_coefficient(m) < 0]
            prods = [m for m in rxn.metabolites if rxn.get_coefficient(m) > 0]
            
            if subs and prods:
                s_main = sorted(subs, key=lambda m: abs(rxn.get_coefficient(m)), reverse=True)[0]
                p_main = sorted(prods, key=lambda m: abs(rxn.get_coefficient(m)), reverse=True)[0]
                
                # We use the name if possible, otherwise ID
                s_label = s_main.name if hasattr(s_main, 'name') and s_main.name else s_main.id
                p_label = p_main.name if hasattr(p_main, 'name') and p_main.name else p_main.id
                
                G.add_edge(s_label, p_label, id=rxn_id, delta=d_val)
        except KeyError:
            continue

    if not G.edges():
        print("No significant flux changes found to plot.")
        return

    # 3. Plotting
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.0, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#F5F5F5', edgecolors='#CCCCCC', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Prepare edges
    edges = G.edges(data=True)
    delta_vals = [d['delta'] for u, v, d in edges]
    max_d = max([abs(x) for x in delta_vals]) if delta_vals else 1

    # Colors: Blue for positive delta (increase), Red for negative (decrease)
    # Verdigris (#43B3AE) could be used for the positive ones if preferred, 
    # but Red/Blue is standard for Delta. Let's use Verdigris for increases!
    edge_colors = ['#43B3AE' if d['delta'] > 0 else '#FF5252' for u, v, d in edges]
    edge_widths = [1 + 8 * (abs(d['delta'])/max_d) for u, v, d in edges]

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                           arrowsize=20, alpha=0.7, connectionstyle="arc3,rad=0.1")

    # Edge labels: ID + value
    edge_labels = {(u, v): f"{d['id']}\n({d['delta']:.2e})" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    # Legend/Info
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#43B3AE', lw=4, label='Flux Increase'),
        Line2D([0], [0], color='#FF5252', lw=4, label='Flux Decrease')
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.title(f"Metabolic Flux Delta (Inhibition Impact)\nTop {top_n} Reactions by Magnitude of Change", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Delta visualization saved to {output_png}")

if __name__ == "__main__":
    root_data = "data"
    model = os.path.join(root_data, "Human-GEM.xml")
    base = os.path.join(root_data, "baseline_fluxes.csv")
    inhib = os.path.join(root_data, "inhibited_fluxes.csv")
    output = os.path.join(root_data, "flux_change.png")

    # Fallbacks
    if not os.path.exists(model):
        for prefix in ["../../", "../"]:
            if os.path.exists(os.path.join(prefix, model)):
                model = os.path.join(prefix, model)
                base = os.path.join(prefix, base)
                inhib = os.path.join(prefix, inhib)
                output = os.path.join(prefix, output)
                break

    visualize_flux_delta(model, base, inhib, output, top_n=25)
