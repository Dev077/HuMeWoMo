import cobra

def simulate_batch_knockout(model, gene_list):
    with model:
        valid_genes = [gene for gene in gene_list if gene in model.genes]
        print(f"Knocking out: {valid_genes}")
        
        cobra.manipulation.knock_out_model_genes(model, valid_genes)
        solution = model.optimize()
        return solution


if __name__ == "__main__":
    print("Loading model...")
    model = cobra.io.read_sbml_model("data/Human-GEM.xml")
    model.objective = "MAR13082"
    
    gene_knockouts = [
        "ENSG00000134057",
        "ENSG00000140740",
        "ENSG00000073578"
    ]
    
    print("Optimizing...")
    mutant_solution = simulate_batch_knockout(model, gene_knockouts)
    
    if mutant_solution.status == 'optimal':
        print(f"Knockout Objective: {mutant_solution.objective_value}")
    else:
        print("Model could not find an optimal solution (infeasible).")
