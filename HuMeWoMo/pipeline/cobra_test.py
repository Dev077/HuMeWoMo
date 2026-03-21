import cobra
import ast
import re

def eval_fractional_ast(node, gene_capacities):
    if isinstance(node, ast.Name):
        # Revert the safe formatting to look up the original gene ID
        original_id = node.id.replace("_DASH_", "-").replace("_DOT_", ".")
        return gene_capacities.get(original_id, 1.0)
    elif isinstance(node, ast.BoolOp):
        values = [eval_fractional_ast(v, gene_capacities) for v in node.values]
        if isinstance(node.op, ast.And):
            return min(values)
        elif isinstance(node.op, ast.Or):
            return max(values)
    elif isinstance(node, ast.Expression):
        return eval_fractional_ast(node.body, gene_capacities)
    return 1.0

def calculate_reaction_capacity(rule_str, gene_capacities):
    if not rule_str:
        return 1.0
    
    # Sanitize gene IDs so they parse as valid Python AST nodes
    safe_rule = rule_str.replace("-", "_DASH_").replace(".", "_DOT_")
    
    try:
        tree = ast.parse(safe_rule, mode='eval')
        return eval_fractional_ast(tree, gene_capacities)
    except SyntaxError:
        return 1.0

def simulate_batch_gene_inhibition(model, baseline_fluxes, gene_inhibitions):
    # Convert inhibition fractions to functional capacities (e.g., 0.8 inhibition -> 0.2 capacity)
    gene_capacities = {k: max(0.0, 1.0 - v) for k, v in gene_inhibitions.items()}
    
    with model:
        for rxn in model.reactions:
            if not rxn.gene_reaction_rule:
                continue
            
            capacity = calculate_reaction_capacity(rxn.gene_reaction_rule, gene_capacities)
            
            if capacity < 1.0:
                wt_flux = baseline_fluxes[rxn.id]
                if wt_flux > 0:
                    rxn.upper_bound = wt_flux * capacity
                elif wt_flux < 0:
                    rxn.lower_bound = wt_flux * capacity

        solution = model.optimize()
        return solution



if __name__ == "__main__":
    print("Loading model...")
    model = cobra.io.read_sbml_model("data/Human-GEM.xml")
    model.objective = "MAR13082"
    
    print("Optimizing wild...")
    wt_solution = model.optimize()
    baseline_fluxes = wt_solution.fluxes
    
    # Dictionary of Enzyme (Gene) IDs and their predicted inhibition fraction
    gene_predictions = {
        "ENSG00000134057": 0.80,
        "ENSG00000140740": 0.50,
        "ENSG00000073578": 0.95
    }
    
    print("Optimizing inhibited...")
    mutant_solution = simulate_batch_gene_inhibition(model, baseline_fluxes, gene_predictions)
    
    if mutant_solution.status == 'optimal':
        print(f"Inhibited Objective: {mutant_solution.objective_value}")
    else:
        print("Model could not find an optimal solution (infeasible).")
