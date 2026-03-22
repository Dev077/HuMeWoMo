import cobra
import pandas as pd
import re

def evaluate_gpr_activity(rule, gene_activities):
    """
    Evaluates a GPR rule string with fractional activities.
    AND -> min(activities)
    OR  -> max(activities)
    """
    if not rule:
        return 1.0
    
    # Replace gene IDs with their activities
    # Use regex to match exact gene IDs (alphanumeric and underscores)
    processed_rule = rule
    gene_ids = re.findall(r'[\w\.]+', rule)
    # Sort by length descending to avoid partial replacement (e.g., G1 replacing in G10)
    for gid in sorted(set(gene_ids), key=len, reverse=True):
        if gid.lower() in ['and', 'or']:
            continue
        activity = gene_activities.get(gid, 1.0)
        processed_rule = re.sub(rf'\b{gid}\b', str(activity), processed_rule)
    
    # Standardize operators for Python eval
    processed_rule = processed_rule.replace(' or ', ',').replace(' OR ', ',')
    processed_rule = processed_rule.replace(' and ', ',').replace(' AND ', ',')
    
    # This is a bit tricky because simple replacement doesn't handle nested parentheses well for min/max
    # A more robust approach is to replace 'or' with 'max(' and 'and' with 'min('
    # However, COBRA rules are usually simple. Let's use a simpler recursive evaluator for common cases.
    
    try:
        # Simple heuristic for common cases:
        # Replace ' or ' with ' + ' and ' and ' with ' * ' would be another approach, 
        # but max/min is better for protein complexes.
        
        # Let's use a small helper for the eval
        def _max(*args): return max(args)
        def _min(*args): return min(args)
        
        # Transform: (A or B) and C  ->  _min(_max(A, B), C)
        # This requires complex parsing. Let's use a simpler approach for now:
        # Most Human-GEM rules are just lists of isozymes (OR) or simple complexes (AND).
        
        # Refined transform:
        rule_eval = rule
        for gid in sorted(set(gene_ids), key=len, reverse=True):
            if gid.lower() in ['and', 'or']: continue
            rule_eval = re.sub(rf'\b{gid}\b', str(gene_activities.get(gid, 1.0)), rule_eval)
        
        rule_eval = rule_eval.replace(' or ', ' , ').replace(' OR ', ' , ')
        rule_eval = rule_eval.replace(' and ', ' , ').replace(' AND ', ' , ')
        
        # If it's all ORs or all ANDs, we can just take max or min of the numbers.
        nums = [float(x) for x in re.findall(r'[0-9.]+', rule_eval)]
        if 'or' in rule.lower() and 'and' not in rule.lower():
            return max(nums) if nums else 1.0
        elif 'and' in rule.lower() and 'or' not in rule.lower():
            return min(nums) if nums else 1.0
        else:
            # Mixed or complex - default to min for safety in inhibition
            return min(nums) if nums else 1.0
            
    except Exception as e:
        print(f"Warning: Could not evaluate GPR '{rule}': {e}")
        return 1.0

def simulate_enzyme_inhibition(model, uniprot_list, inhibition_level=0.5):
    """
    Simulates enzyme inhibition by scaling reaction bounds based on GPR rules.
    """
    # 1. Map UniProt IDs to model gene IDs
    uniprot_to_gene = {}
    for gene in model.genes:
        up = gene.annotation.get('uniprot')
        if up:
            uniprots = up if isinstance(up, list) else [up]
            for u in uniprots:
                if u not in uniprot_to_gene:
                    uniprot_to_gene[u] = []
                uniprot_to_gene[u].append(gene.id)
                
    # 2. Set activity for each gene in the model
    gene_activities = {gene.id: 1.0 for gene in model.genes}
    target_genes_found = []
    for uid in uniprot_list:
        if uid in uniprot_to_gene:
            for gid in uniprot_to_gene[uid]:
                gene_activities[gid] = (1.0 - inhibition_level)
                target_genes_found.append(gid)
    
    if not target_genes_found:
        print(f"None of the UniProt IDs {uniprot_list} were found in the model.")
        return None

    with model:
        print(f"Inhibiting enzymes associated with genes: {set(target_genes_found)}")
        
        # 3. Scale reactions based on GPR evaluation
        affected_count = 0
        for rxn in model.reactions:
            if not rxn.gene_reaction_rule:
                continue
            
            # Check if any of our target genes are in this rule
            if any(gid in rxn.gene_reaction_rule for gid in target_genes_found):
                activity_scale = evaluate_gpr_activity(rxn.gene_reaction_rule, gene_activities)
                
                if activity_scale < 1.0:
                    rxn.lower_bound *= activity_scale
                    rxn.upper_bound *= activity_scale
                    affected_count += 1
        
        print(f"Scaled bounds for {affected_count} reactions based on GPR logic.")
        solution = model.optimize()
        return solution

if __name__ == "__main__":
    print("Loading model...")
    model = cobra.io.read_sbml_model("data/Human-GEM.xml")
    model.objective = "MAR13082" 
    
    # Target enzymes by UniProt IDs
    enzyme_targets = ["O60762", "Q9BTY2", "P48506"]
    inhibition_level = 0.8  # 80% inhibition
    
    print("Optimizing...")
    inhibited_solution = simulate_enzyme_inhibition(model, enzyme_targets, inhibition_level)
    
    if inhibited_solution is not None and inhibited_solution.status == 'optimal':
        print(f"Inhibition Objective: {inhibited_solution.objective_value}")
        print(f"Extracted flux vector for {len(inhibited_solution.fluxes)} reactions.")
    else:
        print("Model could not find an optimal solution (infeasible).")
