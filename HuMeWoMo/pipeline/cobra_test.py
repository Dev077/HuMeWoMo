import cobra


def simulate_batch_inhibition(model, baseline_fluxes, target_inhibitions):
    with model:
        for target_id, inhibition_fraction in target_inhibitions.items():
            try:
                rxn = model.reactions.get_by_id(target_id)
            except KeyError:
                continue 

            wt_flux = baseline_fluxes[target_id]
            capacity = 1.0 - inhibition_fraction
            
            if wt_flux > 0:
                rxn.upper_bound = wt_flux * capacity
            elif wt_flux < 0:
                rxn.lower_bound = wt_flux * capacity

        solution = model.optimize()
        return solution

    
if __name__ == "__main__":
    model = cobra.io.read_sbml_model("data/Human-GEM.xml")
    model.objective = "MAR13082"
    
    wt_solution = model.optimize()
    baseline_fluxes = wt_solution.fluxes
    
    # Mapping enzyme inhibitions to reaction constraints based on GPR logic
    # would occur here before passing the dictionary to the simulator.
    drug_profile = {
        "MAR04358": 0.80,
        "MAR04359": 0.50,
        "MAR01234": 0.95
    }
    
    mutant_solution = simulate_batch_inhibition(model, baseline_fluxes, drug_profile)
    
    if mutant_solution.status == 'optimal':
        print(f"Inhibited Objective: {mutant_solution.objective_value}")
    else:
        print("Model could not find an optimal solution (infeasible).")

