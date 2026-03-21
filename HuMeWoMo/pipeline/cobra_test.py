import cobra

MODEL_PATH = "data/Human-GEM.xml"
OBJECTIVE = "MAR13082" 
TARGET_REACTION = "MAR04358" # The reaction your drug targets
INHIBITION_FRACTION = 0.80   # 80% inhibition

print(f"Loading {MODEL_PATH}...")
model = cobra.io.read_sbml_model(MODEL_PATH)
model.objective = OBJECTIVE

# 1. Establish the wild-type baseline
wt_solution = model.optimize()
wt_flux = wt_solution.fluxes[TARGET_REACTION]
print(f"Wild-Type Objective: {wt_solution.objective_value}")
print(f"Wild-Type Target Flux: {wt_flux}")

# 2. Apply the fractional constraint
rxn = model.reactions.get_by_id(TARGET_REACTION)

# Check directionality to apply the constraint to the active bound
if wt_flux > 0:
    rxn.upper_bound = wt_flux * (1 - INHIBITION_FRACTION)
elif wt_flux < 0:
    rxn.lower_bound = wt_flux * (1 - INHIBITION_FRACTION)

# 3. Simulate the inhibited network
inhibited_solution = model.optimize()
print(f"Inhibited Objective: {inhibited_solution.objective_value}")
print(f"Inhibited Target Flux: {inhibited_solution.fluxes[TARGET_REACTION]}")
