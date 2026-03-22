import cobra
import os

def generate_enzyme_list(model_path, output_path):
    print(f"Loading model from {model_path}...")
    model = cobra.io.read_sbml_model(model_path)
    
    uniprot_ids = set()
    
    print(f"Extracting UniProt IDs from {len(model.genes)} genes...")
    for gene in model.genes:
        up = gene.annotation.get('uniprot')
        if up:
            if isinstance(up, list):
                for u in up:
                    if u.strip():
                        uniprot_ids.add(u.strip())
            elif isinstance(up, str):
                if up.strip():
                    uniprot_ids.add(up.strip())
    
    sorted_ids = sorted(list(uniprot_ids))
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Writing {len(sorted_ids)} unique UniProt IDs to {output_path}...")
    with open(output_path, 'w') as f:
        for uid in sorted_ids:
            f.write(f"{uid}\n")
    
    print("Done.")

if __name__ == "__main__":
    MODEL_PATH = "data/Human-GEM.xml"
    OUTPUT_PATH = "data/enzymes.txt"
    
    if os.path.exists(MODEL_PATH):
        generate_enzyme_list(MODEL_PATH, OUTPUT_PATH)
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
