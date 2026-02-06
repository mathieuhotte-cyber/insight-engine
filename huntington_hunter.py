import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATA GENERATION ---
def generate_huntington_seq(repeats):
    prefix = "ATGAAGGCCTTCGAGTCCCTCAAGTCCTTCCAGCAGCAG" 
    suffix = "CAACAGCCGCCACCGCCGCCGCCGCCGCCGCCGCCTCCTCAGCTTCCTCAG"
    return prefix + ("CAG" * repeats) + suffix

def get_clinical_variants():
    return [
        ("Healthy (18 repeats)", generate_huntington_seq(18)),
        ("Gray Zone (30 repeats)", generate_huntington_seq(30)),
        ("Pathogenic (45 repeats)", generate_huntington_seq(45))
    ]

# --- 2. THE LEAKY INTEGRATOR ---
MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

class LeakyIntegrator(nn.Module):
    def __init__(self):
        super().__init__()
        # Stickiness: How much does 'CAG' add to the pile?
        # We learn this parameter, but initialize it to "sticky"
        self.stickiness = nn.Parameter(torch.tensor([0.1, 0.1, 0.5, 0.1])) # G (index 2) is sticky
        
        # Clearance Rate: How fast does the cell clean up?
        self.clearance_rate = nn.Parameter(torch.tensor(0.05))
        
        # Aggregation Threshold: When does the cell die?
        self.threshold = 5.0
        
    def forward(self, seq):
        aggregate_mass = 0.0
        trajectory = []
        
        for char in seq:
            if char in MAP:
                # 1. Ingestion: Add mass based on stickiness of the base
                idx = MAP[char]
                aggregate_mass += torch.abs(self.stickiness[idx])
                
                # 2. Clearance: The cell tries to clean up
                aggregate_mass -= torch.abs(self.clearance_rate)
                
                # Physics constraint: Mass cannot be negative
                if aggregate_mass < 0: aggregate_mass = 0
                
                trajectory.append(aggregate_mass)
                
        return torch.tensor(trajectory)

# --- 3. EXPERIMENT ---
def run_biophysical_test():
    print("--- HUNTINGTON'S DISEASE: LEAKY INTEGRATOR MODEL ---")
    print("Simulating Protein Aggregation vs. Cellular Clearance\n")
    
    variants = get_clinical_variants()
    model = LeakyIntegrator()
    
    # Visualization
    plt.figure(figsize=(10, 6))
    colors = {'Healthy (18 repeats)': 'green', 
              'Gray Zone (30 repeats)': 'orange', 
              'Pathogenic (45 repeats)': 'red'}
    
    final_masses = {}
    
    for name, seq in variants:
        traj = model(seq).detach().numpy()
        final_masses[name] = np.max(traj)
        
        plt.plot(traj, label=name, color=colors[name], linewidth=2)
    
    # Add Threshold Line
    plt.axhline(y=model.threshold, color='black', linestyle='--', label='Cell Death Threshold')
    
    plt.title("Biophysical Simulation: Poly-Q Aggregation")
    plt.ylabel("Aggregate Mass (Arbitrary Units)")
    plt.xlabel("Sequence Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Results
    print("--- DIAGNOSTIC READOUT ---")
    h_mass = final_masses["Healthy (18 repeats)"]
    p_mass = final_masses["Pathogenic (45 repeats)"]
    
    print(f"Healthy Peak Mass:    {h_mass:.2f}")
    print(f"Pathogenic Peak Mass: {p_mass:.2f}")
    
    if p_mass > model.threshold and h_mass < model.threshold:
        print("\nDIAGNOSIS: SUCCESS.")
        print("Model correctly predicts overflow for Pathogenic variant only.")
    else:
        print("\nDIAGNOSIS: FAILURE.")

if __name__ == "__main__":
    run_biophysical_test()