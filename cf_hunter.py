import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATA INGESTION (CFTR Exon 11) ---
def fetch_cftr_data():
    # CFTR Exon 11 (hg38) - The location of the Delta F508 mutation
    # Sequence centered on the F508 region (Ile507 - Phe508 - Gly509)
    # WT: ... ATC TTT GGT ...
    CFTR_WT = "AATAACTTTGCAACAGTGGAGGAAAGCCTTTGGAGTGATACCACAGGTGCCATCAAAACAAGGCAATGTTGAAAAGGCACCTATGCCTAAAATGATGATATTTGGATCTTTGGATGAACTTTCCTATGAAAACAGCTGAAGAGCTTTGCAT"
    return CFTR_WT

def induce_cf_mutations(wild_type):
    # Locate the critical "ATCTTTGGT" motif (Ile-Phe-Gly)
    # 1. Healthy (Wild Type)
    # 2. Benign (Silent Mutation): Change TTT (Phe) to TTC (Phe).
    #    This changes the letter but keeps the meaning (Protein stays the same).
    benign_seq = wild_type.replace("ATCTTTGGT", "ATCTTCGGT")
    
    # 3. Pathogenic (Delta F508): DELETE the "TTT" (or CTT).
    #    This removes the Phenylalanine. The protein misfolds and is destroyed.
    #    Sequence becomes: ATC --- GGT
    pathogenic_seq = wild_type.replace("ATCTTTGGT", "ATCGGT") # Deletion
    
    if benign_seq == wild_type: print("ERROR: Failed to create Benign variant")
    if pathogenic_seq == wild_type: print("ERROR: Failed to create Pathogenic variant")
        
    return [("Healthy CFTR", wild_type), ("Benign (Silent)", benign_seq), ("Delta F508 (CF)", pathogenic_seq)]

# --- 2. THE CONSCIOUSNESS ENGINE (Re-used) ---
MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def encode(seq):
    X, y = [], []
    win = 15
    for i in range(len(seq)-win):
        window = seq[i:i+win]
        target = seq[i+win]
        if any(c not in MAP for c in window) or target not in MAP: continue
        oh = torch.zeros(win, 4)
        for p, c in enumerate(window): oh[p][MAP[c]] = 1
        X.append(oh.flatten())
        y.append(MAP[target])
    return torch.stack(X), torch.tensor(y)

class ConsciousnessNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Compressing DNA into a "Concept" (Latent Space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 32), # The Bottleneck (Force abstraction)
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 4)
        )
        self.target_compression = 0.3 

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent
    
    def compression_ratio(self, x):
        latent = self.encoder(x)
        return torch.std(latent).item()

class DiseaseDetector:
    def train_healthy_model(self, healthy_seq):
        X, y = encode(healthy_seq)
        model = ConsciousnessNet(X.shape[1])
        opt = optim.AdamW(model.parameters(), lr=0.005)
        
        # Learn the "Shape" of Healthy CFTR
        for epoch in range(400):
            opt.zero_grad()
            output, latent = model(X)
            pred_loss = F.cross_entropy(output, y)
            # Homeostasis Loss: Keep the compression tight but stable
            comp_loss = torch.abs(torch.std(latent) - model.target_compression)
            total_loss = pred_loss + 0.1 * comp_loss
            total_loss.backward()
            opt.step()
        return model
    
    def measure_adaptation_energy(self, base_model, test_seq):
        """
        The Core Insight: 
        How much energy (gradient descent) does it take to understand this mutation?
        """
        X, y = encode(test_seq)
        
        # Clone the healthy mind
        test_model = ConsciousnessNet(X.shape[1])
        test_model.load_state_dict(base_model.state_dict())
        optimizer = optim.SGD(test_model.parameters(), lr=0.1) # Fast adaptation
        
        energy_spent = 0.0
        
        for step in range(50):
            optimizer.zero_grad()
            output, latent = test_model(X)
            loss = F.cross_entropy(output, y)
            
            # The "Frustration": If loss is high, we must adapt (spend energy)
            loss.backward()
            optimizer.step()
            
            # We measure how much the 'mind' had to change structure (Latent Variance)
            energy_spent += torch.std(latent).item()
            
        return energy_spent

# --- 3. RUN EXPERIMENT ---
def run_cystic_fibrosis_test():
    print("--- CYSTIC FIBROSIS CHALLENGE (CONSCIOUSNESS METRIC) ---")
    
    raw_seq = fetch_cftr_data()
    variants = induce_cf_mutations(raw_seq)
    
    detector = DiseaseDetector()
    
    print("1. Learning Healthy CFTR Structure...")
    healthy_model = detector.train_healthy_model(variants[0][1])
    print("   Structure Learned.")
    
    print("\n2. Measuring 'Cognitive Adaptation Energy' for Variants...")
    # N=50 Monte Carlo for stability
    n_trials = 50
    benign_scores = []
    pathogenic_scores = []
    
    for i in range(n_trials):
        # Benign
        e_benign = detector.measure_adaptation_energy(healthy_model, variants[1][1])
        benign_scores.append(e_benign)
        
        # Pathogenic
        e_pathogenic = detector.measure_adaptation_energy(healthy_model, variants[2][1])
        pathogenic_scores.append(e_pathogenic)
        
        if i % 10 == 0: print(".", end="", flush=True)

    print("\n\n--- RESULTS ---")
    avg_benign = np.mean(benign_scores)
    avg_path = np.mean(pathogenic_scores)
    
    print(f"Benign Mutation Energy:     {avg_benign:.4f}")
    print(f"Delta F508 (Lethal) Energy: {avg_path:.4f}")
    
    # Calculate Separation Signal
    signal = avg_path / avg_benign
    print(f"Diagnostic Signal Strength: {signal:.2f}x")

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(benign_scores, alpha=0.7, color='gold', label='Benign (Silent)', bins=15)
    plt.hist(pathogenic_scores, alpha=0.7, color='red', label='Delta F508 (Lethal)', bins=15)
    plt.xlabel("Adaptation Energy (Incompressibility)")
    plt.title(f"Cystic Fibrosis Detection\nSignal Strength: {signal:.2f}x")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_cystic_fibrosis_test()