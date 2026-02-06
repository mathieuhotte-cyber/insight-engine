import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- HELPER: PRIME GENERATION ---
def get_prime_gaps(n_primes):
    """Generates the first n prime gaps."""
    primes = []
    candidate = 2
    while len(primes) < n_primes + 1:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    return gaps

# --- THE ARCHITECTURES ---

class FunctionalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, flat_weights):
        # Layer 1: 1 -> 8
        w1 = flat_weights[0:8].reshape(8, 1)
        b1 = flat_weights[8:16]
        # Layer 2: 8 -> 1
        w2 = flat_weights[16:24].reshape(1, 8)
        b2 = flat_weights[24:25]
        
        x = F.linear(x, w1, b1)
        x = F.relu(x)
        x = F.linear(x, w2, b2)
        return x

class Hypernetwork(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.output_dim = 25 
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        )

    def forward(self, z):
        return self.generator(z)

# --- THE ENGINE ---

class InsightEngine:
    def __init__(self):
        self.hypernet = Hypernetwork()
        self.schemas = [] 
        self.current_schema = FunctionalPredictor()
        self.latent_dim = 8
        self.normal_weights = nn.Parameter(torch.randn(25)) 

    def train_normal(self, X, y, epochs=500):
        optimizer = optim.Adam([self.normal_weights], lr=0.01)
        criterion = nn.MSELoss()
        losses = []
        for i in range(epochs):
            pred = self.current_schema(X, self.normal_weights)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses[-1]

    def ponder(self, X, y, max_ponder_steps=2000):
        print("\n--- INITIATING PRIME HUNT (CONTROLLED PSYCHOSIS) ---")
        latent_thought = torch.randn(1, self.latent_dim, requires_grad=True)
        thought_optimizer = optim.Adam([latent_thought], lr=0.05)
        criterion = nn.MSELoss()
        
        history = []
        
        for step in range(max_ponder_steps):
            generated_weights = self.hypernet(latent_thought).flatten()
            pred = self.current_schema(X, generated_weights)
            loss = criterion(pred, y)
            
            thought_optimizer.zero_grad()
            loss.backward()
            thought_optimizer.step()
            
            history.append(loss.item())
            
            if step % 100 == 0:
                print(f"Deep Thought Step {step}: Error = {loss.item():.5f}")
            
            if loss.item() < 0.5: 
                 print(f"ANOMALY DETECTED! Low error state found at step {step}")
                 return self.current_schema, history, generated_weights.detach()

        print("Maximum ponder steps reached. No perfect schema found (Expected for Primes).")
        return None, history, generated_weights.detach()

# --- THE EXPERIMENT ---

def run_prime_hunt():
    engine = InsightEngine()
    print("Generating Prime Gaps...")
    raw_gaps = get_prime_gaps(150)
    
    X_data = torch.tensor([[g] for g in raw_gaps[:-1]], dtype=torch.float32)
    y_data = torch.tensor([[g] for g in raw_gaps[1:]], dtype=torch.float32)
    
    X_train = X_data[:100]
    y_train = y_data[:100]
    X_test = X_data[100:]
    y_test = y_data[100:]
    
    baseline_error = torch.var(y_train).item()
    print(f"Baseline Variance (Random Guessing): {baseline_error:.5f}")

    print("Phase 1: Attempting Standard Logic...")
    std_loss = engine.train_normal(X_train, y_train)
    print(f"Standard Training Final Loss: {std_loss:.5f}")
    
    _, frustration_curve, best_weights = engine.ponder(X_train, y_train)
    
    plt.figure(figsize=(12, 6))
    plt.plot(frustration_curve, color='purple', label='Insight Struggle')
    plt.axhline(y=baseline_error, color='gray', linestyle='--', label='Baseline (Random)')
    plt.axhline(y=std_loss, color='red', linestyle=':', label='Standard NN')
    plt.title("Hunting for Order in Prime Gaps")
    plt.xlabel("Cognitive Steps")
    plt.ylabel("Prediction Error (MSE)")
    plt.legend()
    plt.show()

    if best_weights is not None:
        print("\n--- PHASE 3: THE UNSEEN (Generalization) ---")
        with torch.no_grad():
            pred_test = engine.current_schema(X_test, best_weights)
            test_loss = nn.MSELoss()(pred_test, y_test).item()
            
        print(f"Test Set (Future Gaps) Error: {test_loss:.5f}")
        print(f"Did we beat Baseline? {'YES' if test_loss < baseline_error else 'NO'}")
        
        print("\nSample Predictions (Input -> Predicted vs Actual):")
        for i in range(5):
            print(f"Gap {int(X_test[i].item())} -> {pred_test[i].item():.2f} (Actual: {int(y_test[i].item())})")

if __name__ == "__main__":
    run_prime_hunt()