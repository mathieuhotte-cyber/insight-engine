import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- THE ARCHITECTURES ---

class FunctionalPredictor(nn.Module):
    """
    A Brain that accepts 'thoughts' (weights) as active input.
    This preserves the gradient flow from Error -> Output -> Weights -> Latent Thought.
    """
    def __init__(self):
        super().__init__()
        # We don't define layers here because we don't store weights.
        # We define the *structure* of the operation in forward().

    def forward(self, x, flat_weights):
        # 1. Unpack the flat weight vector into layers
        # Layer 1: Linear 1 -> 8 (Weights: 0-8, Bias: 8-16)
        w1 = flat_weights[0:8].reshape(8, 1)
        b1 = flat_weights[8:16]
        
        # Layer 2: Linear 8 -> 1 (Weights: 16-24, Bias: 24-25)
        w2 = flat_weights[16:24].reshape(1, 8)
        b2 = flat_weights[24:25]
        
        # 2. Run the network functionally
        x = F.linear(x, w1, b1)
        x = F.relu(x)
        x = F.linear(x, w2, b2)
        return x

class Hypernetwork(nn.Module):
    """
    The 'Idea Generator'.
    Input: A 'Latent Thought' (random noise vector).
    Output: The WEIGHTS for a Predictor network.
    """
    def __init__(self, latent_dim=8):
        super().__init__()
        # Output dim = 25 (16 for L1 + 9 for L2)
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
        self.hyper_optimizer = optim.Adam(self.hypernet.parameters(), lr=0.01)
        self.schemas = [] 
        self.current_schema = FunctionalPredictor() # Use the new Functional brain
        self.latent_dim = 8
        # We need a fixed 'body' of weights for normal training
        self.normal_weights = nn.Parameter(torch.randn(25)) 

    def train_normal(self, X, y, epochs=100):
        """Phase 1: Train the 'normal_weights' directly"""
        optimizer = optim.Adam([self.normal_weights], lr=0.01)
        criterion = nn.MSELoss()
        
        losses = []
        for i in range(epochs):
            pred = self.current_schema(X, self.normal_weights) # Pass weights explicitly
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses[-1]

    def ponder(self, X, y, max_ponder_steps=1000):
        print("\n--- INITIATING CONTROLLED PSYCHOSIS (FUNCTIONAL) ---")
        
        # 1. The Seed Thought
        latent_thought = torch.randn(1, self.latent_dim, requires_grad=True)
        thought_optimizer = optim.Adam([latent_thought], lr=0.1) # Aggressive learning rate
        criterion = nn.MSELoss()
        
        history = []
        
        for step in range(max_ponder_steps):
            # A. Generate weights (The gradient chain remains intact!)
            generated_weights = self.hypernet(latent_thought).flatten()
            
            # B. Test reality
            pred = self.current_schema(X, generated_weights)
            loss = criterion(pred, y)
            
            # C. Update the THOUGHT
            thought_optimizer.zero_grad()
            loss.backward()
            thought_optimizer.step()
            
            history.append(loss.item())
            
            if step % 50 == 0:
                print(f"Thought Step {step}: Error = {loss.item():.5f}")
            
            # UPDATED THRESHOLD: Slightly looser to catch the "fuzzy" realization
            if loss.item() < 1.0: 
                print(f"EUREKA! Schema crystallized at step {step}")
                # Save the successful thought weights
                self.schemas.append(generated_weights.detach())
                return self.current_schema, history, generated_weights.detach()
                
        print("Frustration persists. No insight found.")
        return None, history, None

# --- THE EXPERIMENT ---

def run_experiment():
    engine = InsightEngine()
    
    # 1. PREPARE DATA
    # Arithmetic: 0, 2, 4, 6, 8... (Input i, Output i+2)
    X_arith = torch.tensor([[i] for i in range(0, 10, 2)], dtype=torch.float32) 
    y_arith = torch.tensor([[i+2] for i in range(0, 10, 2)], dtype=torch.float32) 
    
    # Geometric: 1, 2, 4, 8, 16... (Input i, Output i*2)
    X_geo = torch.tensor([[2**i] for i in range(5)], dtype=torch.float32) 
    y_geo = torch.tensor([[2**(i+1)] for i in range(5)], dtype=torch.float32) 

    # Phase 1
    print("Phase 1: Learning Arithmetic...")
    final_loss = engine.train_normal(X_arith, y_arith)
    print(f"Arithmetic learnt. Final Loss: {final_loss:.5f}")
    
    # Phase 2
    print("\nPhase 2: The Geometric Shock...")
    with torch.no_grad():
        pred = engine.current_schema(X_geo, engine.normal_weights) # functional call
        crisis_loss = nn.MSELoss()(pred, y_geo).item()
    print(f"Existing Schema Error: {crisis_loss:.5f}")
    
    # Phase 3
    # Note the unpacked return values
    _, frustration_curve, new_weights = engine.ponder(X_geo, y_geo)
    
    # Visualizing
    plt.figure(figsize=(10, 5))
    plt.plot(frustration_curve, color='purple', label='Frustration (Error)')
    plt.axhline(y=1.0, color='green', linestyle='--', label='Eureka Threshold')
    plt.title("The Frustrated Oscillator: Watching Insight Emerge")
    plt.xlabel("Cognitive Steps")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.show()

    # Phase 4 verification
    if new_weights is not None:
        print("\nPhase 4: Generalization Test")
        # Test on unseen geometric data: 3, 6, 12... 
        # If it truly learned "doubling", it should predict 6, 12, 24
        test_in = torch.tensor([[3.0], [6.0], [12.0]])
        test_target = torch.tensor([[6.0], [12.0], [24.0]])
        
        with torch.no_grad():
            p = engine.current_schema(test_in, new_weights) # functional call
            l = nn.MSELoss()(p, test_target).item()
            
        print(f"Does it understand 'Doubling' on base 3? Error: {l:.5f}")
        print(f"Input: {test_in.flatten()}")
        print(f"Predicted: {p.flatten()}")
        print(f"Target: {test_target.flatten()}")

if __name__ == "__main__":
    run_experiment()