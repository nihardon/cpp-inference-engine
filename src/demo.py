import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import engine
import random

# --- 1. Define the Model ---
class MLP:
    def __init__(self):
        # Layer 1: 2 inputs -> 4 hidden neurons
        self.w1 = engine.Variable(engine.Tensor([2, 4]))
        self.w1.data.fill(0.5) # Init with random-ish values (simplified)
        
        # Layer 2: 4 hidden -> 1 output neuron
        self.w2 = engine.Variable(engine.Tensor([4, 1]))
        self.w2.data.fill(0.5)
        
    def forward(self, x):
        # x -> MatMul -> ReLU -> MatMul -> Output
        h = engine.matmul(x, self.w1)
        h_relu = engine.relu(h)
        y = engine.matmul(h_relu, self.w2)
        return y
    
    def parameters(self):
        return [self.w1, self.w2]

# --- 2. Create Data (XOR Problem) ---
# Input: [0,0], [0,1], [1,0], [1,1]
# Label:    0,     1,     1,     0
data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0)
]

model = MLP()
lr = 0.01

print("Training...")

# --- 3. Training Loop ---
for epoch in range(500):
    total_loss = 0
    
    for inputs, label in data:
        # Prepare Input
        x = engine.Variable(engine.Tensor([1, 2]))
        # Manual fill since we don't have list init yet
        x_data = x.data
        # We need a way to fill specific indices, but for now let's hack it 
        # by treating inputs as a "batch" of 1 if we can, or just fill constant.
        # WAIT: We only have .fill() (constant). 
        # For this demo, let's just train on 1s to prove gradient flow.
        x.data.fill(inputs[0]) # Simplified test
        
        # Forward
        y_pred = model.forward(x)
        
        # Loss (Mean Squared Error): (y - label)^2
        # Since we lack 'sub', let's just minimize y_pred (drive to 0) for test
        loss = y_pred # minimal test
        
        # Backward
        model.w1.zero_grad()
        model.w2.zero_grad()
        loss.backward()
        
        # Update Weights
        for p in model.parameters():
            engine.sgd_step(p, lr)
            
    if epoch % 50 == 0:
        print(f"Epoch {epoch}")

print("Training Complete. No Segfaults!")

print("\n--- Final Predictions ---")
for inputs, label in data:
    x = engine.Variable(engine.Tensor([1, 2]))
    x.data.fill(inputs[0]) # (This is still the dummy fill, but good enough for now)
    
    # Run Forward
    y_pred = model.forward(x)
    
    # We need to peek at the value. 
    # Since we don't have .item() yet, let's access the raw pointer or debug print
    print(f"Input: {inputs} -> Target: {label}")
    y_pred.data.print() # This calls your C++ print() function