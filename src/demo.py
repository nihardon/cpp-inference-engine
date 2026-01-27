import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, "../build")
sys.path.append(build_dir)

import engine
import time
import random

# Helper to initialize weights
def random_tensor(rows, cols):
    t = engine.Tensor([rows, cols])

    t.fill(0.5) 
    return t

print("--- MINI-PYTORCH DEMO ---")

input_size = 64
hidden_size = 128
output_size = 10
batch_size = 32

print(f"Model: [Input {input_size}] -> [Hidden {hidden_size}] -> ReLU -> [Output {output_size}]")

# Initialize Weights & Inputs
# X: Batch of 32 inputs
X = random_tensor(batch_size, input_size) 
# W1: Input -> Hidden
W1 = random_tensor(input_size, hidden_size)
# W2: Hidden -> Output
W2 = random_tensor(hidden_size, output_size)

print("Running Forward Pass...")
start = time.time()

# Layer 1: Linear
# H = X * W1
H = engine.matmul(X, W1)

# Activation: ReLU
# H = ReLU(H)
engine.relu(H)

# Layer 2: Linear
# Y = H * W2
Y = engine.matmul(H, W2)

end = time.time()

print(f"Inference finished in {end - start:.6f} seconds.")
print(f"Output Shape: {Y.shape()}") # Should be [32, 10]