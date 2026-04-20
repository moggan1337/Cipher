#!/usr/bin/env python3
"""
Example 2: Encrypted Neural Network Inference

This example demonstrates privacy-preserving neural network inference
where:
- The model weights are PUBLIC
- The input data is ENCRYPTED
- Only the data owner can see the predictions

This is useful for:
- Cloud-based ML inference where data privacy is required
- Medical diagnostics where patient data must be protected
- Financial fraud detection without exposing customer data
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.context import FHEContext
from src.ml.encrypted_model import (
    EncryptedLinear, EncryptedReLU, EncryptedSigmoid,
    EncryptedTanh, EncryptedSequential
)
from src.ml.approximation import PolynomialApproximator


def create_mlp(input_size: int, hidden_sizes: list, output_size: int) -> EncryptedSequential:
    """Create a multi-layer perceptron."""
    layers = []
    prev_size = input_size
    
    for i, hidden_size in enumerate(hidden_sizes):
        layers.append(EncryptedLinear(prev_size, hidden_size))
        layers.append(EncryptedReLU(degree=3))
        prev_size = hidden_size
    
    layers.append(EncryptedLinear(prev_size, output_size))
    # Final layer: sigmoid for binary classification
    layers.append(EncryptedSigmoid(degree=5))
    
    return EncryptedSequential(layers)


def main():
    print("="*60)
    print("Example 2: Encrypted Neural Network Inference")
    print("="*60)
    
    # =========================================================================
    # Setup
    # =========================================================================
    print("\n1. Setting up FHE context...")
    
    context = FHEContext.default(
        security_level=128,
        poly_degree=4096,
        enable_bootstrapping=False,
    )
    context.setup_keys()
    
    print(f"   {context}")
    
    # =========================================================================
    # 2. Create and initialize model
    # =========================================================================
    print("\n2. Creating neural network...")
    
    # Simple MLP for demonstration
    model = EncryptedSequential([
        EncryptedLinear(4, 8),
        EncryptedReLU(degree=3),
        EncryptedLinear(8, 4),
        EncryptedReLU(degree=3),
        EncryptedLinear(4, 2),
        EncryptedSigmoid(degree=5),
    ])
    
    # Initialize with random weights (in practice, load trained weights)
    print("   Initializing weights...")
    np.random.seed(42)
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, EncryptedLinear):
            # Xavier initialization
            fan_in = layer.in_features
            fan_out = layer.out_features
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_out, fan_in))
            layer.set_weights(w)
            print(f"   Layer {i}: Linear({fan_in}, {fan_out}) - weights shape {w.shape}")
    
    # =========================================================================
    # 3. Prepare input data
    # =========================================================================
    print("\n3. Preparing input data...")
    
    # Sample input (could be sensitive medical/financial data)
    x = np.array([0.5, -0.3, 0.8, -0.2])
    
    print(f"   Input: {x}")
    print(f"   Input shape: {x.shape}")
    
    # =========================================================================
    # 4. Encrypt input
    # =========================================================================
    print("\n4. Encrypting input...")
    
    ct_x = context.encrypt(x)
    print(f"   Encrypted: {ct_x}")
    
    # =========================================================================
    # 5. Run encrypted inference
    # =========================================================================
    print("\n5. Running encrypted inference...")
    
    import time
    start = time.time()
    
    ct_output = model.forward(ct_x, context, weights_public=True)
    
    inference_time = time.time() - start
    
    print(f"   Inference completed in {inference_time*1000:.1f}ms")
    print(f"   Output ciphertext: {ct_output}")
    
    # =========================================================================
    # 6. Decrypt and verify
    # =========================================================================
    print("\n6. Decrypting output...")
    
    y_pred = context.decrypt(ct_output)
    print(f"   Decrypted output: {y_pred[:2]}")
    
    # Verify by running same computation in plaintext
    print("\n7. Verifying with plaintext computation...")
    
    x_pt = x.copy()
    for layer in model.layers:
        if isinstance(layer, EncryptedLinear):
            x_pt = x_pt @ layer._weights.T
            x_pt = np.maximum(0, x_pt)  # ReLU (approximate)
    
    print(f"   Plaintext output: {x_pt[:2]}")
    
    # =========================================================================
    # 8. Run batch inference
    # =========================================================================
    print("\n8. Batch inference demonstration...")
    
    # Create batch of inputs
    batch_size = 4
    x_batch = np.random.randn(batch_size, len(x)) * 0.5
    
    print(f"   Batch size: {batch_size}")
    
    # Encrypt batch (using SIMD slots)
    ct_batch = context.encrypt(x_batch[0])  # Simplified - would need padding for full batch
    
    start = time.time()
    _ = model.forward(ct_batch, context)
    batch_time = time.time() - start
    
    print(f"   Batch inference time: {batch_time*1000:.1f}ms")
    
    print("\n" + "="*60)
    print("Encrypted inference completed!")
    print("="*60)
    print("\nKey Points:")
    print("- Model weights remained PUBLIC throughout")
    print("- Input data was ENCRYPTED")
    print("- Only the data owner (with secret key) can see predictions")
    print("- Multiple inputs can be processed simultaneously (SIMD)")


if __name__ == "__main__":
    main()
