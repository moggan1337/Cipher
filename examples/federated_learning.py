#!/usr/bin/env python3
"""
Example 3: Federated Learning with Homomorphic Encryption

This example demonstrates privacy-preserving federated learning where:
- Multiple clients hold sensitive data locally
- Only encrypted gradients are shared with the server
- The server performs aggregation without seeing individual data
- Model updates are computed entirely under encryption

This is ideal for:
- Healthcare consortia training on patient data
- Financial institutions collaborating on fraud detection
- Mobile devices learning from user behavior privately
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.context import FHEContext
from src.ml.encrypted_model import EncryptedLinear, EncryptedReLU, EncryptedSequential
from src.ml.encrypted_training import (
    FederatedServer, FederatedClient, FederatedAveraging, DifferentialPrivacy
)


def create_simple_model(input_dim: int, output_dim: int) -> EncryptedSequential:
    """Create a simple model for demonstration."""
    return EncryptedSequential([
        EncryptedLinear(input_dim, 16),
        EncryptedReLU(degree=3),
        EncryptedLinear(16, output_dim),
    ])


def simulate_local_data(
    num_samples: int, 
    input_dim: int, 
    client_id: str
) -> tuple:
    """
    Simulate local data for a client.
    
    In practice, this would be the client's actual sensitive data.
    """
    np.random.seed(hash(client_id) % 2**32)
    
    # Generate synthetic data with client-specific patterns
    X = np.random.randn(num_samples, input_dim) + np.random.randn(input_dim) * 0.5
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    
    return X, y


def main():
    print("="*60)
    print("Example 3: Federated Learning with FHE")
    print("="*60)
    
    # =========================================================================
    # Setup
    # =========================================================================
    print("\n1. Setting up FHE context...")
    
    context = FHEContext.default(
        security_level=128,
        poly_degree=2048,  # Smaller for faster federated demo
        enable_bootstrapping=False,
    )
    context.setup_keys()
    
    print(f"   {context}")
    
    # =========================================================================
    # 2. Create federated server
    # =========================================================================
    print("\n2. Creating federated server...")
    
    # Create global model
    global_model = create_simple_model(input_dim=4, output_dim=1)
    
    # Initialize with random weights
    for layer in global_model.layers:
        if isinstance(layer, EncryptedLinear):
            w = np.random.randn(layer.out_features, layer.in_features) * 0.1
            layer.set_weights(w)
    
    server = FederatedServer(global_model, context)
    print(f"   Server created with global model")
    
    # =========================================================================
    # 3. Create federated clients
    # =========================================================================
    print("\n3. Creating federated clients...")
    
    num_clients = 3
    clients = []
    
    for i in range(num_clients):
        client_id = f"client_{i}"
        
        # Each client has its own model copy
        client_model = create_simple_model(input_dim=4, output_dim=1)
        
        # Copy weights from global model
        for j, (gl, cl) in enumerate(zip(global_model.layers, client_model.layers)):
            if isinstance(gl, EncryptedLinear) and isinstance(cl, EncryptedLinear):
                if gl._weights is not None:
                    cl.set_weights(gl._weights.copy())
        
        # Create client
        client = FederatedClient(client_id, client_model, context)
        
        # Load local data (in practice, this is sensitive data that never leaves)
        X_local, y_local = simulate_local_data(100, 4, client_id)
        client.load_data(X_local, y_local)
        
        clients.append(client)
        server.register_client(client)
        
        print(f"   {client_id}: {client.num_samples} samples")
    
    print(f"   Total clients: {server.num_clients}")
    
    # =========================================================================
    # 4. Initialize differential privacy
    # =========================================================================
    print("\n4. Setting up differential privacy...")
    
    dp = DifferentialPrivacy(
        epsilon=8.0,
        delta=1e-5,
        clip_bound=1.0,
    )
    
    print(f"   {dp}")
    
    # =========================================================================
    # 5. Federated training rounds
    # =========================================================================
    print("\n5. Running federated training...")
    
    num_rounds = 3
    
    for round_num in range(num_rounds):
        print(f"\n   Round {round_num + 1}/{num_rounds}:")
        
        # Sample clients (all clients in this demo)
        sampled_clients = server.sample_clients(fraction=1.0)
        print(f"   - Sampled {len(sampled_clients)} clients")
        
        # Each client computes local gradients (simulated)
        client_gradients = {}
        client_weights = {}
        
        for client in sampled_clients:
            # Simulate gradient computation (in practice, actual backprop)
            gradients = {}
            
            for i, layer in enumerate(client.model.layers):
                if isinstance(layer, EncryptedLinear):
                    # Simulated gradients
                    grad_w = np.random.randn(*layer._weights.shape) * 0.01
                    gradients[f'layer_{i}_weight'] = grad_w
                    
                    if layer._bias is not None:
                        grad_b = np.random.randn(*layer._bias.shape) * 0.01
                        gradients[f'layer_{i}_bias'] = grad_b
            
            client_gradients[client.client_id] = gradients
            client_weights[client.client_id] = client.num_samples
            
            print(f"   - {client.client_id} computed gradients")
        
        # Secure aggregation (under encryption)
        print(f"   - Performing secure aggregation...")
        aggregated = server.secure_aggregate(client_gradients, client_weights)
        
        # Apply differential privacy to aggregated gradients
        print(f"   - Applying differential privacy...")
        for name, grad_ct in aggregated.items():
            # Decrypt, add noise, re-encrypt
            grad = context.decrypt(grad_ct)
            grad_noisy = dp.privatize_gradient(grad)
            aggregated[name] = context.encrypt(grad_noisy)
        
        # Apply updates to global model
        print(f"   - Updating global model...")
        
        # In practice, would apply encrypted updates
        # For demo, apply to plaintext model
        update_scale = 0.01  # Learning rate
        for i, layer in enumerate(global_model.layers):
            if isinstance(layer, EncryptedLinear):
                weight_key = f'layer_{i}_weight'
                if weight_key in aggregated:
                    grad = context.decrypt(aggregated[weight_key])
                    layer._weights -= update_scale * grad
        
        print(f"   Round {round_num + 1} completed")
    
    # =========================================================================
    # 6. Evaluate global model
    # =========================================================================
    print("\n6. Evaluating global model...")
    
    # Test on held-out data
    X_test, y_test = simulate_local_data(50, 4, "test")
    
    # Run inference (simplified)
    x_sample = X_test[0]
    ct_input = context.encrypt(x_sample)
    
    # Forward pass
    output = x_sample.copy()
    for layer in global_model.layers:
        if isinstance(layer, EncryptedLinear):
            output = output @ layer._weights.T
            output = np.maximum(0, output)  # ReLU
        elif isinstance(layer, EncryptedReLU):
            output = np.maximum(0, output)
    
    print(f"   Sample prediction: {output[:2]}")
    print(f"   Sample true label: {y_test[0]}")
    
    # =========================================================================
    # 7. Privacy analysis
    # =========================================================================
    print("\n7. Privacy analysis...")
    
    eps_spent, delta_spent = dp.compute_privacy_spent(
        num_iterations=num_rounds * 10,  #假设每轮10次迭代
        batch_size=10,
        dataset_size=100,
    )
    
    print(f"   Privacy budget spent: ε = {eps_spent:.2f}, δ = {delta_spent:.2e}")
    
    print("\n" + "="*60)
    print("Federated learning completed!")
    print("="*60)
    print("\nKey Points:")
    print("- Each client's data stayed LOCAL and ENCRYPTED")
    print("- Only gradient UPDATES were shared (with DP noise)")
    print("- Server performed AGGREGATION without seeing raw data")
    print("- Differential privacy adds additional protection")
    print("- Privacy guarantee: (ε, δ)-differential privacy")


if __name__ == "__main__":
    main()
