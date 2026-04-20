#!/usr/bin/env python3
"""
Example 1: Basic CKKS Operations

This example demonstrates the fundamental CKKS homomorphic encryption operations:
- Key generation
- Encoding and encryption
- Homomorphic addition and multiplication
- Decoding and verification
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.context import FHEContext


def main():
    print("="*60)
    print("Example 1: Basic CKKS Operations")
    print("="*60)
    
    # =========================================================================
    # Setup: Create FHE context and generate keys
    # =========================================================================
    print("\n1. Setting up FHE context...")
    
    context = FHEContext.default(
        security_level=128,
        poly_degree=4096,  # Smaller for faster execution
        enable_bootstrapping=False,
    )
    
    context.setup_keys()
    
    print(f"   Context: {context}")
    print(f"   Slot count: {context.params.slot_count}")
    
    # =========================================================================
    # 2. Encrypt data
    # =========================================================================
    print("\n2. Encrypting data...")
    
    # Create some test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    
    print(f"   x = {x}")
    print(f"   y = {y}")
    
    # Encrypt
    ct_x = context.encrypt(x)
    ct_y = context.encrypt(y)
    
    print(f"   x encrypted: {ct_x}")
    print(f"   y encrypted: {ct_y}")
    
    # =========================================================================
    # 3. Homomorphic Addition
    # =========================================================================
    print("\n3. Homomorphic Addition: x + y")
    
    ct_sum = context.add(ct_x, ct_y)
    result_sum = context.decrypt(ct_sum)
    
    expected_sum = x + y
    print(f"   Expected: {expected_sum}")
    print(f"   Got:      {result_sum[:len(expected_sum)]}")
    print(f"   Error:    {np.abs(expected_sum - result_sum[:len(expected_sum)]).max():.6f}")
    
    # =========================================================================
    # 4. Homomorphic Multiplication
    # =========================================================================
    print("\n4. Homomorphic Multiplication: x * y")
    
    ct_prod = context.multiply(ct_x, ct_y)
    result_prod = context.decrypt(ct_prod)
    
    expected_prod = x * y
    print(f"   Expected: {expected_prod}")
    print(f"   Got:      {result_prod[:len(expected_prod)]}")
    print(f"   Error:    {np.abs(expected_prod - result_prod[:len(expected_prod)]).max():.6f}")
    
    # =========================================================================
    # 5. Scalar Multiplication
    # =========================================================================
    print("\n5. Scalar Multiplication: 2 * x")
    
    ct_scaled = context.multiply_scalar(ct_x, 2.0)
    result_scaled = context.decrypt(ct_scaled)
    
    expected_scaled = 2.0 * x
    print(f"   Expected: {expected_scaled}")
    print(f"   Got:      {result_scaled[:len(expected_scaled)]}")
    print(f"   Error:    {np.abs(expected_scaled - result_scaled[:len(expected_scaled)]).max():.6f}")
    
    # =========================================================================
    # 6. More Complex Expression: (x + y) * (x - y)
    # =========================================================================
    print("\n6. Complex Expression: (x + y) * (x - y) = x² - y²")
    
    ct_x_plus_y = context.add(ct_x, ct_y)
    ct_x_minus_y = context.sub(ct_x, ct_y)
    ct_result = context.multiply(ct_x_plus_y, ct_x_minus_y)
    
    result_complex = context.decrypt(ct_result)
    expected_complex = x**2 - y**2
    
    print(f"   Expected: {expected_complex}")
    print(f"   Got:      {result_complex[:len(expected_complex)]}")
    print(f"   Error:    {np.abs(expected_complex - result_complex[:len(expected_complex)]).max():.6f}")
    
    # =========================================================================
    # 7. Squaring
    # =========================================================================
    print("\n7. Squaring: x²")
    
    ct_squared = context.square(ct_x)
    result_squared = context.decrypt(ct_squared)
    
    expected_squared = x**2
    print(f"   Expected: {expected_squared}")
    print(f"   Got:      {result_squared[:len(expected_squared)]}")
    print(f"   Error:    {np.abs(expected_squared - result_squared[:len(expected_squared)]).max():.6f}")
    
    print("\n" + "="*60)
    print("Basic operations completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
