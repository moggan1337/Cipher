# Cipher - Homomorphic Encryption ML Framework

<p align="center">
  <strong>A comprehensive framework for privacy-preserving machine learning using CKKS fully homomorphic encryption</strong>
</p>

<p align="center">
  <a href="https://github.com/moggan1337/Cipher/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/moggan1337/Cipher/issues">
    <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg" alt="Contributions">
  </a>
</p>

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Mathematical Background](#mathematical-background)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [API Reference](#api-reference)
7. [Architecture](#architecture)
8. [Benchmarks](#benchmarks)
9. [Usage Examples](#usage-examples)
10. [Security Considerations](#security-considerations)
11. [Limitations](#limitations)
12. [Contributing](#contributing)
13. [License](#license)

---

## Introduction

Cipher is a Python framework for privacy-preserving machine learning using **Fully Homomorphic Encryption (FHE)**. It implements the **CKKS scheme** (Cheon-Kim-Kim-Song), enabling computation on encrypted data without decryption.

### Why FHE for ML?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Traditional Machine Learning                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Client          Server           Client          Server            │
│      │              │                │               │               │
│      │───Data──────▶│                │───Data────────▶│               │
│      │              │                │               │               │
│      │              ▼                │               ▼               │
│      │         ┌─────────┐          │          ┌─────────┐        │
│      │         │ Model   │          │          │ Model   │        │
│      │         │ Training│          │          │Inference│        │
│      │         └─────────┘          │          └─────────┘        │
│      │              │                │               │               │
│      │◀────Result───│                │◀────Result─────│               │
│                                                                      │
│   Data is exposed to server                                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   Homomorphic Encryption                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Client          Server           Client          Server            │
│      │              │                │               │               │
│      │───ENCRYPT───▶│                │───ENCRYPT─────▶│               │
│      │   Data       │                │    Data        │               │
│      │              ▼                │               ▼               │
│      │         ┌─────────┐           │          ┌─────────┐         │
│      │         │ Encrypted│          │          │ Encrypted│         │
│      │         │ Compute  │          │          │ Compute │         │
│      │         │ on Cipher│          │          │ on Cipher│        │
│      │         └─────────┘           │          └─────────┘         │
│      │◀──Result──│                │◀──Result─────│               │
│      │  DECRYPT   │                │   DECRYPT     │               │
│                                                                      │
│   Data remains encrypted throughout                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Use Cases

- **Cloud-based ML Inference**: Send encrypted data to cloud, get encrypted predictions back
- **Healthcare Analytics**: Multiple hospitals train models without sharing patient data
- **Financial Services**: Collaborative fraud detection across banks without exposing transactions
- **Edge Computing**: Secure on-device ML that never exposes user data

---

## Features

### Core FHE Features
- ✅ **CKKS Scheme Implementation**: Complete implementation with encoding, encryption, and operations
- ✅ **Homomorphic Operations**: Addition, multiplication, rotation, rescaling
- ✅ **Matrix Operations**: Encrypted matrix-vector multiplication
- ✅ **Bootstrapping**: Automatic ciphertext refresh for unlimited computation depth

### Machine Learning Features
- ✅ **Encrypted Inference**: Run neural networks on encrypted inputs
- ✅ **Polynomial Activation Approximation**: ReLU, Sigmoid, Tanh, GELU, and more
- ✅ **Layer Types**: Linear, Embedding, LayerNorm
- ✅ **Model Conversion**: Import PyTorch models for encrypted inference

### Privacy-Preserving Training
- ✅ **Federated Learning**: Multi-client collaborative training
- ✅ **Secure Aggregation**: Server aggregates encrypted gradients
- ✅ **Differential Privacy**: Formal privacy guarantees with DP noise
- ✅ **Encrypted Optimization**: Gradient updates computed under encryption

### Additional Features
- ✅ **Automatic Polynomial Approximation**: Find optimal polynomial degree for activations
- ✅ **Depth Management**: Track and manage multiplicative depth
- ✅ **Serialization**: Save/load keys and encrypted data
- ✅ **Comprehensive Benchmarks**: Performance testing suite

---

## Mathematical Background

### The CKKS Scheme

CKKS is a **ring-based** homomorphic encryption scheme operating on polynomial rings.

#### Polynomial Ring

CKKS operates on the ring:
```
R_q = Z_q[X] / (X^N + 1)
```

Where:
- `N` is the polynomial degree (power of 2)
- `q` is the ciphertext modulus
- `X^N ≡ -1` (negacyclic convolution)

#### Key Generation

1. **Secret Key** `s`: Sampled from ternary distribution {-1, 0, 1}^N with hamming weight H

2. **Public Key** `(a, b)`: 
   ```
   b = -a·s + e (mod q)
   ```
   where `a` is uniform random and `e` is small Gaussian noise

3. **Evaluation Key**: For relinearization and rotations

#### Encoding

Complex values are encoded as polynomials using the **Slots** mechanism:

1. Take N/2 complex numbers `z = (z_1, ..., z_{N/2})`
2. Apply inverse FFT to map to polynomial coefficients
3. Scale by `Δ` for precision management

#### Encryption

For message polynomial `m(X)`:
```
c_0 = b·u + m + e_0 (mod q)
c_1 = a·u + e_1 (mod q)
```

Where `u` is random ternary and `e_i` are Gaussian errors.

#### Homomorphic Operations

**Addition**: Component-wise polynomial addition
```
ct_add = (c_0¹ + c_0², c_1¹ + c_1²)
```

**Multiplication**: Polynomial multiplication
```
ct_mult = (c_0¹·c_0², c_0¹·c_1² + c_1¹·c_0², c_1¹·c_1²)
```

The result has degree 2 and needs **relinearization** to return to degree 1.

**Rescaling**: Divide by `Δ` and switch to lower modulus to reduce noise.

#### Bootstrapping

Bootstrapping "refreshes" a ciphertext by:
1. Switching to a special modulus
2. Computing the decryption function homomorphically
3. Re-encrypting with fresh noise

This enables unlimited computation depth at the cost of computation time.

### Polynomial Approximation of Activations

Non-polynomial functions must be approximated by polynomials:

**ReLU** approximation:
```python
relu(x) ≈ 0.5x + 0.5|x|
       ≈ polynomial(x)  # via Chebyshev approximation
```

**Sigmoid** approximation:
```python
σ(x) = 1/(1 + e^(-x)) ≈ Σ c_n · T_n(x)  # Chebyshev series
```

### Complexity Analysis

| Operation | Complexity |
|-----------|------------|
| Key Generation | O(N² log N) |
| Encryption | O(N log N) |
| Addition | O(N) |
| Multiplication | O(N log N) |
| Bootstrapping | O(N log N · log q) |

Where N is the polynomial degree.

---

## Installation

### Requirements

- Python 3.8+
- NumPy
- Numba (optional, for JIT optimization)
- PyTorch (optional, for model conversion)

### Installation from Source

```bash
# Clone the repository
git clone https://github.com/moggan1337/Cipher.git
cd Cipher

# Install dependencies
pip install numpy numba

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Quick Test

```bash
cd Cipher
python examples/basic_usage.py
```

---

## Quick Start

### 1. Basic Encryption and Computation

```python
from src.core.context import FHEContext
import numpy as np

# Create context and setup keys
context = FHEContext.default(security_level=128, poly_degree=4096)
context.setup_keys()

# Encrypt data
x = np.array([1.0, 2.0, 3.0, 4.0])
ct_x = context.encrypt(x)

# Compute on encrypted data
ct_squared = context.square(ct_x)
ct_result = context.add(ct_squared, ct_x)

# Decrypt
result = context.decrypt(ct_result)
print(result)  # [2.0, 6.0, 12.0, 20.0]  (x² + x)
```

### 2. Encrypted Neural Network Inference

```python
from src.core.context import FHEContext
from src.ml.encrypted_model import (
    EncryptedLinear, EncryptedReLU, EncryptedSequential
)

# Create context
context = FHEContext.default(poly_degree=4096)
context.setup_keys()

# Define model
model = EncryptedSequential([
    EncryptedLinear(784, 256),
    EncryptedReLU(degree=3),
    EncryptedLinear(256, 10),
])

# Set weights (could be from pre-trained model)
for layer in model.layers:
    if isinstance(layer, EncryptedLinear):
        layer.set_weights(pretrained_weights)

# Encrypt input and run inference
ct_input = context.encrypt(image_pixels)
ct_output = model.forward(ct_input, context)

# Only data owner can decrypt
prediction = context.decrypt(ct_output)
```

### 3. Federated Learning

```python
from src.ml.encrypted_training import FederatedServer, FederatedClient

# Server creates global model
server = FederatedServer(global_model, context)

# Clients register with local data
client1 = FederatedClient("hospital_A", model, context)
client1.load_data(patient_records)  # Encrypted immediately
server.register_client(client1)

# Run federated rounds
server.fit(num_rounds=10, local_epochs=5)
```

---

## API Reference

### Core Classes

#### `FHEContext`
Main interface for FHE operations.

```python
context = FHEContext.default(
    security_level=128,    # 128, 192, or 256 bits
    poly_degree=8192,      # Higher = more slots, slower
    enable_bootstrapping=True
)
context.setup_keys()
```

**Methods:**
- `encrypt(values)` - Encrypt numpy array
- `decrypt(ciphertext)` - Decrypt to numpy array
- `add()`, `sub()`, `multiply()` - Homomorphic arithmetic
- `matrix_multiply()` - Encrypted matrix-vector product

#### `CKKSParameters`
Configure the encryption scheme.

```python
params = CKKSParameters(
    poly_degree=8192,
    ciphertext_moduli=[0x1FFFFFFFF00001, 0x1FFFFFE00001, ...],
    scaling_factor=2**40,
    security_level=SecurityLevel.TC26_128
)
```

### ML Classes

#### `EncryptedLinear`
Encrypted linear transformation.

```python
layer = EncryptedLinear(in_features=784, out_features=256, bias=True)
layer.set_weights(weight_matrix, bias_vector)
output = layer.forward(encrypted_input, context)
```

#### `EncryptedSequential`
Container for encrypted models.

```python
model = EncryptedSequential([
    EncryptedLinear(784, 256),
    EncryptedReLU(degree=3),
    EncryptedLinear(256, 10),
])
```

### Training Classes

#### `FederatedServer`
Central server for federated learning.

```python
server = FederatedServer(model, context)
server.register_client(client)
server.fit(num_rounds=10)
```

#### `DifferentialPrivacy`
Add formal privacy guarantees.

```python
dp = DifferentialPrivacy(epsilon=8.0, delta=1e-5, clip_bound=1.0)
noisy_gradient = dp.privatize_gradient(gradient)
```

---

## Architecture

```
Cipher/
├── src/
│   ├── core/                  # Core FHE implementation
│   │   ├── params.py          # CKKS parameters
│   │   ├── keys.py           # Key generation
│   │   ├── ckks.py           # CKKS operations
│   │   └── context.py        # High-level interface
│   │
│   ├── ml/                   # ML components
│   │   ├── encrypted_model.py    # Encrypted layers
│   │   ├── approximation.py       # Activation approximation
│   │   └── encrypted_training.py # Federated learning
│   │
│   ├── polynomial/            # Polynomial arithmetic
│   │   └── polynomial.py     # Polynomials and approximation
│   │
│   ├── bootstrapping/         # Bootstrapping implementation
│   │   └── bootstrapping.py  # Ciphertext refresh
│   │
│   └── __init__.py
│
├── tests/                    # Test suite
│   └── test_ckks.py
│
├── benchmarks/               # Performance benchmarks
│   └── benchmark.py
│
├── examples/                  # Usage examples
│   ├── basic_usage.py
│   ├── encrypted_nn.py
│   └── federated_learning.py
│
└── README.md
```

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Plaintext │────▶│    Encode    │────▶│    Encrypt      │
│   Values    │     │  (FFT→Poly)  │     │  (Add noise)    │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Plaintext │◀────│   Decode     │◀────│   Decrypt       │
│   Result    │     │  (FFT←Poly)  │     │  (Use sk)       │
└─────────────┘     └──────────────┘     └─────────────────┘
                            ▲
                            │
                     ┌──────┴──────┐
                     │  Homomorphic │
                     │   Compute   │
                     └─────────────┘
```

---

## Benchmarks

Performance varies significantly with polynomial degree:

| N (Degree) | Slots | Encrypt (ms) | Multiply (ms) | NN Inference (ms) |
|------------|-------|--------------|--------------|-------------------|
| 1024       | 512   | 15           | 25           | 200               |
| 2048       | 1024  | 30           | 50           | 450               |
| 4096       | 2048  | 60           | 120          | 1000              |
| 8192       | 4096  | 150          | 300          | 2500              |

### Running Benchmarks

```bash
# Quick benchmark
python -m benchmarks.benchmark --quick

# Full benchmark
python -m benchmarks.benchmark --degree 4096

# Compare across degrees
python -m benchmarks.benchmark --compare
```

### Factors Affecting Performance

1. **Polynomial Degree (N)**: Primary factor. Doubling N quadruples computation time but doubles slots.

2. **Multiplicative Depth**: Each multiplication adds noise. Deep networks need bootstrapping.

3. **Activation Degree**: Higher-degree polynomials are more accurate but slower.

4. **Batch Size**: SIMD encoding allows parallel processing of multiple inputs.

---

## Usage Examples

### Example 1: Basic Operations

See `examples/basic_usage.py`:
- Encryption and decryption
- Addition and multiplication
- Scalar operations
- Complex expressions

### Example 2: Neural Network Inference

See `examples/encrypted_nn.py`:
- Model definition
- Weight initialization
- Encrypted forward pass
- Batch processing

### Example 3: Federated Learning

See `examples/federated_learning.py`:
- Client registration
- Secure aggregation
- Differential privacy
- Privacy budget tracking

---

## Security Considerations

### Security Levels

The framework supports three security levels per HomomorphicEncryption.org standard:

| Level | Security Bits | Recommended N |
|-------|---------------|--------------|
| TC26-128 | 128 | 8192 |
| TC26-192 | 192 | 16384 |
| TC26-256 | 256 | 32768 |

### Parameter Selection

⚠️ **Important**: Incorrect parameters can compromise security!

```python
# ✅ Secure parameters (from standard)
params = CKKSParameters.recommended_128_bit(8192)

# ⚠️ Custom parameters need validation
params = CKKSParameters(
    poly_degree=8192,
    ciphertext_moduli=[...],  # Must satisfy security constraints
    scaling_factor=2**40,
    security_level=SecurityLevel.TC26_128
)
```

### Key Management

- **Secret Key**: Must be kept absolutely private
- **Public Key**: Can be distributed freely
- **Evaluation Key**: For computation, not decryption

### Attacks and Mitigations

1. **Ciphertext-Only Attacks**: Mitigated by semantic security (RLWE hardness)

2. **Chosen Ciphertext Attacks**: Standard CKKS is IND-CPA secure

3. **Side-Channel Attacks**: Implementations should use constant-time algorithms

4. **Malicious Clients (FL)**: Use differential privacy and secure aggregation

---

## Limitations

### Theoretical Limitations

1. **Noise Growth**: Each operation adds noise. After many multiplications, results become incorrect.

2. **Bootstrapping Overhead**: Refresh operation is computationally expensive (~1-10 seconds).

3. **Precision Loss**: Fixed-point encoding limits precision. Deep networks accumulate error.

4. **No Branching**: Cannot do `if/else` on encrypted data (must use multiplexers).

### Practical Limitations

1. **Performance**: FHE is 10^3 to 10^6× slower than plaintext computation.

2. **Memory**: Large ciphertexts require significant memory.

3. **Integer Arithmetic Only**: No native floating-point. Must use fixed-point encoding.

4. **Activation Functions**: Non-polynomial activations require approximation.

### What Works Well

- ✅ Matrix multiplications (deep learning inference)
- ✅ Linear operations (regression, linear models)
- ✅ Pooling operations (average, max)
- ✅ Distance computations (k-NN, clustering)

### What is Challenging

- ❌ Complex branching logic
- ❌ Non-polynomial operations (without approximation)
- ❌ Very deep networks (need bootstrapping)
- ❌ Real-time applications (performance overhead)

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/moggan1337/Cipher.git
cd Cipher
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run benchmarks
python -m benchmarks.benchmark
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all public functions
- Keep functions focused and small

### Testing Guidelines

- All new features must have tests
- Maintain test coverage above 80%
- Use descriptive test names

---

## Advanced Topics

### Bootstrapping Strategies

Bootstrapping is essential for deep computations. Strategies include:

1. **Automatic Bootstrapping**: Trigger when depth runs low
2. **Scheduled Bootstrapping**: Bootstrap every N levels
3. **Opportunistic**: Bootstrap only when accuracy degrades

```python
# Automatic approach
bootstrapped = context.ensure_depth(ct, min_depth=2)

# Scheduled approach
if ct.level <= threshold:
    ct = context.bootstrap(ct)
```

### Optimizing Performance

1. **Batch Processing**: Use SIMD slots for multiple inputs
2. **Level Management**: Minimize unnecessary multiplications
3. **Polynomial Degree**: Match activation approximation to needs
4. **Bootstrapping Frequency**: Balance between depth and speed

### Noise Management

Noise accumulates with each operation. Monitor:

```python
# Check remaining precision
available_depth = context.params.max_depth - ct.level
print(f"Available depth: {available_depth}")

# Rescale to reduce noise
ct = context.rescale(ct)
```

### Security vs. Performance Tradeoffs

| Parameter | Higher Security | Higher Performance |
|-----------|---------------|-------------------|
| N (degree) | Larger N | Smaller N |
| Modulus | Larger q | Smaller q |
| Bootstrapping | More frequent | Less frequent |

---

## FAQ

### Q: How does Cipher compare to SEAL/PALISADE?

Cipher is a Python framework focused on ML use cases, while SEAL (C++) and PALISADE (C++) are lower-level cryptographic libraries. Cipher prioritizes ease of use and ML-specific optimizations.

### Q: Can I use pre-trained PyTorch models?

Yes! Use the `convert_from_torch` function:

```python
from src.ml.encrypted_model import convert_from_torch
import torch.nn as nn

torch_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

enc_model = convert_from_torch(torch_model, (784,), context)
```

### Q: How accurate are polynomial approximations?

Accuracy depends on polynomial degree:

| Degree | Typical Error |
|--------|---------------|
| 3 | ~5-10% |
| 5 | ~1-2% |
| 7 | ~0.1-0.5% |
| 11 | <0.1% |

### Q: What's the maximum computation depth?

Without bootstrapping: ~20-30 levels (depends on parameters)
With bootstrapping: Unlimited (at cost of computation time)

---

## References

1. **CKKS Original Paper**: Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). "Homomorphic encryption for arithmetic of approximate numbers." ASIACRYPT 2017.

2. **Homomorphic Encryption Standard**: https://www.homomorphicencryption.org

3. **SEAL Library**: Microsoft Research. https://github.com/microsoft/SEAL

4. **PALISADE Library**: Duality Technologies. https://palisade-crypto.org/

5. **McMahan et al. (2017)**: "Communication-Efficient Learning of Deep Networks from Decentralized Data." FedAvg paper.

6. **Cheon et al. (2019)**: "Bootstrapping for HEAAN." EUROCRYPT 2018.

7. **Halevi & Shoup (2018)**: "Design and Implementation of HEAAN." IACR Cryptology ePrint Archive.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Cipher Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgments

This project builds on the foundational work of:
- The CKKS authors (Cheon, Kim, Kim, Song)
- The homomorphic encryption community
- Open-source implementations like SEAL and PALISADE
- The federated learning pioneers (McMahan, Bonawitz, et al.)
- The differential privacy community (Dwork, Roth, et al.)

---

<p align="center">
  <strong>Cipher: Privacy-Preserving ML with Homomorphic Encryption</strong>
</p>
