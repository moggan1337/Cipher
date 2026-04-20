"""
Test suite for Cipher - Homomorphic Encryption ML Framework
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.params import CKKSParameters, SecurityLevel
from src.core.keys import KeyGenerator
from src.core.ckks import CKKS, Ciphertext, Plaintext
from src.core.context import FHEContext
from src.ml.encrypted_model import (
    EncryptedLinear, EncryptedReLU, EncryptedSigmoid,
    EncryptedSequential, EncryptedTanh
)
from src.ml.approximation import PolynomialApproximator
from src.polynomial.polynomial import Polynomial, ChebyshevApproximator


class TestCKKSParameters:
    """Test CKKS parameter validation."""
    
    def test_poly_degree_power_of_2(self):
        """Poly degree must be power of 2."""
        with pytest.raises(ValueError):
            CKKSParameters(
                poly_degree=1000,
                ciphertext_moduli=[0x1FFFFFE00001, 0x1FFFFFC00001],
                scaling_factor=2**40,
            )
    
    def test_security_validation(self):
        """Security parameters validated."""
        params = CKKSParameters.recommended_128_bit(4096)
        assert params.security_level == SecurityLevel.TC26_128
        assert params.slot_count == 2048
    
    def test_recommended_params(self):
        """Test recommended parameter generation."""
        for degree in [4096, 8192]:
            params = CKKSParameters.recommended_128_bit(degree)
            assert params.poly_degree == degree
            assert params.slot_count == degree // 2


class TestPolynomial:
    """Test polynomial operations."""
    
    def test_basic_operations(self):
        """Test polynomial arithmetic."""
        p1 = Polynomial(np.array([1.0, 2.0, 3.0]))  # 1 + 2x + 3x²
        p2 = Polynomial(np.array([0.5, 1.0]))       # 0.5 + x
        
        # Addition
        p_sum = p1 + p2
        assert np.allclose(p_sum.coefficients, [1.5, 3.0, 3.0])
        
        # Multiplication
        p_prod = p1 * p2
        assert len(p_prod.coefficients) == 5
    
    def test_evaluation(self):
        """Test polynomial evaluation."""
        p = Polynomial(np.array([1.0, 2.0, 1.0]))  # (x+1)²
        assert np.isclose(p(2), 9.0)
        assert np.isclose(p(-1), 0.0)
    
    def test_chebyshev_approximation(self):
        """Test Chebyshev approximation."""
        def f(x):
            return np.sin(x)
        
        approx = ChebyshevApproximator.approximate(f, degree=5, domain=(-1, 1))
        
        # Check approximation quality
        x_test = np.linspace(-1, 1, 100)
        errors = np.abs(f(x_test) - approx(x_test))
        max_error = np.max(errors)
        
        assert max_error < 0.01, f"Chebyshev approximation error too high: {max_error}"


class TestKeyGeneration:
    """Test key generation."""
    
    @pytest.fixture
    def params(self):
        """Create small parameters for testing."""
        return CKKSParameters(
            poly_degree=512,
            ciphertext_moduli=[0xFFFFFE00001, 0xFFFFFC00001],
            scaling_factor=2**30,
        )
    
    @pytest.fixture
    def keygen(self, params):
        """Create key generator."""
        return KeyGenerator(params, seed=42)
    
    def test_secret_key_generation(self, keygen):
        """Test secret key generation."""
        sk = keygen.generate_secret_key()
        assert len(sk.coefficients) == 512
        assert sk.hamming_weight > 0
    
    def test_public_key_generation(self, keygen):
        """Test public key generation."""
        sk = keygen.generate_secret_key()
        pk = keygen.generate_public_key(sk)
        assert len(pk.a) == 512
        assert len(pk.b) == 512
    
    def test_evaluation_key_generation(self, keygen):
        """Test evaluation key generation."""
        sk = keygen.generate_secret_key()
        ek = keygen.generate_evaluation_key(sk)
        assert len(ek.relin_keys) > 0


class TestCKKSOperations:
    """Test core CKKS operations."""
    
    @pytest.fixture
    def context(self):
        """Create test context."""
        return FHEContext.default(security_level=128, poly_degree=1024)
    
    def test_encrypt_decrypt(self, context):
        """Test basic encryption and decryption."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        ct = context.encrypt(values)
        decrypted = context.decrypt(ct)
        
        # Check approximate equality (some precision loss expected)
        assert np.allclose(values, decrypted[:len(values)], rtol=0.1)
    
    def test_addition(self, context):
        """Test homomorphic addition."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([0.5, 1.5, 2.5, 3.5])
        
        ct_a = context.encrypt(a)
        ct_b = context.encrypt(b)
        ct_sum = context.add(ct_a, ct_b)
        
        result = context.decrypt(ct_sum)
        expected = a + b
        
        assert np.allclose(expected, result[:len(expected)], rtol=0.1)
    
    def test_multiplication(self, context):
        """Test homomorphic multiplication."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        
        ct_a = context.encrypt(a)
        ct_b = context.encrypt(b)
        ct_prod = context.multiply(ct_a, ct_b)
        
        result = context.decrypt(ct_prod)
        expected = a * b
        
        assert np.allclose(expected, result[:len(expected)], rtol=0.1)
    
    def test_square(self, context):
        """Test squaring."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        
        ct = context.encrypt(values)
        ct_sq = context.square(ct)
        
        result = context.decrypt(ct_sq)
        expected = values ** 2
        
        assert np.allclose(expected, result[:len(expected)], rtol=0.1)


class TestPolynomialApproximation:
    """Test polynomial approximation of activations."""
    
    @pytest.fixture
    def context(self):
        """Create test context."""
        return FHEContext.default(security_level=128, poly_degree=512)
    
    def test_relu_approximation(self, context):
        """Test ReLU approximation."""
        values = np.array([0.5, -0.5, 1.0, -1.0, 0.0])
        
        ct = context.encrypt(values)
        approx = PolynomialApproximator(context)
        poly = approx.get_relu(degree=3)
        
        ct_relu = approx.evaluate(ct, poly)
        result = context.decrypt(ct_relu)
        
        expected = np.maximum(0, values)
        assert np.allclose(expected, result[:len(expected)], rtol=0.5)
    
    def test_sigmoid_approximation(self, context):
        """Test sigmoid approximation."""
        values = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        
        ct = context.encrypt(values)
        approx = PolynomialApproximator(context)
        poly = approx.get_sigmoid(degree=5)
        
        ct_sigmoid = approx.evaluate(ct, poly)
        result = context.decrypt(ct_sigmoid)
        
        expected = 1 / (1 + np.exp(-values))
        assert np.allclose(expected, result[:len(expected)], rtol=0.3)


class TestEncryptedLayers:
    """Test encrypted neural network layers."""
    
    @pytest.fixture
    def context(self):
        """Create test context."""
        return FHEContext.default(security_level=128, poly_degree=512)
    
    def test_encrypted_linear(self, context):
        """Test encrypted linear layer."""
        # Create layer
        layer = EncryptedLinear(4, 2)
        weights = np.random.randn(2, 4)
        layer.set_weights(weights)
        
        # Encrypt input
        x = np.array([1.0, 2.0, 3.0, 4.0])
        ct_x = context.encrypt(x)
        
        # Forward pass
        ct_y = layer.forward(ct_x, context, weights_public=True)
        
        # Decrypt and check
        y = context.decrypt(ct_y)
        expected = x @ weights.T
        
        assert np.allclose(expected, y[:2], rtol=0.1)
    
    def test_encrypted_relu(self, context):
        """Test encrypted ReLU."""
        relu = EncryptedReLU(degree=3)
        
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        ct_x = context.encrypt(x)
        
        ct_y = relu(ct_x, context)
        y = context.decrypt(ct_y)
        
        expected = np.maximum(0, x)
        assert np.allclose(expected, y[:len(expected)], rtol=0.5)


class TestEncryptedSequential:
    """Test sequential model."""
    
    @pytest.fixture
    def context(self):
        """Create test context."""
        return FHEContext.default(security_level=128, poly_degree=512)
    
    def test_sequential_forward(self, context):
        """Test forward pass through sequential model."""
        model = EncryptedSequential([
            EncryptedLinear(4, 8),
            EncryptedReLU(),
            EncryptedLinear(8, 2),
        ])
        
        # Set dummy weights
        for i, layer in enumerate(model.layers):
            if isinstance(layer, EncryptedLinear):
                w = np.random.randn(layer.out_features, layer.in_features) * 0.1
                layer.set_weights(w)
        
        # Forward pass
        x = np.array([1.0, 2.0, 3.0, 4.0])
        ct_x = context.encrypt(x)
        ct_y = model.forward(ct_x, context)
        
        # Should work without error
        assert ct_y is not None


class TestFederatedLearning:
    """Test federated learning components."""
    
    @pytest.fixture
    def context(self):
        """Create test context."""
        return FHEContext.default(security_level=128, poly_degree=512)
    
    def test_client_registration(self, context):
        """Test federated client registration."""
        from src.ml.encrypted_training import FederatedServer, FederatedClient, EncryptedSequential
        
        # Create server and model
        model = EncryptedSequential([
            EncryptedLinear(4, 2),
        ])
        
        server = FederatedServer(model, context)
        
        # Create clients
        client1 = FederatedClient("client1", model, context)
        client2 = FederatedClient("client2", model, context)
        
        # Register
        server.register_client(client1)
        server.register_client(client2)
        
        assert server.num_clients == 2
        
        # Unregister
        server.unregister_client("client1")
        assert server.num_clients == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
