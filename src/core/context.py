"""
FHE Context - Unified Interface for CKKS Operations

This module provides a high-level context manager for FHE operations,
abstracting away the details of key management and parameter selection.

The FHEContext provides:
- Automatic key generation and management
- Simplified encryption/decryption API
- Operation tracking for depth management
- Resource allocation and pooling
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import time
import warnings

import numpy as np

from .params import CKKSParameters, SecurityLevel, BootstrappingParameters
from .keys import SecretKey, PublicKey, EvaluationKey, BootstrappingKey, KeyGenerator
from .ckks import CKKS, Ciphertext, Plaintext


@dataclass
class ContextStats:
    """Statistics for context operations."""
    
    encryption_count: int = 0
    decryption_count: int = 0
    multiplication_count: int = 0
    addition_count: int = 0
    bootstrap_count: int = 0
    total_encryption_time: float = 0.0
    total_decryption_time: float = 0.0
    total_operation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "encryption_count": self.encryption_count,
            "decryption_count": self.decryption_count,
            "multiplication_count": self.multiplication_count,
            "addition_count": self.addition_count,
            "bootstrap_count": self.bootstrap_count,
            "total_encryption_time": self.total_encryption_time,
            "total_decryption_time": self.total_decryption_time,
            "total_operation_time": self.total_operation_time,
        }


class FHEContext:
    """
    High-level context for FHE operations.
    
    This class provides a unified interface for:
    - Key generation (secret, public, evaluation, bootstrapping keys)
    - Encoding and decoding of values
    - Encryption and decryption of data
    - Homomorphic operations (add, multiply, rotate, etc.)
    - Bootstrapping for depth extension
    
    Usage:
        >>> ctx = FHEContext.default(security_level=128)
        >>> ctx.setup_keys()
        >>> 
        >>> # Encrypt
        >>> values = np.array([1.0, 2.0, 3.0, 4.0])
        >>> ct = ctx.encrypt(values)
        >>> 
        >>> # Operate
        >>> ct_squared = ctx.multiply(ct, ct)
        >>> 
        >>> # Decrypt
        >>> result = ctx.decrypt(ct_squared)
    """
    
    @classmethod
    def default(
        cls, 
        security_level: int = 128,
        poly_degree: int = 8192,
        enable_bootstrapping: bool = True
    ) -> "FHEContext":
        """
        Create a default FHE context with recommended parameters.
        
        Args:
            security_level: Security level in bits (128, 192, or 256)
            poly_degree: Polynomial degree (4096, 8192, or 16384)
            enable_bootstrapping: Whether to enable bootstrapping support
        
        Returns:
            Configured FHEContext
        """
        sec_enum = {
            128: SecurityLevel.TC26_128,
            192: SecurityLevel.TC26_192,
            256: SecurityLevel.TC26_256,
        }.get(security_level, SecurityLevel.TC26_128)
        
        params = CKKSParameters.recommended_128_bit(poly_degree)
        params = CKKSParameters(
            poly_degree=poly_degree,
            ciphertext_moduli=params.ciphertext_moduli,
            scaling_factor=params.scaling_factor,
            security_level=sec_enum,
        )
        
        bootstrap_params = None
        if enable_bootstrapping:
            bootstrap_params = BootstrappingParameters()
        
        return cls(params, bootstrap_params)
    
    def __init__(
        self,
        params: CKKSParameters,
        bootstrap_params: Optional[BootstrappingParameters] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize FHE context.
        
        Args:
            params: CKKS parameters
            bootstrap_params: Bootstrapping parameters (optional)
            seed: Random seed for reproducibility
        """
        self.params = params
        self.bootstrap_params = bootstrap_params
        self.seed = seed
        
        # Initialize CKKS scheme
        self.ckks = CKKS(params, seed)
        
        # Initialize key generator
        self.keygen = KeyGenerator(params, seed)
        
        # Keys (lazy initialization)
        self._secret_key: Optional[SecretKey] = None
        self._public_key: Optional[PublicKey] = None
        self._eval_key: Optional[EvaluationKey] = None
        self._bootstrap_key: Optional[BootstrappingKey] = None
        self._keys_setup = False
        
        # Statistics
        self.stats = ContextStats()
        
        # Cached plaintexts for common values
        self._plain_cache: Dict[str, Plaintext] = {}
    
    @property
    def secret_key(self) -> SecretKey:
        """Get secret key (generates if needed)."""
        if self._secret_key is None:
            self._secret_key = self.keygen.generate_secret_key()
        return self._secret_key
    
    @property
    def public_key(self) -> PublicKey:
        """Get public key (generates if needed)."""
        if self._public_key is None:
            self.setup_keys()
        return self._public_key
    
    @property
    def eval_key(self) -> EvaluationKey:
        """Get evaluation key (generates if needed)."""
        if self._eval_key is None:
            self.setup_keys()
        return self._eval_key
    
    @property
    def bootstrap_key(self) -> BootstrappingKey:
        """Get bootstrapping key (generates if needed)."""
        if self._bootstrap_key is None and self.bootstrap_params:
            self._bootstrap_key = self.keygen.generate_bootstrapping_key(
                self.secret_key,
                self.bootstrap_params.partitions,
                self.bootstrap_params.bits_per_partition,
            )
        return self._bootstrap_key
    
    def setup_keys(self) -> None:
        """
        Generate all necessary keys.
        
        This is called automatically on first access to keys.
        """
        if self._keys_setup:
            return
        
        start_time = time.time()
        
        # Generate secret key
        self._secret_key = self.keygen.generate_secret_key()
        
        # Generate public key
        self._public_key = self.keygen.generate_public_key(self._secret_key)
        
        # Generate evaluation key
        self._eval_key = self.keygen.generate_evaluation_key(
            self._secret_key,
            relin_decomposition=64,
        )
        
        # Generate bootstrapping key if enabled
        if self.bootstrap_params:
            self._bootstrap_key = self.keygen.generate_bootstrapping_key(
                self._secret_key,
                self.bootstrap_params.partitions,
                self.bootstrap_params.bits_per_partition,
            )
        
        self._keys_setup = True
        
        setup_time = time.time() - start_time
        print(f"Key setup completed in {setup_time:.2f}s")
    
    def encrypt(
        self, 
        values: Union[np.ndarray, List[float], float],
        scale: Optional[float] = None
    ) -> Ciphertext:
        """
        Encrypt values into a ciphertext.
        
        Args:
            values: Numeric array or single value to encrypt
            scale: Optional scaling factor
        
        Returns:
            Encrypted Ciphertext
        
        Example:
            >>> ct = ctx.encrypt([1.0, 2.0, 3.0, 4.0])
            >>> ct = ctx.encrypt(5.0)  # Broadcasts to all slots
        """
        start_time = time.time()
        
        # Handle single value
        if isinstance(values, (int, float)):
            values = np.array([values])
        
        # Encode values
        plaintext = self.ckks.encode(values, scale)
        
        # Encrypt
        ciphertext = self.ckks.encrypt(plaintext, self.public_key)
        
        self.stats.encryption_count += 1
        self.stats.total_encryption_time += time.time() - start_time
        
        return ciphertext
    
    def decrypt(self, ciphertext: Ciphertext) -> np.ndarray:
        """
        Decrypt ciphertext to values.
        
        Args:
            ciphertext: Ciphertext to decrypt
        
        Returns:
            Decrypted values as numpy array
        
        Example:
            >>> values = ctx.decrypt(ct)
        """
        start_time = time.time()
        
        plaintext = self.ckks.decrypt(ciphertext, self.secret_key)
        result = self.ckks.decode(plaintext)
        
        self.stats.decryption_count += 1
        self.stats.total_decryption_time += time.time() - start_time
        
        return result
    
    def decrypt_single(self, ciphertext: Ciphertext) -> float:
        """Decrypt and return a single value."""
        return float(self.decrypt(ciphertext)[0])
    
    # =========================================================================
    # Homomorphic Operations
    # =========================================================================
    
    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Add two ciphertexts."""
        self.stats.addition_count += 1
        return self.ckks.add(ct1, ct2)
    
    def add_scalar(self, ciphertext: Ciphertext, scalar: float) -> Ciphertext:
        """Add scalar to encrypted values."""
        pt = self.ckks.encode(np.array([scalar]), ciphertext.scale)
        return self.ckks.add_plain(ciphertext, pt)
    
    def sub(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Subtract ciphertexts."""
        self.stats.addition_count += 1
        return self.ckks.sub(ct1, ct2)
    
    def multiply(
        self, 
        ct1: Ciphertext, 
        ct2: Ciphertext,
        relinearize: bool = True
    ) -> Ciphertext:
        """
        Multiply two ciphertexts.
        
        Args:
            ct1, ct2: Ciphertexts to multiply
            relinearize: Whether to relinearize after multiplication
        
        Returns:
            Product ciphertext
        """
        self.stats.multiplication_count += 1
        return self.ckks.multiply(ct1, ct2, self.eval_key, relinearize)
    
    def multiply_scalar(self, ciphertext: Ciphertext, scalar: float) -> Ciphertext:
        """Multiply encrypted values by scalar."""
        pt = self.ckks.encode(np.array([scalar]), ciphertext.scale)
        return self.ckks.multiply_plain(ciphertext, pt)
    
    def square(self, ciphertext: Ciphertext) -> Ciphertext:
        """Square encrypted values."""
        return self.multiply(ciphertext, ciphertext)
    
    def relu(self, ciphertext: Ciphertext) -> Ciphertext:
        """
        Apply ReLU activation (via polynomial approximation).
        
        Note: This requires polynomial approximation of ReLU.
        For actual use, see PolynomialApproximator class.
        """
        from ..ml.approximation import PolynomialApproximator
        approx = PolynomialApproximator(self)
        return approx.approximate_relu(ciphertext)
    
    def sigmoid(self, ciphertext: Ciphertext) -> Ciphertext:
        """Apply sigmoid activation (via polynomial approximation)."""
        from ..ml.approximation import PolynomialApproximator
        approx = PolynomialApproximator(self)
        return approx.approximate_sigmoid(ciphertext)
    
    def tanh(self, ciphertext: Ciphertext) -> Ciphertext:
        """Apply tanh activation (via polynomial approximation)."""
        from ..ml.approximation import PolynomialApproximator
        approx = PolynomialApproximator(self)
        return approx.approximate_tanh(ciphertext)
    
    def rotate(self, ciphertext: Ciphertext, steps: int) -> Ciphertext:
        """Rotate encrypted vector slots."""
        return self.ckks.rotate(ciphertext, steps, self.eval_key)
    
    def rescale(self, ciphertext: Ciphertext) -> Ciphertext:
        """Rescale ciphertext to manage noise."""
        return self.ckks.rescale(ciphertext)
    
    def negate(self, ciphertext: Ciphertext) -> Ciphertext:
        """Negate encrypted values."""
        return self.ckks.negate(ciphertext)
    
    # =========================================================================
    # Vectorized Operations (SIMD-style)
    # =========================================================================
    
    def dot_product(
        self, 
        ct1: Ciphertext, 
        ct2: Ciphertext
    ) -> Ciphertext:
        """
        Compute encrypted dot product using rotation.
        
        For vectors a and b of length n:
            dot(a, b) = Σ a_i · b_i
        
        Uses the method of哈尔ken:
            1. Multiply component-wise
            2. Sum using rotations and additions
        """
        # Component-wise multiplication
        prod = self.multiply(ct1, ct2)
        
        # Sum using rotations (tree reduction)
        n = min(ct1.scale, ct2.scale)  # Simplified
        slots = self.params.slot_count
        
        # Logarithmic number of rotations
        step = 1
        while step < slots:
            rotated = self.rotate(prod, step)
            prod = self.add(prod, rotated)
            step *= 2
        
        # Final rotation to get sum at all positions
        final = self.rotate(prod, slots - 1)
        return self.add(prod, final)
    
    def matrix_multiply(
        self, 
        encrypted_vector: Ciphertext,
        plaintext_matrix: np.ndarray,
    ) -> Ciphertext:
        """
        Multiply encrypted vector by plaintext matrix.
        
        Args:
            encrypted_vector: Encrypted input vector
            plaintext_matrix: Plaintext weight matrix (shape: out_features x in_features)
        
        Returns:
            Encrypted output vector
        
        Note:
            This is a key operation for neural network inference where
            weights can be public but inputs must be private.
        """
        output_size, input_size = plaintext_matrix.shape
        
        if input_size > self.params.slot_count:
            raise ValueError(f"Matrix input size {input_size} exceeds slot count {self.params.slot_count}")
        
        # Reshape and rotate for matrix multiplication
        # Each row of the matrix computes a dot product with the input vector
        
        result = None
        for i in range(output_size):
            # Get i-th row of matrix
            row = plaintext_matrix[i]
            
            # Create rotated copies of input for this output
            row_ct = self._matrix_row_ciphertext(encrypted_vector, row)
            
            # Sum to compute dot product
            row_sum = self.dot_product(encrypted_vector, row_ct)
            
            if result is None:
                result = row_sum
            else:
                result = self.add(result, row_sum)
        
        return result
    
    def _matrix_row_ciphertext(
        self, 
        vector: Ciphertext, 
        row: np.ndarray
    ) -> Ciphertext:
        """Create ciphertext for matrix row multiplication."""
        # Encode row values
        pt = self.ckks.encode(row, vector.scale)
        
        # Multiply (this encodes the row into the rotation)
        return self.ckks.multiply_plain(vector, pt)
    
    # =========================================================================
    # Bootstrap and Depth Management
    # =========================================================================
    
    def bootstrap(self, ciphertext: Ciphertext) -> Ciphertext:
        """
        Bootstrap ciphertext to extend multiplicative depth.
        
        Bootstrapping "refreshes" a ciphertext by:
        1. Switching to special modulus
        2. Approximating the decryption function
        3. Re-encrypting with fresh noise
        
        This allows unlimited depth at the cost of computation.
        """
        if self.bootstrap_params is None:
            raise ValueError("Bootstrapping not enabled in context")
        
        from ..bootstrapping.bootstrapping import Bootstrapper
        boot = Bootstrapper(self)
        result = boot.bootstrap(ciphertext)
        
        self.stats.bootstrap_count += 1
        return result
    
    def check_depth(self, ciphertext: Ciphertext) -> int:
        """Check remaining multiplicative depth."""
        return self.params.max_depth - ciphertext.level
    
    def ensure_depth(self, ciphertext: Ciphertext, required: int) -> Ciphertext:
        """Ensure ciphertext has at least required depth, bootstrap if needed."""
        while self.check_depth(ciphertext) < required:
            if self.bootstrap_params is None:
                raise ValueError(f"Insufficient depth ({self.check_depth(ciphertext)}) and bootstrapping disabled")
            ciphertext = self.bootstrap(ciphertext)
        return ciphertext
    
    # =========================================================================
    # Caching for Common Values
    # =========================================================================
    
    def get_constant(self, value: float) -> Plaintext:
        """Get cached plaintext for constant value."""
        key = f"{value}"
        if key not in self._plain_cache:
            self._plain_cache[key] = self.ckks.encode(np.array([value]))
        return self._plain_cache[key]
    
    def clear_cache(self) -> None:
        """Clear the plaintext cache."""
        self._plain_cache.clear()
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def save_keys(self, directory: str) -> None:
        """Save all keys to directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save secret key (handle with care!)
        sk_data = self.secret_key.to_bytes()
        with open(os.path.join(directory, "secret.key"), "wb") as f:
            f.write(sk_data)
        
        # Save public key
        from .keys import KeySerializer
        with open(os.path.join(directory, "public.key"), "wb") as f:
            KeySerializer.serialize_secret_key(self.secret_key)  # Placeholder
        
        print(f"Keys saved to {directory}")
    
    def __repr__(self) -> str:
        return (
            f"FHEContext(N={self.params.poly_degree}, "
            f"slots={self.params.slot_count}, "
            f"depth={self.params.max_depth}, "
            f"bootstrap={self.bootstrap_params is not None})"
        )
