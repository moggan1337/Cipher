"""
Cryptographic Keys for CKKS Scheme

This module implements key generation for the CKKS fully homomorphic encryption scheme.

Key Types:
1. Secret Key (sk): For decryption, kept private
2. Public Key (pk): For encryption, can be shared
3. Evaluation Key (ek): For homomorphic operations (relinearization, rotations)
4. Bootstrapping Key (bk): For bootstrapping operation

Mathematical Background:
- Secret key s is sampled from discrete Gaussian or ternary distribution
- Public key (a, b) = (-as + e, s) where a is uniform, e is small error
- Relinearization keys allow converting degree-2 ciphertexts back to degree-1
- Rotation keys enable slot rotations for SIMD operations
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict, Any
import time
from dataclasses import dataclass, field

from .params import CKKSParameters, SecurityLevel


@dataclass
class SecretKey:
    """
    CKKS Secret Key.
    
    The secret key s is a polynomial in the ring R_q = Z_q[X]/(X^N + 1).
    It is typically sampled from a ternary distribution {-1, 0, 1} with 
    hamming weight H (sparse secret key variant).
    
    Mathematical Form:
        s(X) = Σ s_i X^i where s_i ∈ {-1, 0, 1} and |{i: s_i ≠ 0}| = H
    
    Security depends on the hardness of the Ring-LWE problem.
    """
    
    coefficients: NDArray[np.int64]
    params: CKKSParameters
    hamming_weight: int = 0
    distribution: str = "ternary"  # or "gaussian"
    
    def __post_init__(self):
        if self.hamming_weight == 0:
            self.hamming_weight = int(np.sum(self.coefficients != 0))
    
    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return len(self.coefficients)
    
    def to_bytes(self) -> bytes:
        """Serialize secret key to bytes."""
        return self.coefficients.tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes, params: CKKSParameters) -> "SecretKey":
        """Deserialize secret key from bytes."""
        coeffs = np.frombuffer(data, dtype=np.int64)
        return cls(coefficients=coeffs, params=params)
    
    def __repr__(self) -> str:
        return f"SecretKey(N={self.degree}, H={self.hamming_weight})"


@dataclass 
class PublicKey:
    """
    CKKS Public Key.
    
    Public key is generated from secret key as:
        pk = (a, b) where b = -a·s + e (mod q)
    
    The public key is essentially an RLWE sample with secret s.
    Anyone can encrypt using pk, but only holder of sk can decrypt.
    
    The ciphertext consists of two polynomials (c_0, c_1).
    """
    
    a: NDArray[np.int64]  # Uniform random polynomial
    b: NDArray[np.int64]  # -a·s + e mod q
    params: CKKSParameters
    ciphertext_modulus: int  # Current modulus level
    
    def __post_init__(self):
        if len(self.a) != self.params.poly_degree:
            raise ValueError(f"Polynomial length mismatch: expected {self.params.poly_degree}")
    
    @property
    def polynomial_pair(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the public key as (c_0, c_1) = (b, a)."""
        return (self.b, self.a)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "modulus": self.ciphertext_modulus,
        }
    
    def __repr__(self) -> str:
        return f"PublicKey(N={self.params.poly_degree}, log(q)≈{np.log2(self.ciphertext_modulus):.1f})"


@dataclass
class EvaluationKey:
    """
    CKKS Evaluation Key for relinearization.
    
    Relinearization converts a multiplication ciphertext (degree 2)
    back to a standard ciphertext (degree 1).
    
    For CKKS with BV key-switching:
        rlk[0] = [-P·a_0 + e_0]_Q·P + Q·ū_0
        rlk[1] = [-P·a_1 + e_1]_Q·P + Q·ū_1
    
    Where P is a large special modulus, and (a_i, ū_i) are RLWE samples.
    """
    
    relin_keys: List[Tuple[NDArray[np.int64], NDArray[np.int64]]]
    rotation_keys: Dict[int, Tuple[NDArray[np.int64], NDArray[np.int64]]]
    params: CKKSParameters
    
    def get_relin_key(self, key_index: int) -> Tuple[NDArray, NDArray]:
        """Get relinearization key for given index."""
        if key_index >= len(self.relin_keys):
            raise ValueError(f"No relin key at index {key_index}")
        return self.relin_keys[key_index]
    
    def get_rotation_key(self, step: int) -> Optional[Tuple[NDArray, NDArray]]:
        """Get rotation key for given rotation step."""
        return self.rotation_keys.get(step)
    
    def has_rotation_key(self, step: int) -> bool:
        """Check if rotation key exists for step."""
        return step in self.rotation_keys
    
    def __repr__(self) -> str:
        return f"EvaluationKey(relin={len(self.relin_keys)}, rotations={len(self.rotation_keys)})"


@dataclass
class BootstrappingKey:
    """
    CKKS Bootstrapping Key.
    
    Bootstrapping requires special keys for:
    1. Expanding the ciphertext for slot-wise operations
    2. Computing the modular reduction polynomial
    3. Compressing back to fresh ciphertext
    
    The bootstrapping key is essentially a set of Gadget vectors
    for the baby-step giant-step approach.
    """
    
    # Partition count for the sine polynomial
    partitions: int
    
    # Number of bits per partition
    bits_per_partition: int
    
    # Gadget vectors for each partition
    gadget_vectors: List[NDArray[np.int64]]
    
    # Powers of the secret key
    secret_powers: List[NDArray[np.int64]]
    
    params: CKKSParameters
    
    @property
    def gadget_dim(self) -> int:
        """Gadget dimension."""
        return len(self.gadget_vectors)
    
    def __repr__(self) -> str:
        return f"BootstrappingKey(partitions={self.partitions})"


class KeyGenerator:
    """
    Key Generator for CKKS scheme.
    
    Handles generation of all key types with configurable security parameters.
    
    Usage:
        >>> params = CKKSParameters.recommended_128_bit(8192)
        >>> generator = KeyGenerator(params)
        >>> sk = generator.generate_secret_key()
        >>> pk = generator.generate_public_key(sk)
        >>> ek = generator.generate_evaluation_key(sk)
    """
    
    def __init__(self, params: CKKSParameters, seed: Optional[int] = None):
        """
        Initialize key generator.
        
        Args:
            params: CKKS scheme parameters
            seed: Random seed for reproducibility
        """
        self.params = params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._generation_time = 0.0
    
    def generate_secret_key(
        self, 
        hamming_weight: Optional[int] = None,
        distribution: str = "ternary"
    ) -> SecretKey:
        """
        Generate secret key.
        
        Args:
            hamming_weight: Number of non-zero coefficients (default: N/4)
            distribution: Distribution type ("ternary" or "gaussian")
        
        Returns:
            SecretKey instance
        
        Mathematical Details:
            Ternary: s_i ~ U({-1, 0, 1}) with hamming weight H
            Gaussian: s_i ~ N(0, σ²) with truncation
        """
        start_time = time.time()
        
        n = self.params.poly_degree
        hw = hamming_weight or n // 4  # Default hamming weight
        
        if distribution == "ternary":
            # Sample hamming weight positions
            indices = self.rng.choice(n, size=hw, replace=False)
            coefficients = np.zeros(n, dtype=np.int64)
            signs = self.rng.choice([-1, 1], size=hw)
            coefficients[indices] = signs
        else:
            # Gaussian distribution (for higher security)
            sigma = 3.19  # Standard deviation for ~128-bit security
            coefficients = np.round(self.rng.normal(0, sigma, size=n)).astype(np.int64)
            coefficients = np.mod(coefficients, self.params.ciphertext_modulus)
            # Center around 0
            coefficients = np.where(coefficients > self.params.ciphertext_modulus // 2,
                                   coefficients - self.params.ciphertext_modulus, coefficients)
        
        self._generation_time = time.time() - start_time
        
        return SecretKey(
            coefficients=coefficients,
            params=self.params,
            hamming_weight=hw,
            distribution=distribution,
        )
    
    def generate_public_key(self, secret_key: SecretKey) -> PublicKey:
        """
        Generate public key from secret key.
        
        Args:
            secret_key: The secret key
        
        Returns:
            PublicKey instance
        
        Mathematical Process:
            1. Sample a uniform polynomial a ← R_q
            2. Sample error e from discrete Gaussian
            3. Compute b = -a·s + e (mod q)
            
        The pair (a, b) is an RLWE sample.
        """
        n = self.params.poly_degree
        q = self.params.ciphertext_modulus
        
        # Sample a uniformly from Z_q^N
        a = self.rng.integers(0, q, size=n, dtype=np.int64)
        
        # Sample error from discrete Gaussian
        sigma = 3.19
        e = np.round(self.rng.normal(0, sigma, size=n)).astype(np.int64)
        
        # Compute b = -a·s + e (mod q)
        # Use NTT for efficient polynomial multiplication
        a_s = self._poly_mult(a, secret_key.coefficients, q)
        b = np.mod(-a_s + e, q)
        
        return PublicKey(
            a=a,
            b=b,
            params=self.params,
            ciphertext_modulus=q,
        )
    
    def generate_evaluation_key(
        self, 
        secret_key: SecretKey,
        relin_decomposition: int = 64,
        rotation_steps: Optional[List[int]] = None
    ) -> EvaluationKey:
        """
        Generate evaluation key (relinearization and rotation keys).
        
        Args:
            secret_key: The secret key
            relin_decomposition: Gadget dimension for relinearization
            rotation_steps: List of rotation steps to generate keys for
        
        Returns:
            EvaluationKey containing relin and rotation keys
        """
        q = self.params.ciphertext_modulus
        p = 2**60  # Special modulus P for relinearization
        
        # Relinearization keys
        relin_keys = []
        for i in range(relin_decomposition):
            # Sample (a_i, u_i) RLWE pair
            a_i = self.rng.integers(0, p * q, size=self.params.poly_degree, dtype=np.int64)
            e_i = np.round(self.rng.normal(0, 3.19, size=self.params.poly_degree)).astype(np.int64)
            
            # Compute ū_i = a_i·s + e_i (mod q) and form key
            a_i_s = self._poly_mult(a_i % q, secret_key.coefficients, q)
            u_i = np.mod(a_i_s + e_i, q)
            
            # Key format: (K_0, K_1) = (P·ū_i + P·a_i + e, -P·a_i)
            # Simplified for demonstration
            k0 = np.mod(p * u_i, p * q)
            k1 = np.mod(-p * (a_i % q), p * q)
            
            relin_keys.append((k0, k1))
        
        # Rotation keys (simplified - real implementation needs automorphisms)
        rotation_keys = {}
        if rotation_steps is None:
            rotation_steps = list(range(1, min(64, self.params.slot_count)))
        
        for step in rotation_steps:
            a_rot = self.rng.integers(0, q, size=self.params.poly_degree, dtype=np.int64)
            e_rot = np.round(self.rng.normal(0, 3.19, size=self.params.poly_degree)).astype(np.int64)
            
            # For rotation: key that allows computing s^k · ct
            a_s = self._poly_mult(a_rot, secret_key.coefficients, q)
            k0 = np.mod(-a_s + e_rot, q)
            
            rotation_keys[step] = (k0, a_rot)
        
        return EvaluationKey(
            relin_keys=relin_keys,
            rotation_keys=rotation_keys,
            params=self.params,
        )
    
    def generate_bootstrapping_key(
        self,
        secret_key: SecretKey,
        partitions: int = 8,
        bits_per_partition: int = 8
    ) -> BootstrappingKey:
        """
        Generate bootstrapping key for ciphertext refresh.
        
        Args:
            secret_key: The secret key
            partitions: Number of partitions for gadget decomposition
            bits_per_partition: Bits per partition (total = partitions * bits_per_partition)
        
        Returns:
            BootstrappingKey instance
        """
        q = self.params.ciphertext_modulus
        
        # Compute secret powers: s, s², s³, ...
        secret_powers = [secret_key.coefficients.copy()]
        for _ in range(partitions - 1):
            next_power = self._poly_mult(secret_powers[-1], secret_key.coefficients, q)
            secret_powers.append(np.mod(next_power, q))
        
        # Gadget vectors for baby-step giant-step
        gadget_vectors = []
        for power_idx in range(partitions):
            # Gadget vector: [1, B, B², ..., B^{bits-1}] ⊗ s^power
            base = 2 ** bits_per_partition
            g = []
            for b in range(1 << bits_per_partition):
                g.append(b * secret_powers[power_idx])
            gadget_vectors.append(np.array(g))
        
        return BootstrappingKey(
            partitions=partitions,
            bits_per_partition=bits_per_partition,
            gadget_vectors=gadget_vectors,
            secret_powers=secret_powers,
            params=self.params,
        )
    
    def _poly_mult(
        self, 
        a: NDArray[np.int64], 
        b: NDArray[np.int64], 
        modulus: int
    ) -> NDArray[np.int64]:
        """
        Polynomial multiplication in Z_mod[X]/(X^N + 1).
        
        Uses convolution with wrap-around for the negative power.
        
        Args:
            a, b: Input polynomials
            modulus: Modulus for result
        
        Returns:
            Product polynomial modulo X^N + 1
        """
        n = len(a)
        # Simple convolution (use NTT for large N)
        result = np.convolve(a, b, mode='full')
        
        # Wrap around for X^N = -1 (i.e., X^N ≡ -1 mod X^N + 1)
        if len(result) > n:
            # High degree terms wrap around with negative sign
            result[:len(result) - n] += result[len(result) - n:]
            result = result[:n]
        
        # Apply X^N = -1
        result = np.mod(result, modulus)
        return result
    
    @property
    def generation_time(self) -> float:
        """Time spent on key generation."""
        return self._generation_time


class KeySerializer:
    """Serialization utilities for keys."""
    
    @staticmethod
    def serialize_secret_key(sk: SecretKey) -> bytes:
        """Serialize secret key to bytes."""
        data = {
            "coefficients": sk.coefficients.tobytes(),
            "params": {
                "poly_degree": sk.params.poly_degree,
                "moduli": sk.params.ciphertext_moduli,
            },
        }
        import pickle
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize_secret_key(data: bytes) -> SecretKey:
        """Deserialize secret key from bytes."""
        import pickle
        d = pickle.loads(data)
        params = CKKSParameters(
            poly_degree=d["params"]["poly_degree"],
            ciphertext_moduli=d["params"]["moduli"],
            scaling_factor=2**40,  # Default, should be stored
        )
        coeffs = np.frombuffer(d["coefficients"], dtype=np.int64)
        return SecretKey(coefficients=coeffs, params=params)
