"""
CKKS Parameters and Security Configuration

This module defines the parameters for the CKKS homomorphic encryption scheme,
including polynomial degree, ciphertext modulus, scaling factor, and security levels.

Mathematical Background:
- CKKS operates on polynomial rings R = Z[X]/(X^N + 1)
- The polynomial degree N determines the capacity (slot count = N/2 for complex numbers)
- Ciphertext modulus q must be a product of prime factors for modular arithmetic
- Scaling factor Δ controls precision through rescaling operations
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import math


class SecurityLevel(Enum):
    """Security levels based on Homomorphic Encryption Standard."""
    NONE = 0
    TC26_128 = 128  # ~128-bit security (TC26-PKE 2023)
    TC26_192 = 192  # ~192-bit security
    TC26_256 = 256  # ~256-bit security


@dataclass
class CKKSParameters:
    """
    CKKS scheme parameters.
    
    Attributes:
        poly_degree: Polynomial degree N (ring dimension). Must be a power of 2.
                    Higher values provide more slots but increase computation time.
        ciphertext_moduli: List of prime factors [q_0, q_1, ..., q_{L}] composing 
                           the ciphertext modulus Q = ∏ q_i
        scaling_factor: Initial scaling factor Δ for precision management
        security_level: Target security level in bits
        batch_size: Number of plaintext slots (poly_degree // 2 for complex)
        level_before_bootstrap: Residual levels kept for bootstrapping
    
    Mathematical Constraints:
        - N must be a power of 2: N = 2^k
        - For complex encoding: slots = N/2 (using roots of unity)
        - Ciphertext modulus Q must be composite for modulus switching
        - Security requires Q < 2^(N * 1.1) for ~128-bit security at N=2^15
    
    Example:
        >>> params = CKKSParameters(
        ...     poly_degree=8192,
        ...     ciphertext_moduli=[0x1FFFFFFFF00001, 0x1FFFFE00001, 0x1FFFFC0001],
        ...     scaling_factor=2**40,
        ...     security_level=SecurityLevel.TC26_128
        ... )
    """
    
    poly_degree: int
    ciphertext_moduli: List[int]
    scaling_factor: int
    security_level: SecurityLevel = SecurityLevel.TC26_128
    level_before_bootstrap: int = 2
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        # Check poly_degree is power of 2
        if self.poly_degree == 0 or (self.poly_degree & (self.poly_degree - 1)) != 0:
            raise ValueError(f"poly_degree must be a power of 2, got {self.poly_degree}")
        
        # Check scaling factor is reasonable
        if self.scaling_factor < 2:
            raise ValueError(f"scaling_factor must be >= 2, got {self.scaling_factor}")
        
        # Validate security level matches parameters
        self._validate_security()
    
    def _validate_security(self):
        """Validate parameters provide claimed security level."""
        log_q = sum(math.log2(p) for p in self.ciphertext_moduli)
        
        # Estimate security (simplified version of HomomorphicEncryption.org standard)
        # For detailed validation, use the estimate_ccastle or EstimateSec library
        if self.security_level == SecurityLevel.TC26_128:
            if log_q > self.poly_degree * 1.1:
                raise ValueError(
                    f"Parameters may not provide 128-bit security. "
                    f"log2(Q) = {log_q:.1f} > N * 1.1 = {self.poly_degree * 1.1}"
                )
        elif self.security_level == SecurityLevel.TC26_192:
            if log_q > self.poly_degree * 1.7:
                raise ValueError(
                    f"Parameters may not provide 192-bit security. "
                    f"log2(Q) = {log_q:.1f} > N * 1.7"
                )
        elif self.security_level == SecurityLevel.TC26_256:
            if log_q > self.poly_degree * 2.2:
                raise ValueError(
                    f"Parameters may not provide 256-bit security. "
                    f"log2(Q) = {log_q:.1f} > N * 2.2"
                )
    
    @property
    def slot_count(self) -> int:
        """Number of plaintext slots for SIMD encoding (complex numbers)."""
        return self.poly_degree // 2
    
    @property
    def max_depth(self) -> int:
        """Maximum multiplicative depth (number of modulus levels)."""
        return len(self.ciphertext_moduli) - 1
    
    @property
    def ciphertext_modulus(self) -> int:
        """Total ciphertext modulus Q = q_0 * q_1 * ... * q_L."""
        result = 1
        for m in self.ciphertext_moduli:
            result *= m
        return result
    
    @property
    def ring_degree(self) -> int:
        """Alias for poly_degree for consistency."""
        return self.poly_degree
    
    def get_modulus_at_level(self, level: int) -> int:
        """
        Get ciphertext modulus at given level.
        
        Args:
            level: Current level (0 = base, max_depth = fresh ciphertext)
        
        Returns:
            Ciphertext modulus q_0 * ... * q_level
        """
        if level < 0 or level > self.max_depth:
            raise ValueError(f"Level must be between 0 and {self.max_depth}")
        
        result = 1
        for i in range(level + 1):
            result *= self.ciphertext_moduli[i]
        return result
    
    def get_residual_modulus(self, level: int) -> int:
        """
        Get modulus for modulus switching (Q_l = Q / q_0 ... q_l).
        
        Args:
            level: Current level
        
        Returns:
            Residual modulus Q_l
        """
        if level < 0 or level > self.max_depth:
            raise ValueError(f"Level must be between 0 and {self.max_depth}")
        
        result = 1
        for i in range(level + 1, len(self.ciphertext_moduli)):
            result *= self.ciphertext_moduli[i]
        return result
    
    @classmethod
    def recommended_128_bit(cls, poly_degree: int = 8192) -> "CKKSParameters":
        """
        Create recommended parameters for ~128-bit security.
        
        Args:
            poly_degree: Polynomial degree (default 8192 for good balance)
        
        Returns:
            CKKSParameters with recommended settings
        
        Note:
            Parameters from HomomorphicEncryption.org standard v1.1
        """
        if poly_degree == 4096:
            # 6 moduli for ~128 bits
            return cls(
                poly_degree=4096,
                ciphertext_moduli=[
                    0xFFFFFE00001,      # 44 bits
                    0x3FFFFC00001,      # 46 bits  
                    0x3FFFF80001,       # 46 bits
                    0x3FFFC00001,       # 46 bits
                    0x3FFF800001,       # 50 bits
                ],
                scaling_factor=2**40,
                security_level=SecurityLevel.TC26_128,
            )
        elif poly_degree == 8192:
            # 7 moduli for ~128 bits
            return cls(
                poly_degree=8192,
                ciphertext_moduli=[
                    0x1FFFFFFFF00001,   # 53 bits
                    0x1FFFFFE00001,     # 49 bits
                    0x1FFFFF00001,      # 45 bits
                    0x1FFFC00001,       # 45 bits
                    0x1FFF800001,       # 45 bits
                    0x1FFF0000001,      # 48 bits
                ],
                scaling_factor=2**40,
                security_level=SecurityLevel.TC26_128,
            )
        elif poly_degree == 16384:
            # 8 moduli for ~128 bits
            return cls(
                poly_degree=16384,
                ciphertext_moduli=[
                    0x3FFFFFFFF00001,  # 55 bits
                    0x3FFFFFFE0001,     # 54 bits
                    0x3FFFFFC00001,     # 50 bits
                    0x3FFFFF800001,     # 50 bits
                    0x3FFFFF0000001,    # 52 bits
                ],
                scaling_factor=2**40,
                security_level=SecurityLevel.TC26_128,
            )
        else:
            raise ValueError(f"No recommended params for poly_degree={poly_degree}")
    
    def __repr__(self) -> str:
        return (
            f"CKKSParameters(N={self.poly_degree}, "
            f"log2(Q)≈{sum(math.log2(p) for p in self.ciphertext_moduli):.1f}, "
            f"slots={self.slot_count}, "
            f"security={self.security_level.name})"
        )


@dataclass 
class BootstrappingParameters:
    """
    Parameters for CKKS bootstrapping (refresh operation).
    
    Bootstrapping allows unlimited multiplicative depth by refreshing
    ciphertexts that have exhausted their levels.
    
    Mathematical Process:
        1. Modulus switching: ct -> ct mod q_i (reduce noise)
        2. Decryption approximation: Compute decryption as polynomial
        3. Re-encryption: Encrypt the approximated plaintext
    
    The key component is the sine/cosine approximation for the modular reduction.
    """
    
    # Number of phases for sine approximation
    phases: int = 8
    
    # Number of coefficients in polynomial approximation
    poly_coeffs: int = 64
    
    # Iterations for Newton's method in evaluation
    newton_iterations: int = 4
    
    # Number of bootstrapping moduli
    bootstrap_moduli: int = 4
    
    # Scaling factor for bootstrapping
    bootstrap_scaling: int = 2**40
    
    # Number of slots to use (must be <= poly_degree / 2)
    bootstrap_slots: Optional[int] = None
    
    def __post_init__(self):
        if self.bootstrap_slots is None:
            self.bootstrap_slots = self.phases * self.newton_iterations


@dataclass 
class EncodingParameters:
    """Parameters for CKKS encoding/decoding."""
    
    # Number of slots to use for encoding
    slot_count: int
    
    # Scaling factor for encoding
    scaling_factor: int
    
    # Polynomial degree
    poly_degree: int
    
    @property
    def complex_slots(self) -> int:
        """Number of complex number slots."""
        return self.slot_count // 2
    
    @property
    def rotation_group_size(self) -> int:
        """Size of rotation group for rotations."""
        return self.slot_count


@dataclass
class KeyGenerationStats:
    """Statistics from key generation for reproducibility."""
    
    poly_degree: int
    security_level: int
    secret_key_hamming_weight: int
    generation_time: float
    seed: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "poly_degree": self.poly_degree,
            "security_bits": self.security_level,
            "secret_hamming_weight": self.secret_key_hamming_weight,
            "generation_time_seconds": self.generation_time,
            "seed": self.seed,
        }
