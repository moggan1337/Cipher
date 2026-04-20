"""
CKKS Core Implementation

This module implements the core CKKS (Cheon-Kim-Kim-Song) homomorphic encryption
scheme for encrypted approximate arithmetic.

Mathematical Background:
------------------------
CKKS operates on the ring R = Z[X]/(X^N + 1) where N is a power of 2.

Key Components:
1. Encoding: Complex vectors ↔ Polynomials (with scaling)
2. Encryption: Plaintexts ↔ Ciphertexts (using RLWE)
3. Operations: Addition, Multiplication, Rotation, Rescaling
4. Decryption: Ciphertexts → Plaintexts → Complex vectors

Encryption Scheme:
- Plaintext: m(X) + v(X) where m is message polynomial, v is noise
- Ciphertext: ct = (c_0, c_1) where:
  c_0 = a·sk + m + e_0 (mod q)
  c_1 = b·sk + e_1 (mod q)
  where (a, b) is the public key

Multiplication:
- ct_mult = ct_1 ⊙ ct_2 = (c_0¹·c_0², c_0¹·c_1² + c_1¹·c_0², c_1¹·c_1²)
- Result is degree-2, needs relinearization
- After relin: (c_0' + c_1'·sk, c_1') where c_1' is decomposition of c_1

Rescaling:
- After each multiplication, scale by Δ and round
- This reduces the scaling factor and removes some noise
- Essentially: ⌊ct/Δ⌉
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Union, Callable
from dataclasses import dataclass, field
import cmath
import warnings

from .params import CKKSParameters, EncodingParameters
from .keys import SecretKey, PublicKey, EvaluationKey


@dataclass
class Plaintext:
    """
    CKKS Plaintext polynomial.
    
    A plaintext is a polynomial in the message space with associated scaling.
    """
    
    coefficients: NDArray[np.float64]
    scale: float
    params: CKKSParameters
    level: int = 0  # Current modulus level
    
    @property
    def poly_degree(self) -> int:
        return len(self.coefficients)
    
    def rescale(self, factor: float) -> "Plaintext":
        """Rescale plaintext by given factor."""
        return Plaintext(
            coefficients=self.coefficients / factor,
            scale=self.scale / factor,
            params=self.params,
            level=self.level,
        )
    
    def __repr__(self) -> str:
        return f"Plaintext(N={self.poly_degree}, scale={self.scale:.2e}, level={self.level})"


@dataclass
class Ciphertext:
    """
    CKKS Ciphertext.
    
    A ciphertext is a pair of polynomials (c_0, c_1) ∈ R_q × R_q.
    
    Mathematical Invariant:
        After decryption: m ≈ c_0 + c_1·sk (mod q)
        The approximation error grows with homomorphic operations.
    """
    
    c0: NDArray[np.int64]
    c1: NDArray[np.int64]
    scale: float
    params: CKKSParameters
    level: int = 0
    
    def __post_init__(self):
        if len(self.c0) != self.params.poly_degree:
            raise ValueError(f"c0 length mismatch: expected {self.params.poly_degree}")
        if len(self.c1) != self.params.poly_degree:
            raise ValueError(f"c1 length mismatch: expected {self.params.poly_degree}")
    
    @property
    def degree(self) -> int:
        """Ciphertext degree (1 for fresh, 2 after multiplication)."""
        return 2 if self.c2 is not None else 1
    
    @property
    def c2(self) -> Optional[NDArray[np.int64]]:
        """Second polynomial for degree-2 ciphertexts."""
        return getattr(self, '_c2', None)
    
    @c2.setter
    def c2(self, value):
        self._c2 = value
    
    @property
    def is_relinearized(self) -> bool:
        """Whether ciphertext is relinearized (degree 1)."""
        return self.c2 is None
    
    def rescale(self, factor: float, mod: int) -> "Ciphertext":
        """
        Rescale ciphertext by dividing by scaling factor.
        
        Args:
            factor: Scaling factor to divide by
            mod: New modulus for the level
        
        Returns:
            Rescaled ciphertext
        """
        new_c0 = np.round(self.c0 / factor).astype(np.int64)
        new_c1 = np.round(self.c1 / factor).astype(np.int64)
        
        # Apply modulus
        if mod != self.params.get_modulus_at_level(self.level):
            scale_ratio = mod / self.params.get_modulus_at_level(self.level)
            new_c0 = np.mod(new_c0, mod)
            new_c1 = np.mod(new_c1, mod)
        
        return Ciphertext(
            c0=new_c0,
            c1=new_c1,
            scale=self.scale / factor,
            params=self.params,
            level=self.level + 1,
        )
    
    def mod_switch(self, new_mod: int) -> "Ciphertext":
        """Switch to a different modulus (for modulus chaining)."""
        new_c0 = np.mod(self.c0, new_mod)
        new_c1 = np.mod(self.c1, new_mod)
        
        return Ciphertext(
            c0=new_c0,
            c1=new_c1,
            scale=self.scale,
            params=self.params,
            level=self.level,
        )
    
    def __repr__(self) -> str:
        deg = self.degree
        return f"Ciphertext(degree={deg}, scale={self.scale:.2e}, level={self.level})"


class CKKS:
    """
    CKKS Homomorphic Encryption Scheme Implementation.
    
    This class implements the core encryption, decryption, and homomorphic
    operations for the CKKS scheme.
    
    Usage:
        >>> params = CKKSParameters.recommended_128_bit()
        >>> ckks = CKKS(params)
        >>> # Encode, encrypt, operate, decrypt, decode
    """
    
    def __init__(self, params: CKKSParameters, rng_seed: Optional[int] = None):
        """
        Initialize CKKS scheme.
        
        Args:
            params: CKKS parameters
            rng_seed: Random seed for reproducibility
        """
        self.params = params
        self.rng = np.random.default_rng(rng_seed)
        self.encoding_params = EncodingParameters(
            slot_count=params.slot_count,
            scaling_factor=params.scaling_factor,
            poly_degree=params.poly_degree,
        )
    
    # =========================================================================
    # Encoding and Decoding
    # =========================================================================
    
    def encode(
        self, 
        values: np.ndarray, 
        scale: Optional[float] = None,
        level: int = 0
    ) -> Plaintext:
        """
        Encode complex/real values into a CKKS plaintext polynomial.
        
        Args:
            values: Complex or real array of values to encode
            scale: Scaling factor (default: use params.scaling_factor)
            level: Encoding level
        
        Returns:
            Plaintext polynomial
        
        Mathematical Process:
            1. Pad values to slot_count (simulate N/2 complex slots)
            2. Map to N-th roots of unity domain via inverse FFT
            3. Scale by Δ
            4. Round to integers
            5. Embed as polynomial coefficients
        """
        n = self.params.poly_degree
        slots = self.params.slot_count
        
        if scale is None:
            scale = self.params.scaling_factor
        
        # Ensure values fit in slots
        if len(values) > slots:
            warnings.warn(f"Values truncated from {len(values)} to {slots} slots")
            values = values[:slots]
        
        # Pad to slot count
        padded = np.zeros(slots, dtype=np.complex128)
        padded[:len(values)] = values
        
        # Scale the values
        scaled = padded * scale
        
        # Round to integers
        rounded = np.round(scaled.real).astype(np.int64)
        
        # Create polynomial via inverse NTT-like transform
        # Use FFT to map to polynomial coefficients
        # For ring (X^N + 1), we need specific mapping
        
        # Simplified: use FFT-based encoding
        coeffs = self._encode_fft(rounded, n)
        
        return Plaintext(
            coefficients=coeffs,
            scale=scale,
            params=self.params,
            level=level,
        )
    
    def _encode_fft(self, values: np.ndarray, n: int) -> NDArray[np.float64]:
        """FFT-based encoding for CKKS."""
        # For CKKS encoding, we embed complex values at roots of unity
        # Simplified version using standard FFT
        
        # Pad to n/2 (complex) then use full n via conjugate symmetry
        half_n = n // 2
        
        # Create input for inverse FFT
        fft_input = np.zeros(n, dtype=np.complex128)
        fft_input[:len(values)] = values
        
        # Inverse FFT to get polynomial coefficients
        coeffs = np.fft.ifft(fft_input) * n
        
        return np.real(coeffs)
    
    def decode(self, plaintext: Plaintext) -> np.ndarray:
        """
        Decode plaintext polynomial back to complex values.
        
        Args:
            plaintext: Plaintext to decode
        
        Returns:
            Complex array of decoded values
        """
        n = self.params.poly_degree
        slots = self.params.slot_count
        
        # Extract values via FFT
        coeffs = plaintext.coefficients * plaintext.scale
        
        # Forward FFT
        fft_result = np.fft.fft(coeffs) / n
        
        # Take first slots
        values = fft_result[:slots]
        
        return values
    
    # =========================================================================
    # Encryption and Decryption  
    # =========================================================================
    
    def encrypt(self, plaintext: Plaintext, public_key: PublicKey) -> Ciphertext:
        """
        Encrypt a plaintext using the public key.
        
        Args:
            plaintext: Plaintext to encrypt
            public_key: Public key for encryption
        
        Returns:
            Ciphertext
        
        Mathematical Process:
            Sample random polynomial u from R_q
            Sample error polynomials e_0, e_1 from discrete Gaussian
            Compute:
                c_0 = b·u + m + e_0 (mod q)
                c_1 = a·u + e_1 (mod q)
        """
        n = self.params.poly_degree
        q = self.params.get_modulus_at_level(plaintext.level)
        
        # Sample random u
        u = self.rng.integers(0, 2, size=n, dtype=np.int64)  # Ternary
        
        # Sample error polynomials
        sigma = 3.19
        e0 = np.round(self.rng.normal(0, sigma, size=n)).astype(np.int64)
        e1 = np.round(self.rng.normal(0, sigma, size=n)).astype(np.int64)
        
        # Get message polynomial coefficients
        m = np.round(plaintext.coefficients).astype(np.int64)
        
        # c_0 = b·u + m + e_0
        b_u = self._poly_mult(public_key.b, u, q)
        c0 = np.mod(b_u + m + e0, q)
        
        # c_1 = a·u + e_1
        a_u = self._poly_mult(public_key.a, u, q)
        c1 = np.mod(a_u + e1, q)
        
        return Ciphertext(
            c0=c0,
            c1=c1,
            scale=plaintext.scale,
            params=self.params,
            level=plaintext.level,
        )
    
    def decrypt(self, ciphertext: Ciphertext, secret_key: SecretKey) -> Plaintext:
        """
        Decrypt a ciphertext using the secret key.
        
        Args:
            ciphertext: Ciphertext to decrypt
            secret_key: Secret key for decryption
        
        Returns:
            Plaintext polynomial
        
        Mathematical Process:
            m' = c_0 + c_1·sk (mod q)
            For degree-2 ciphertexts, include c_2 term.
        """
        q = self.params.get_modulus_at_level(ciphertext.level)
        n = self.params.poly_degree
        
        # Compute c_0 + c_1·sk
        c1_sk = self._poly_mult(ciphertext.c1, secret_key.coefficients, q)
        m = np.mod(ciphertext.c0 + c1_sk, q)
        
        # Handle degree-2 ciphertext
        if ciphertext.c2 is not None:
            c2_sk = self._poly_mult(ciphertext.c2, secret_key.coefficients, q)
            m = np.mod(m + c2_sk, q)
        
        # Convert to float coefficients for decoding
        m_float = m.astype(np.float64) / ciphertext.scale
        
        return Plaintext(
            coefficients=m_float,
            scale=ciphertext.scale,
            params=self.params,
            level=ciphertext.level,
        )
    
    # =========================================================================
    # Homomorphic Operations
    # =========================================================================
    
    def add(
        self, 
        ct1: Ciphertext, 
        ct2: Ciphertext
    ) -> Ciphertext:
        """
        Homomorphic addition of two ciphertexts.
        
        Args:
            ct1, ct2: Ciphertexts to add
        
        Returns:
            ct1 + ct2
        
        Mathematical:
            ct_add = (c0_1 + c0_2, c1_1 + c1_2) mod q
        """
        if ct1.level != ct2.level:
            raise ValueError("Cannot add ciphertexts at different levels")
        
        q = self.params.get_modulus_at_level(ct1.level)
        
        c0_sum = np.mod(ct1.c0 + ct2.c0, q)
        c1_sum = np.mod(ct1.c1 + ct2.c1, q)
        
        return Ciphertext(
            c0=c0_sum,
            c1=c1_sum,
            scale=max(ct1.scale, ct2.scale),
            params=self.params,
            level=ct1.level,
        )
    
    def add_plain(self, ciphertext: Ciphertext, plaintext: Plaintext) -> Ciphertext:
        """
        Add plaintext to ciphertext.
        
        Args:
            ciphertext: Ciphertext
            plaintext: Plaintext to add
        
        Returns:
            ciphertext + plaintext
        """
        if ciphertext.level != plaintext.level:
            raise ValueError("Level mismatch")
        
        q = self.params.get_modulus_at_level(ciphertext.level)
        m = np.round(plaintext.coefficients * ciphertext.scale).astype(np.int64)
        
        c0_sum = np.mod(ciphertext.c0 + m, q)
        
        return Ciphertext(
            c0=c0_sum,
            c1=ciphertext.c1.copy(),
            scale=ciphertext.scale,
            params=self.params,
            level=ciphertext.level,
        )
    
    def multiply(
        self, 
        ct1: Ciphertext, 
        ct2: Ciphertext,
        eval_key: Optional[EvaluationKey] = None,
        relinearize: bool = True
    ) -> Ciphertext:
        """
        Homomorphic multiplication of two ciphertexts.
        
        Args:
            ct1, ct2: Ciphertexts to multiply
            eval_key: Evaluation key for relinearization
            relinearize: Whether to relinearize after multiplication
        
        Returns:
            ct1 × ct2
        
        Mathematical:
            For fresh ciphertexts (degree 1):
                ct_mult = ct1 ⊙ ct2
                    = (c0_1·c0_2, c0_1·c1_2 + c1_1·c0_2, c1_1·c1_2)
            
            After relinearization (degree 1 again):
                ct_relin = (c0' + c1'·sk, c1')
                where c1' = decompose(c1_1·c1_2)
        """
        if ct1.level != ct2.level:
            raise ValueError("Cannot multiply ciphertexts at different levels")
        
        q = self.params.get_modulus_at_level(ct1.level)
        
        # Degree-2 multiplication
        c0_mult = self._poly_mult(ct1.c0, ct2.c0, q)
        c1_mult = np.mod(
            self._poly_mult(ct1.c0, ct2.c1, q) + 
            self._poly_mult(ct1.c1, ct2.c0, q), 
            q
        )
        c2 = self._poly_mult(ct1.c1, ct2.c1, q)
        
        # Create degree-2 ciphertext
        result = Ciphertext(
            c0=c0_mult,
            c1=c1_mult,
            scale=ct1.scale * ct2.scale,
            params=self.params,
            level=ct1.level,
        )
        result.c2 = c2
        
        # Relinearize if key provided and requested
        if relinearize and eval_key is not None:
            result = self.relinearize(result, eval_key)
        
        return result
    
    def multiply_plain(self, ciphertext: Ciphertext, plaintext: Plaintext) -> Ciphertext:
        """
        Multiply ciphertext by plaintext (no relinearization needed).
        
        Args:
            ciphertext: Ciphertext
            plaintext: Plaintext scalar
        
        Returns:
            ciphertext × plaintext
        """
        q = self.params.get_modulus_at_level(ciphertext.level)
        
        # Scale plaintext polynomial
        m = np.round(plaintext.coefficients * ciphertext.scale).astype(np.int64)
        
        c0_mult = self._poly_mult(ciphertext.c0, m, q)
        c1_mult = self._poly_mult(ciphertext.c1, m, q)
        
        return Ciphertext(
            c0=c0_mult,
            c1=c1_mult,
            scale=ciphertext.scale * plaintext.scale,
            params=self.params,
            level=ciphertext.level,
        )
    
    def relinearize(self, ciphertext: Ciphertext, eval_key: EvaluationKey) -> Ciphertext:
        """
        Relinearize a degree-2 ciphertext to degree-1.
        
        Args:
            ciphertext: Degree-2 ciphertext
            eval_key: Relinearization key
        
        Returns:
            Relinearized degree-1 ciphertext
        
        Mathematical:
            Input: (c0, c1, c2) with decryption m = c0 + c1·s + c2·s²
            Output: (c0', c1') with decryption m ≈ c0' + c1'·s
            
            Use gadget decomposition: c2 = Σ b_i · g_i
            Then: c0' = c0 + Σ b_i · rlk_0[i]
                  c1' = c1 + Σ b_i · rlk_1[i]
        """
        if ciphertext.c2 is None:
            return ciphertext  # Already relinearized
        
        q = self.params.get_modulus_at_level(ciphertext.level)
        n = self.params.poly_degree
        
        # Decompose c2 using gadget vector
        gadget_dim = len(eval_key.relin_keys)
        base = 2 ** 16  # Gadget base (from key generation)
        
        # Simplified decomposition
        c2_scaled = ciphertext.c2.astype(np.int64)
        
        # Apply relinearization
        c0_new = ciphertext.c0.copy()
        c1_new = ciphertext.c1.copy()
        
        # Simple relinearization (simplified BSGS decomposition)
        for i, (rlk0, rlk1) in enumerate(eval_key.relin_keys):
            if i * len(c2_scaled) // gadget_dim < len(c2_scaled):
                start = i * len(c2_scaled) // gadget_dim
                end = (i + 1) * len(c2_scaled) // gadget_dim
                chunk = c2_scaled[start:end]
                
                if len(chunk) > 0:
                    # Apply relin key contribution
                    chunk_mod = np.mod(chunk, q)
                    c0_new = np.mod(c0_new + chunk_mod * rlk0[:len(chunk)], q)
                    c1_new = np.mod(c1_new + chunk_mod * rlk1[:len(chunk)], q)
        
        result = Ciphertext(
            c0=c0_new,
            c1=c1_new,
            scale=ciphertext.scale,
            params=self.params,
            level=ciphertext.level,
        )
        
        return result
    
    def rescale(self, ciphertext: Ciphertext, delta: Optional[float] = None) -> Ciphertext:
        """
        Rescale ciphertext by dividing by scaling factor.
        
        This is critical for managing noise growth. After each multiplication,
        the scaling factor grows. Rescaling reduces it back while removing noise.
        
        Args:
            ciphertext: Ciphertext to rescale
            delta: Rescaling factor (default: ciphertext.scale)
        
        Returns:
            Rescaled ciphertext
        """
        if delta is None:
            delta = ciphertext.scale
        
        q_new = self.params.get_modulus_at_level(ciphertext.level + 1)
        q_old = self.params.get_modulus_at_level(ciphertext.level)
        
        # Round and reduce modulus
        c0_scaled = np.round(ciphertext.c0 / delta).astype(np.int64)
        c1_scaled = np.round(ciphertext.c1 / delta).astype(np.int64)
        
        # Modulus switch to lower level
        c0_new = np.mod(c0_scaled, q_new)
        c1_new = np.mod(c1_scaled, q_new)
        
        return Ciphertext(
            c0=c0_new,
            c1=c1_new,
            scale=ciphertext.scale / delta,
            params=self.params,
            level=ciphertext.level + 1,
        )
    
    def rotate(self, ciphertext: Ciphertext, steps: int, eval_key: EvaluationKey) -> Ciphertext:
        """
        Rotate ciphertext slots cyclically.
        
        Args:
            ciphertext: Ciphertext to rotate
            steps: Number of positions to rotate (positive = left)
            eval_key: Rotation key
        
        Returns:
            Rotated ciphertext
        
        Mathematical:
            Rotation corresponds to automorphism X -> X·ζ^k
            where ζ is a primitive 2N-th root of unity.
        """
        n = self.params.poly_degree
        slots = self.params.slot_count
        q = self.params.get_modulus_at_level(ciphertext.level)
        
        # Normalize steps
        steps = steps % slots
        
        if steps == 0:
            return ciphertext
        
        # Get rotation key
        rot_key = eval_key.get_rotation_key(steps)
        if rot_key is None:
            raise ValueError(f"No rotation key for step {steps}")
        
        k0, k1 = rot_key
        
        # Apply rotation to ciphertext polynomials
        # Simplified: use cyclic polynomial rotation
        c0_rot = np.roll(ciphertext.c0, steps)
        c1_rot = np.roll(ciphertext.c1, steps)
        
        return Ciphertext(
            c0=c0_rot,
            c1=c1_rot,
            scale=ciphertext.scale,
            params=self.params,
            level=ciphertext.level,
        )
    
    def negate(self, ciphertext: Ciphertext) -> Ciphertext:
        """Negate a ciphertext."""
        q = self.params.get_modulus_at_level(ciphertext.level)
        
        return Ciphertext(
            c0=np.mod(-ciphertext.c0, q),
            c1=np.mod(-ciphertext.c1, q),
            scale=ciphertext.scale,
            params=self.params,
            level=ciphertext.level,
        )
    
    def sub(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Subtract ciphertexts."""
        return self.add(ct1, self.negate(ct2))
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _poly_mult(
        self, 
        a: NDArray[np.int64], 
        b: NDArray[np.int64], 
        modulus: int
    ) -> NDArray[np.int64]:
        """
        Polynomial multiplication in Z_mod[X]/(X^N + 1).
        
        Uses the property that X^N ≡ -1, so:
            a(X)·b(X) mod (X^N + 1) = Σ a_i·b_{j-i} - Σ a_i·b_{N+j-i} for wrapped terms
        """
        n = len(a)
        
        # Use convolution
        result = np.convolve(a, b, mode='full')
        
        # Handle wrap-around for X^N = -1
        if len(result) > n:
            high = result[n:]
            low = result[:n]
            # Wrap high terms with sign change (X^N = -1)
            result = np.concatenate([low - high, [0] * (n - len(low) + len(high))])[:n]
            result[:len(low)] = np.mod(low - high[:len(low)], modulus)
        
        return np.mod(result, modulus)
    
    def square(self, ciphertext: Ciphertext, eval_key: Optional[EvaluationKey] = None) -> Ciphertext:
        """Square a ciphertext."""
        return self.multiply(ciphertext, ciphertext, eval_key)
    
    def norm(self, ciphertext: Ciphertext) -> float:
        """Compute approximate norm of underlying plaintext."""
        plaintext = self.decrypt(ciphertext, self._dummy_secret_key)
        return np.linalg.norm(plaintext.coefficients)
    
    @property
    def slot_count(self) -> int:
        """Number of plaintext slots."""
        return self.params.slot_count
    
    def __repr__(self) -> str:
        return f"CKKS(N={self.params.poly_degree}, slots={self.params.slot_count})"
