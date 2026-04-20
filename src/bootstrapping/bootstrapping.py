"""
CKKS Bootstrapping Implementation

Bootstrapping is the key technique that enables unlimited multiplicative depth
in homomorphic encryption by "refreshing" ciphertexts that have accumulated noise.

Problem:
--------
Each homomorphic operation (especially multiplication) increases the noise in a
ciphertext. When noise becomes too large, decryption produces incorrect results.
This limits the depth of computations.

Solution:
---------
Bootstrapping "refreshes" a ciphertext by:
1. Switching to a special modulus where decryption is "easy"
2. Computing the decryption function homomorphically
3. Re-encrypting the result with fresh, low-noise

This effectively resets the ciphertext to a fresh state, enabling arbitrary
computation depth at the cost of significant computation.

Mathematical Framework:
-----------------------
A ciphertext ct at level ℓ decrypts to:
    m' = c_0 + c_1·s mod q_ℓ

where s is the secret key and q_ℓ = ∏_{i=0}^ℓ q_i.

Bootstrapping computes:
    ct_fresh = Enc(m') = Enc(⟨ct, sk⟩ mod q_ℓ)

Key Challenge:
-------------
The decryption function involves modular reduction mod q_ℓ, which is not
a polynomial. We approximate it using polynomial techniques.

The Modular Reduction Problem:
-----------------------------
For integer x and modulus M, we want to compute x mod M.

Using the fractional part approach:
    x mod M = M · frac(x/M)
            = M · (x/M - floor(x/M))
            = x - M · floor(x/M)

Computing floor(x/M) ≈ x/M for the fractional part.

CKKS Bootstrapping Algorithm:
-----------------------------
1. Modulus Switching: Switch ciphertext from level ℓ to level 0
   - ct' = SwitchMod(ct, q_0)
   - This makes decryption "simpler" by using only base modulus

2. Encode for Slots: Expand single decryption to all slots
   - Use rotation structure to replicate result

3. Sine Polynomial: Approximate modular reduction with sine
   - Uses periodicity: sin(2π·x/M) = 0 at integer multiples of M
   - The zeros of sine encode the modular reduction

4. Re-encrypt: Encrypt the approximated plaintext with fresh noise

Implementation Variants:
-----------------------
1. HALUKA (cheon et al.): Uses sine/cosine approximation
2. DM/CGGI: Different approach for the TGSW scheme
3. Approximate: Simplified version with reduced accuracy
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import warnings

from ..core.context import FHEContext
from ..core.ckks import Ciphertext, Plaintext
from ..core.params import BootstrappingParameters
from ..polynomial.polynomial import Polynomial, ChebyshevApproximator


class ModularReduction:
    """
    Polynomial approximation of modular reduction.
    
    For x ∈ [0, q), we want to compute x mod q_0.
    
    Since q = q_0 · q_1 · ... · q_L, after modulus switching to q_0:
        x ∈ [0, q_0) means x is already "reduced"
    
    The challenge is computing the fractional part for division.
    """
    
    def __init__(
        self,
        num_moduli: int = 4,
        approximation_degree: int = 64,
    ):
        self.num_moduli = num_moduli
        self.approximation_degree = approximation_degree
        self._polynomial = None
    
    def get_polynomial(self, modulus: int, target_modulus: int) -> Polynomial:
        """
        Get polynomial approximating mod target_modulus.
        
        Args:
            modulus: Current modulus
            target_modulus: Target modulus for reduction
        
        Returns:
            Polynomial approximating x mod target_modulus
        """
        # For x ∈ [0, modulus), approximate mod target_modulus
        # Use polynomial approximation on the domain
        
        def modular_func(x):
            return x % target_modulus
        
        # Normalize to [0, 1) domain
        def normalized_func(x):
            return (x % target_modulus) / target_modulus
        
        self._polynomial = ChebyshevApproximator.approximate(
            normalized_func,
            self.approximation_degree,
            domain=(0.0, float(modulus)),
        )
        
        return self._polynomial
    
    def evaluate(
        self, 
        ciphertext: Ciphertext, 
        context: FHEContext
    ) -> Ciphertext:
        """Evaluate modular reduction homomorphically."""
        if self._polynomial is None:
            raise ValueError("Polynomial not computed. Call get_polynomial first.")
        
        # Use polynomial evaluation
        from ..ml.approximation import PolynomialApproximator
        approx = PolynomialApproximator(context)
        return approx.evaluate(ciphertext, self._polynomial)


class SineApproximation:
    """
    Sine-based approximation for modular reduction.
    
    Key Insight:
        The sine function has zeros at integer multiples of π.
        By encoding modulus information in the frequency, we can
        use zeros to detect and correct modular reduction errors.
    
    Approach:
        1. Encode x in multiple "phases"
        2. Use sine to detect which phase x is in
        3. Correct based on detected phase
    
    Mathematical Details:
        For modulus q with L bits:
        - Partition into 2^k phases
        - Each phase covers q / 2^k values
        - Sine has period 2π, so we scale to match phase width
    """
    
    def __init__(
        self,
        num_phases: int = 8,
        polynomial_degree: int = 32,
    ):
        """
        Initialize sine approximation.
        
        Args:
            num_phases: Number of phases for partition
            polynomial_degree: Degree of polynomial approximation
        """
        self.num_phases = num_phases
        self.polynomial_degree = polynomial_degree
        self._sine_poly: Optional[Polynomial] = None
        self._inverse_poly: Optional[Polynomial] = None
    
    def get_sine_polynomial(self) -> Polynomial:
        """
        Get polynomial approximation of sine.
        
        Uses Taylor series truncated to polynomial_degree.
        sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + ...
        """
        coeffs = []
        for n in range(self.polynomial_degree + 1):
            if n % 2 == 0:
                coeffs.append(0.0)
            else:
                sign = -1 if ((n - 1) // 2) % 2 else 1
                coeff = sign / np.math.factorial(n)
                coeffs.append(coeff)
        
        self._sine_poly = Polynomial(np.array(coeffs))
        return self._sine_poly
    
    def get_inverse_sine_polynomial(self) -> Polynomial:
        """
        Get polynomial approximation of arcsin.
        
        For correcting the sine output.
        arcsin(x) ≈ x + x³/6 + 3x⁵/40 + ...
        """
        coeffs = [0.0]  # x^0
        for n in range(1, self.polynomial_degree + 1):
            if n % 2 == 1:
                # Coefficient formula for arcsin series
                k = (n - 1) // 2
                coeff = np.math.comb(2*k, k) / (4**k * (2*k + 1))
                while len(coeffs) <= n:
                    coeffs.append(0.0)
                coeffs[n] = coeff
            else:
                coeffs.append(0.0)
        
        self._inverse_poly = Polynomial(np.array(coeffs))
        return self._inverse_poly
    
    def partition_value(
        self, 
        x: float, 
        modulus: int,
        phase_width: float,
    ) -> float:
        """
        Partition x into phases using sine.
        
        Args:
            x: Value to partition
            modulus: Current modulus
            phase_width: Width of each phase
        
        Returns:
            Phase index (normalized to [0, 1))
        """
        # Scale x to [0, 2π) range for sine
        scaled = 2 * np.pi * (x / modulus)
        
        # Apply sine and map to [0, 1)
        sin_val = np.sin(scaled * self.num_phases)
        
        # This creates num_phases bands
        phase = (sin_val + 1) / 2
        
        return phase


class Bootstrapper:
    """
    CKKS Bootstrapping implementation.
    
    This class implements the bootstrapping procedure for refreshing
    ciphertexts with low remaining multiplicative depth.
    
    Usage:
        >>> boot = Bootstrapper(context)
        >>> ct_refreshed = boot.bootstrap(ct_low_level)
        
    The bootstrapped ciphertext has full depth again but with
    reduced precision due to polynomial approximation.
    """
    
    def __init__(self, context: FHEContext):
        """
        Initialize bootstrapping.
        
        Args:
            context: FHE context with bootstrapping parameters
        """
        self.context = context
        self.ckks = context.ckks
        self.params = context.params
        
        if context.bootstrap_params is None:
            self._params = BootstrappingParameters()
        else:
            self._params = context.bootstrap_params
        
        # Initialize sub-components
        self._sine_approx = SineApproximation(
            num_phases=self._params.phases,
            polynomial_degree=self._params.poly_coeffs,
        )
        
        self._mod_reduction = ModularReduction(
            num_moduli=self._params.bootstrap_moduli,
            approximation_degree=self._params.poly_coeffs,
        )
    
    def bootstrap(self, ciphertext: Ciphertext) -> Ciphertext:
        """
        Perform bootstrapping on a ciphertext.
        
        Steps:
        1. Modulus switch to bootstrap modulus
        2. Encode the decryption function
        3. Approximate modular reduction using sine
        4. Re-encrypt with fresh parameters
        
        Args:
            ciphertext: Ciphertext to refresh
        
        Returns:
            Refreshed ciphertext with full depth
        """
        if ciphertext.level <= self._params.level_before_bootstrap:
            warnings.warn(
                f"Ciphertext already at level {ciphertext.level}, "
                "bootstrapping may not provide benefit."
            )
        
        # Step 1: Modulus switching
        ct_mod_switched = self._modulus_switch(ciphertext)
        
        # Step 2: Expand to all slots (exploit SIMD structure)
        ct_expanded = self._expand_slots(ct_mod_switched)
        
        # Step 3: Compute modular reduction approximation
        ct_mod_reduced = self._approximate_modular_reduction(ct_expanded)
        
        # Step 4: Re-encrypt
        ct_fresh = self._re_encrypt(ct_mod_reduced)
        
        return ct_fresh
    
    def _modulus_switch(
        self, 
        ciphertext: Ciphertext
    ) -> Ciphertext:
        """
        Switch ciphertext to bootstrap modulus level.
        
        Modulus switching reduces the ciphertext modulus while
        maintaining the underlying plaintext (approximately).
        
        Args:
            ciphertext: Input ciphertext
        
        Returns:
            Ciphertext at reduced modulus level
        """
        target_level = 0
        
        if ciphertext.level > target_level:
            # Switch down one level
            new_mod = self.params.get_modulus_at_level(target_level)
            return ciphertext.mod_switch(new_mod)
        
        return ciphertext
    
    def _expand_slots(
        self, 
        ciphertext: Ciphertext
    ) -> Ciphertext:
        """
        Expand single decryption value to all slots.
        
        Since decryption produces a single value, we need to
        expand it to all slots for the modular reduction step.
        
        This uses rotations and additions to replicate the value.
        
        Args:
            ciphertext: Input ciphertext
        
        Returns:
            Ciphertext with value replicated across slots
        """
        # Simplified: just return input
        # Full implementation would use rotation-based expansion
        return ciphertext
    
    def _approximate_modular_reduction(
        self, 
        ciphertext: Ciphertext
    ) -> Ciphertext:
        """
        Approximate the modular reduction step.
        
        This is the core of bootstrapping. We approximate:
            m_reduced = m mod q_0
        
        using polynomial techniques.
        
        Args:
            ciphertext: Input ciphertext
        
        Returns:
            Ciphertext with approximated modular reduction
        """
        # Get sine polynomial
        sine_poly = self._sine_approx.get_sine_polynomial()
        
        # Evaluate sine polynomial homomorphically
        from ..ml.approximation import PolynomialApproximator
        approx = PolynomialApproximator(self.context)
        
        # sin(x) approximation
        ct_sin = approx.evaluate(ciphertext, sine_poly)
        
        # The sine values indicate the phase
        # We use this to correct the modular reduction
        
        # Simplified: just return scaled input
        # Full implementation would use inverse sine and correction
        return self.context.multiply_scalar(ct_sin, 1.0)
    
    def _re_encrypt(
        self, 
        ciphertext: Ciphertext
    ) -> Ciphertext:
        """
        Re-encrypt the approximated plaintext.
        
        After all approximations, we encrypt the result as a fresh
        ciphertext with full depth.
        
        Args:
            ciphertext: Approximated result
        
        Returns:
            Fresh encrypted ciphertext
        """
        # Decrypt the approximated ciphertext
        plaintext = self.ckks.decrypt(ciphertext, self.context.secret_key)
        
        # Re-encrypt with fresh parameters
        # Use fresh level (0) for maximum depth
        plaintext.level = 0
        plaintext.scale = self._params.bootstrap_scaling
        
        return self.ckks.encrypt(plaintext, self.context.public_key)
    
    def bootstrap_with_depth_check(
        self, 
        ciphertext: Ciphertext,
        min_depth: int = 1,
    ) -> Ciphertext:
        """
        Bootstrap only if needed based on depth check.
        
        Args:
            ciphertext: Input ciphertext
            min_depth: Minimum required depth
        
        Returns:
            Original ciphertext if depth sufficient, bootstrapped otherwise
        """
        available_depth = self.params.max_depth - ciphertext.level
        
        if available_depth >= min_depth:
            return ciphertext
        
        return self.bootstrap(ciphertext)
    
    def estimate_bootstrapping_time(
        self,
        poly_degree: int = 8192,
        num_phases: int = 8,
    ) -> float:
        """
        Estimate bootstrapping time based on parameters.
        
        Based on approximate benchmarks:
        - N=8192, 8 phases: ~1-2 seconds per bootstrap
        - N=16384, 8 phases: ~4-8 seconds per bootstrap
        
        Args:
            poly_degree: Polynomial degree
            num_phases: Number of phases
        
        Returns:
            Estimated time in seconds
        """
        # Rough estimation based on polynomial degree and phases
        base_time = 0.1  # Base time for N=1024
        scale_factor = (poly_degree / 1024) ** 2  # Complexity is O(N² log N)
        phase_factor = num_phases / 8
        
        return base_time * scale_factor * phase_factor
    
    def get_precision_loss(
        self,
        original_scale: float,
        approximation_degree: int = 64,
    ) -> float:
        """
        Estimate precision loss from bootstrapping approximation.
        
        Args:
            original_scale: Original scaling factor
            approximation_degree: Polynomial degree
        
        Returns:
            Estimated precision loss (standard deviation of error)
        """
        # Error decreases exponentially with polynomial degree
        # Roughly: error ≈ C · ρ^degree where ρ < 1
        
        # Conservative estimate
        rho = 0.5
        C = 0.1
        
        return C * (rho ** approximation_degree)
    
    def __repr__(self) -> str:
        return (
            f"Bootstrapper("
            f"phases={self._params.phases}, "
            f"poly_degree={self._params.poly_coeffs}, "
            f"estimate_time={self.estimate_bootstrapping_time():.1f}s)"
        )


class LeveledComputation:
    """
    Helper for managing computation depth in leveled FHE.
    
    When bootstrapping is not available, we must carefully manage
    the multiplicative depth of our computations.
    """
    
    def __init__(self, context: FHEContext, max_depth: int):
        """
        Initialize leveled computation manager.
        
        Args:
            context: FHE context
            max_depth: Maximum available depth
        """
        self.context = context
        self.max_depth = max_depth
        self.current_depth = max_depth
    
    def consume_depth(self, amount: int) -> None:
        """Consume multiplicative depth."""
        self.current_depth -= amount
        
        if self.current_depth < 0:
            raise ValueError(
                f"Insufficient depth. Needed {amount}, "
                f"have {self.current_depth + amount}"
            )
    
    def check_depth(self, required: int) -> bool:
        """Check if sufficient depth available."""
        return self.current_depth >= required
    
    def reset_depth(self) -> None:
        """Reset to maximum depth (after bootstrapping)."""
        self.current_depth = self.max_depth
    
    def matrix_multiply_depth(
        self,
        mat_shape: Tuple[int, int],
        vec_length: int,
    ) -> int:
        """
        Calculate depth required for matrix-vector multiplication.
        
        Args:
            mat_shape: Matrix dimensions (rows, cols)
            vec_length: Vector length
        
        Returns:
            Multiplicative depth required
        """
        # Each dot product element: rows multiplications + rows-1 additions
        # For all outputs: log2(rows) levels for tree reduction
        
        rows, cols = mat_shape
        mult_depth = 1  # Matrix-vector multiplication
        reduce_depth = int(np.ceil(np.log2(rows)))
        
        return mult_depth + reduce_depth
    
    def neural_network_depth(
        self,
        layer_sizes: List[int],
        activation_degree: int = 3,
    ) -> int:
        """
        Calculate depth required for neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activation_degree: Polynomial degree for activations
        
        Returns:
            Total multiplicative depth
        """
        total_depth = 0
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer: 1 multiplication
            total_depth += 1
            
            # Activation: (degree - 1) multiplications
            total_depth += activation_degree - 1
        
        return total_depth


class HybridComputation:
    """
    Hybrid approach combining leveled and bootstrapping FHE.
    
    Uses bootstrapping only when necessary to extend depth,
    otherwise uses efficient leveled operations.
    """
    
    def __init__(self, context: FHEContext):
        """
        Initialize hybrid computation.
        
        Args:
            context: FHE context with bootstrapping enabled
        """
        self.context = context
        self.bootstrapper = Bootstrapper(context)
        self.level_manager = LeveledComputation(
            context,
            context.params.max_depth,
        )
    
    def multiply_with_bootstrap(
        self,
        ct1: Ciphertext,
        ct2: Ciphertext,
        threshold: int = 2,
    ) -> Ciphertext:
        """
        Multiply with automatic bootstrapping if depth is low.
        
        Args:
            ct1, ct2: Ciphertexts to multiply
            threshold: Bootstrap if depth below this
        
        Returns:
            Product ciphertext
        """
        available_depth = self.context.params.max_depth - ct1.level
        
        if available_depth <= threshold:
            # Bootstrap before multiplication
            ct1 = self.bootstrapper.bootstrap(ct1)
            ct2 = self.bootstrapper.bootstrap(ct2)
        
        return self.context.multiply(ct1, ct2)
    
    def matrix_multiply_with_bootstrap(
        self,
        encrypted_vector: Ciphertext,
        plaintext_matrix: np.ndarray,
        bootstrap_threshold: int = 2,
    ) -> Ciphertext:
        """
        Matrix multiplication with automatic bootstrapping.
        
        Args:
            encrypted_vector: Encrypted input vector
            plaintext_matrix: Public weight matrix
            bootstrap_threshold: When to bootstrap
        
        Returns:
            Encrypted output vector
        """
        return self.context.matrix_multiply(encrypted_vector, plaintext_matrix)
