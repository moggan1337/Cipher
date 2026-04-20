"""
Polynomial Approximation of Activation Functions

This module provides automated polynomial approximation of non-polynomial
activation functions for use in homomorphic encryption contexts.

Challenge:
----------
CKKS supports addition and multiplication natively. However, neural network
activations like ReLU, Sigmoid, and Tanh are non-polynomial functions.

Solution:
---------
Approximate these functions with polynomials that:
1. Are accurate within the relevant domain
2. Have low degree to minimize depth usage
3. Can be evaluated homomorphically

Methods:
--------
1. Chebyshev Approximation (recommended): Minimax optimal on [-1, 1]
2. Taylor Series: Simple but less accurate
3. Piecewise Polynomial: Better accuracy with same degree

For encrypted inference, we typically:
- Use degree 3-5 polynomials for speed
- Use degree 7-11 for better accuracy
- Can use bootstrapping between polynomial evaluations
"""

from typing import Tuple, Optional, Dict, List, Callable, Any
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from ..core.context import FHEContext
from ..core.ckks import Ciphertext, Plaintext
from ..polynomial.polynomial import Polynomial, ChebyshevApproximator, TaylorApproximator


@dataclass
class ApproximationConfig:
    """Configuration for polynomial approximation."""
    
    # Degree of polynomial
    degree: int = 3
    
    # Domain for approximation [min, max]
    domain: Tuple[float, float] = (-1.0, 1.0)
    
    # Method: 'chebyshev', 'taylor', or 'best_uniform'
    method: str = 'chebyshev'
    
    # For piecewise approximation
    num_pieces: int = 1
    
    # Error bound (if using best_uniform)
    error_tolerance: Optional[float] = None
    
    # Number of pieces for piecewise approximation
    piece_boundaries: Optional[List[float]] = None


class ActivationApproximator:
    """
    Base class for activation function approximation.
    
    Provides common utilities for polynomial approximation of activations.
    """
    
    # Approximation coefficients cache
    _cache: Dict[str, Polynomial] = {}
    
    @classmethod
    def get_polynomial(
        cls,
        name: str,
        config: ApproximationConfig,
    ) -> Polynomial:
        """
        Get or compute polynomial approximation.
        
        Args:
            name: Activation name (relu, sigmoid, tanh, etc.)
            config: Approximation configuration
        
        Returns:
            Polynomial approximation
        """
        cache_key = f"{name}_{config.degree}_{config.method}_{config.domain}"
        
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        # Compute approximation based on activation
        func = cls._get_activation_function(name)
        
        if config.method == 'chebyshev':
            poly = ChebyshevApproximator.approximate(
                func,
                config.degree,
                config.domain,
            )
        elif config.method == 'taylor':
            poly = TaylorApproximator.approximate(
                func,
                config.degree,
                center=0.0,
            )
        else:
            from ..polynomial.polynomial import BestUniformApproximator
            poly, _ = BestUniformApproximator.approximate(
                func,
                config.degree,
                config.domain,
            )
        
        cls._cache[cache_key] = poly
        return poly
    
    @staticmethod
    def _get_activation_function(name: str) -> Callable[[float], float]:
        """Get activation function by name."""
        activations = {
            'relu': lambda x: max(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
            'elu': lambda x: x if x >= 0 else np.exp(x) - 1,
            'selu': lambda x: 1.0507 * x if x >= 0 else 1.0507 * 1.67326 * (np.exp(x) - 1),
            'softplus': lambda x: np.log(1 + np.exp(x)),
            'leaky_relu': lambda x: x if x >= 0 else 0.01 * x,
            'swish': lambda x: x * 1 / (1 + np.exp(-x)),
            'mish': lambda x: x * np.tanh(np.log(1 + np.exp(x))),
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        
        return activations[name.lower()]
    
    @classmethod
    def list_activations(cls) -> List[str]:
        """List available activation functions."""
        return ['relu', 'sigmoid', 'tanh', 'gelu', 'elu', 'selu', 
                'softplus', 'leaky_relu', 'swish', 'mish']


class PolynomialApproximator:
    """
    Main class for polynomial approximation of activations in FHE context.
    
    This class handles:
    1. Computing optimal polynomial coefficients
    2. Evaluating polynomials homomorphically on ciphertexts
    3. Managing depth usage for polynomial evaluation
    
    Usage:
        >>> approx = PolynomialApproximator(context)
        >>> approx_relu = approx.get_approximation('relu', degree=3)
        >>> ct_relu = approx.evaluate(ct_input, approx_relu)
    """
    
    def __init__(self, context: FHEContext):
        """
        Initialize approximator.
        
        Args:
            context: FHE context for operations
        """
        self.context = context
        self.ckks = context.ckks
        self.params = context.params
    
    def get_approximation(
        self,
        activation: str,
        degree: int = 3,
        domain: Tuple[float, float] = (-1.0, 1.0),
    ) -> Polynomial:
        """
        Get polynomial approximation for activation.
        
        Args:
            activation: Activation name
            degree: Polynomial degree
            domain: Domain for approximation
        
        Returns:
            Polynomial approximation
        """
        config = ApproximationConfig(
            degree=degree,
            domain=domain,
            method='chebyshev',
        )
        return ActivationApproximator.get_polynomial(activation, config)
    
    def get_relu(self, degree: int = 3) -> Polynomial:
        """
        Get ReLU polynomial approximation.
        
        For ReLU(x) = max(0, x):
        - Simple: 0.5*x + 0.5*|x| (but |x| needs approximation)
        - Better: Polynomial on [0, 1] and [-1, 0]
        """
        return self.get_approximation('relu', degree)
    
    def get_sigmoid(self, degree: int = 5) -> Polynomial:
        """Get Sigmoid approximation: σ(x) = 1/(1+e^(-x))"""
        return self.get_approximation('sigmoid', degree)
    
    def get_tanh(self, degree: int = 5) -> Polynomial:
        """Get Tanh approximation"""
        return self.get_approximation('tanh', degree)
    
    def get_gelu(self, degree: int = 7) -> Polynomial:
        """Get GELU approximation"""
        return self.get_approximation('gelu', degree)
    
    def evaluate(
        self,
        ciphertext: Ciphertext,
        polynomial: Polynomial,
    ) -> Ciphertext:
        """
        Evaluate polynomial homomorphically.
        
        Uses Horner's method to minimize multiplications:
            p(x) = c_0 + x·(c_1 + x·(c_2 + ...))
        
        Args:
            ciphertext: Encrypted input
            polynomial: Polynomial coefficients
        
        Returns:
            Encrypted polynomial evaluation
        """
        coeffs = polynomial.coefficients
        
        # Horner's method: evaluate from highest degree down
        result = None
        
        for i in range(len(coeffs) - 1, -1, -1):
            coeff = coeffs[i]
            
            # Multiply current result by x (ciphertext)
            if result is None:
                # Last coefficient
                result = self._multiply_scalar(ciphertext, coeff)
            else:
                # Multiply by x then add coefficient
                result = self.context.square(result)
                result = self.context.multiply_scalar(result, coeff)
        
        return result
    
    def _multiply_scalar(self, ciphertext: Ciphertext, scalar: float) -> Ciphertext:
        """Multiply ciphertext by scalar."""
        # Encode scalar as plaintext
        pt = self.ckks.encode(np.array([scalar]), ciphertext.scale)
        return self.ckks.multiply_plain(ciphertext, pt)
    
    def evaluate_with_powers(
        self,
        ciphertext: Ciphertext,
        polynomial: Polynomial,
        max_degree: Optional[int] = None,
    ) -> Ciphertext:
        """
        Evaluate polynomial using precomputed powers.
        
        More efficient when evaluating same input multiple times.
        
        Args:
            ciphertext: Encrypted input
            polynomial: Polynomial to evaluate
            max_degree: Maximum power to compute (default: polynomial.degree)
        
        Returns:
            Encrypted result
        """
        if max_degree is None:
            max_degree = polynomial.degree
        
        # Compute powers: x, x^2, x^4, x^8, ... (binary exponentiation style)
        powers = [ciphertext]
        p = ciphertext
        for i in range(1, max_degree.bit_length()):
            p = self.context.square(p)
            powers.append(p)
        
        # Combine powers for each coefficient
        coeffs = polynomial.coefficients
        result = self._multiply_scalar(ciphertext, 0)  # Zero ciphertext
        
        for i, c in enumerate(coeffs):
            if c != 0:
                # Get power i (binary decomposition)
                power_ct = self._get_power(powers, i)
                term = self._multiply_scalar(power_ct, c)
                result = self.context.add(result, term)
        
        return result
    
    def _get_power(self, powers: List[Ciphertext], n: int) -> Ciphertext:
        """Get x^n using binary decomposition of powers."""
        if n == 0:
            return self._multiply_scalar(powers[0], 1)  # Return 1
        if n == 1:
            return powers[0]
        
        result = None
        bit = 0
        while n > 0:
            if n & 1:
                if result is None:
                    result = powers[bit]
                else:
                    result = self.context.multiply(result, powers[bit])
            n >>= 1
            bit += 1
        
        return result
    
    # =========================================================================
    # Specialized Activation Approximations
    # =========================================================================
    
    def approximate_relu(
        self,
        ciphertext: Ciphertext,
        degree: int = 3,
        threshold: float = 0.0,
    ) -> Ciphertext:
        """
        Approximate ReLU activation.
        
        For encrypted ReLU, we use a polynomial approximation.
        Common approach: 0.5 * x + 0.5 * f(x) where f(x) ≈ |x|
        
        The absolute value can be approximated by:
            |x| ≈ polynomial(x) on [-1, 1]
        
        Args:
            ciphertext: Encrypted input
            degree: Polynomial degree
            threshold: ReLU threshold (usually 0)
        
        Returns:
            Encrypted ReLU output
        """
        # Subtract threshold
        x = self.context.sub(ciphertext, self.context.get_constant(threshold))
        
        # Approximate using Chebyshev
        # For ReLU: relu(x) = (x + |x|) / 2
        abs_poly = self._approximate_absolute_value(degree)
        
        # Compute |x|
        x_sq = self.context.square(x)
        abs_x = self.evaluate_with_powers(x_sq, abs_poly)
        abs_x = self.context.sqrt(abs_x)  # sqrt(|x|^2) = |x|
        
        # relu(x) = (x + |x|) / 2
        result = self.context.add(x, abs_x)
        result = self.context.multiply_scalar(result, 0.5)
        
        return result
    
    def approximate_sigmoid(self, ciphertext: Ciphertext, degree: int = 5) -> Ciphertext:
        """
        Approximate sigmoid: σ(x) = 1/(1+e^(-x))
        
        Uses polynomial approximation on [-8, 8] where sigmoid is essentially [0, 1].
        
        Args:
            ciphertext: Encrypted input
            degree: Polynomial degree
        
        Returns:
            Encrypted sigmoid output
        """
        poly = self.get_sigmoid(degree)
        return self.evaluate(ciphertext, poly)
    
    def approximate_tanh(self, ciphertext: Ciphertext, degree: int = 5) -> Ciphertext:
        """
        Approximate tanh activation.
        
        Args:
            ciphertext: Encrypted input
            degree: Polynomial degree
        
        Returns:
            Encrypted tanh output
        """
        poly = self.get_tanh(degree)
        return self.evaluate(ciphertext, poly)
    
    def approximate_gelu(self, ciphertext: Ciphertext, degree: int = 7) -> Ciphertext:
        """
        Approximate GELU activation.
        
        GELU(x) = x · Φ(x) where Φ is standard normal CDF
        Approximation: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        """
        poly = self.get_gelu(degree)
        return self.evaluate(ciphertext, poly)
    
    def _approximate_absolute_value(self, degree: int) -> Polynomial:
        """Approximate |x| as polynomial."""
        # |x| = sqrt(x²), so we approximate sqrt on [0, 1]
        def sqrt_poly(x):
            return np.sqrt(np.clip(x, 0, 1))
        
        return ChebyshevApproximator.approximate(sqrt_poly, degree, (0.0, 1.0))
    
    # =========================================================================
    # Automated Approximation Selection
    # =========================================================================
    
    def auto_approximate(
        self,
        activation: str,
        available_depth: int,
        target_error: float = 0.01,
    ) -> Tuple[Polynomial, int]:
        """
        Automatically select polynomial degree for given depth.
        
        Args:
            activation: Activation function name
            available_depth: Available multiplicative depth
            target_error: Target approximation error
        
        Returns:
            Tuple of (polynomial, actual_depth_used)
        
        Raises:
            ValueError: If even degree-1 polynomial exceeds depth
        """
        # Each polynomial evaluation needs degree-1 multiplications
        # With depth d, max polynomial degree is d+1 (for Horner)
        
        max_degree = available_depth + 1
        
        # Search for smallest degree meeting error target
        for degree in range(1, max_degree + 1):
            poly = self.get_approximation(activation, degree)
            
            # Estimate error
            error = self._estimate_error(activation, poly)
            
            if error <= target_error:
                return poly, degree
        
        # Use highest possible degree
        poly = self.get_approximation(activation, max_degree)
        return poly, max_degree
    
    def _estimate_error(
        self, 
        activation: str, 
        polynomial: Polynomial
    ) -> float:
        """Estimate maximum approximation error over domain."""
        func = ActivationApproximator._get_activation_function(activation)
        
        x_test = np.linspace(-1, 1, 10000)
        y_true = np.array([func(x) for x in x_test])
        y_approx = polynomial(x_test)
        
        return float(np.max(np.abs(y_true - y_approx)))
    
    def analyze_activation(
        self,
        activation: str,
        max_degree: int = 15,
    ) -> Dict[str, Any]:
        """
        Analyze approximation quality for all degrees up to max_degree.
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            'activation': activation,
            'degrees': [],
            'max_errors': [],
            'polynomials': [],
        }
        
        for degree in range(1, max_degree + 1):
            try:
                poly = self.get_approximation(activation, degree)
                error = self._estimate_error(activation, poly)
                
                results['degrees'].append(degree)
                results['max_errors'].append(error)
                results['polynomials'].append(str(poly)[:50])
            except Exception as e:
                results['degrees'].append(degree)
                results['max_errors'].append(float('inf'))
                results['polynomials'].append(str(e))
        
        return results
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print formatted analysis results."""
        print(f"\nActivation Analysis: {analysis['activation']}")
        print("-" * 60)
        print(f"{'Degree':<8} {'Max Error':<15} {'Coefficients'}")
        print("-" * 60)
        
        for i, deg in enumerate(analysis['degrees']):
            error = analysis['max_errors'][i]
            poly = analysis['polynomials'][i]
            error_str = f"{error:.2e}" if error != float('inf') else "N/A"
            print(f"{deg:<8} {error_str:<15} {poly[:40]}")


def approximate_activation(
    context: FHEContext,
    ciphertext: Ciphertext,
    activation: str,
    degree: int = 3,
) -> Ciphertext:
    """
    Convenience function for activation approximation.
    
    Args:
        context: FHE context
        ciphertext: Encrypted input
        activation: Activation name
        degree: Polynomial degree
    
    Returns:
        Encrypted activated output
    """
    approximator = PolynomialApproximator(context)
    return approximator.evaluate(ciphertext, approximator.get_approximation(activation, degree))


def comparative_analysis(context: FHEContext) -> Dict[str, Dict]:
    """
    Generate comparative analysis of all available activations.
    
    Args:
        context: FHE context for operations
    
    Returns:
        Dictionary mapping activation names to analysis results
    """
    approximator = PolynomialApproximator(context)
    activations = ActivationApproximator.list_activations()
    
    results = {}
    for activation in activations:
        try:
            results[activation] = approximator.analyze_activation(activation)
        except Exception as e:
            results[activation] = {'error': str(e)}
    
    return results
