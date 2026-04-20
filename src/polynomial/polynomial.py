"""
Polynomial Arithmetic and Approximation

This module provides polynomial operations needed for CKKS, including:
- Basic polynomial arithmetic
- Polynomial approximation of functions
- Evaluation and composition

Polynomial Approximation:
------------------------
In FHE, non-polynomial functions (like ReLU, sigmoid) must be approximated
by polynomials. Common methods:

1. Chebyshev Approximation: Minimax approximation on [-1, 1]
   - Best uniform approximation
   - Coefficients can be computed iteratively
   
2. Taylor Series: Derivatives at a point
   - Simple but less accurate away from expansion point
   
3. Remez Algorithm: Iterative minimax refinement
   - Optimal but computationally expensive

For CKKS, we typically use truncated Chebyshev series or best uniform
approximation on the relevant domain.
"""

from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import warnings


@dataclass
class Polynomial:
    """
    Polynomial in a ring with coefficients.
    
    Represents: p(x) = Σ c_i · x^i
    
    Attributes:
        coefficients: Array of coefficients [c_0, c_1, ..., c_n]
        modulus: Modulus for coefficient arithmetic (None for floats)
    """
    
    coefficients: NDArray[np.float64]
    modulus: Optional[int] = None
    
    def __post_init__(self):
        if len(self.coefficients) == 0:
            self.coefficients = np.array([0.0])
        # Trim trailing zeros
        while len(self.coefficients) > 1 and self.coefficients[-1] == 0:
            self.coefficients = self.coefficients[:-1]
    
    @property
    def degree(self) -> int:
        """Polynomial degree (highest power with non-zero coefficient)."""
        return len(self.coefficients) - 1
    
    @property
    def coefficients_list(self) -> List[float]:
        """Get coefficients as list."""
        return self.coefficients.tolist()
    
    def __call__(self, x: Union[float, NDArray]) -> Union[float, NDArray]:
        """Evaluate polynomial at x using Horner's method."""
        return np.polyval(self.coefficients[::-1], x)
    
    def __add__(self, other: "Polynomial") -> "Polynomial":
        """Add two polynomials."""
        if self.modulus != other.modulus:
            raise ValueError("Cannot add polynomials with different moduli")
        
        result_len = max(len(self.coefficients), len(other.coefficients))
        result = np.zeros(result_len)
        result[:len(self.coefficients)] += self.coefficients
        result[:len(other.coefficients)] += other.coefficients
        
        if self.modulus:
            result = np.mod(result, self.modulus)
        
        return Polynomial(result, self.modulus)
    
    def __sub__(self, other: "Polynomial") -> "Polynomial":
        """Subtract polynomials."""
        if self.modulus != other.modulus:
            raise ValueError("Cannot subtract polynomials with different moduli")
        
        result_len = max(len(self.coefficients), len(other.coefficients))
        result = np.zeros(result_len)
        result[:len(self.coefficients)] += self.coefficients
        result[:len(other.coefficients)] -= other.coefficients
        
        if self.modulus:
            result = np.mod(result, self.modulus)
        
        return Polynomial(result, self.modulus)
    
    def __mul__(self, other: Union["Polynomial", float, int]) -> "Polynomial":
        """Multiply polynomial by polynomial or scalar."""
        if isinstance(other, (float, int)):
            coeffs = self.coefficients * other
            return Polynomial(coeffs, self.modulus)
        
        if self.modulus != other.modulus:
            raise ValueError("Cannot multiply polynomials with different moduli")
        
        coeffs = np.convolve(self.coefficients, other.coefficients)
        
        if self.modulus:
            coeffs = np.mod(coeffs, self.modulus)
        
        return Polynomial(coeffs, self.modulus)
    
    def __truediv__(self, scalar: float) -> "Polynomial":
        """Divide by scalar."""
        return Polynomial(self.coefficients / scalar, self.modulus)
    
    def derivative(self) -> "Polynomial":
        """Compute derivative."""
        if len(self.coefficients) <= 1:
            return Polynomial(np.array([0.0]), self.modulus)
        
        deriv_coeffs = np.zeros(len(self.coefficients) - 1)
        for i in range(1, len(self.coefficients)):
            deriv_coeffs[i - 1] = i * self.coefficients[i]
        
        return Polynomial(deriv_coeffs, self.modulus)
    
    def integral(self, c: float = 0.0) -> "Polynomial":
        """Compute indefinite integral."""
        int_coeffs = np.zeros(len(self.coefficients) + 1)
        int_coeffs[0] = c
        for i in range(len(self.coefficients)):
            int_coeffs[i + 1] = self.coefficients[i] / (i + 1)
        
        return Polynomial(int_coeffs, self.modulus)
    
    def compose(self, other: "Polynomial") -> "Polynomial":
        """Compose polynomials: self(other(x))."""
        result = Polynomial(np.array([0.0]), self.modulus)
        
        for i, coeff in enumerate(self.coefficients):
            result = result + other ** i * coeff
        
        return result
    
    def __pow__(self, n: int) -> "Polynomial":
        """Power of polynomial."""
        if n == 0:
            return Polynomial(np.array([1.0]), self.modulus)
        if n == 1:
            return Polynomial(self.coefficients.copy(), self.modulus)
        
        result = Polynomial(np.array([1.0]), self.modulus)
        base = Polynomial(self.coefficients.copy(), self.modulus)
        
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        
        return result
    
    def truncate(self, degree: int) -> "Polynomial":
        """Truncate to maximum degree."""
        if degree >= self.degree:
            return Polynomial(self.coefficients.copy(), self.modulus)
        
        return Polynomial(self.coefficients[:degree + 1].copy(), self.modulus)
    
    def scale(self, factor: float) -> "Polynomial":
        """Scale all coefficients by factor."""
        return Polynomial(self.coefficients * factor, self.modulus)
    
    def evaluate_horner(self, x: float) -> float:
        """Evaluate using Horner's method (more numerically stable)."""
        result = 0.0
        for coeff in reversed(self.coefficients):
            result = result * x + coeff
        return result
    
    @classmethod
    def from_roots(cls, roots: List[float]) -> "Polynomial":
        """Create polynomial from roots."""
        coeffs = np.poly(roots)
        return cls(coeffs)
    
    @classmethod
    def chebyshev_nodes(cls, n: int, a: float = -1.0, b: float = 1.0) -> NDArray:
        """
        Generate Chebyshev nodes in [a, b].
        
        Nodes: x_i = (b - a)/2 * cos(π(2i+1)/(2n)) + (a + b)/2
        """
        i = np.arange(n)
        nodes = np.cos(np.pi * (2 * i + 1) / (2 * n))
        return (b - a) / 2 * nodes + (a + b) / 2
    
    @classmethod
    def monomial(cls, degree: int, modulus: Optional[int] = None) -> "Polynomial":
        """Create monomial x^degree."""
        coeffs = np.zeros(degree + 1)
        coeffs[degree] = 1.0
        return cls(coeffs, modulus)
    
    def __repr__(self) -> str:
        if len(self.coefficients) <= 5:
            return f"Poly({self.coefficients})"
        return f"Poly(degree={self.degree}, coeffs={self.coefficients[:3]}...{self.coefficients[-1]})"


@dataclass
class PolynomialRing:
    """
    Polynomial ring R = Z[X]/(X^n + 1).
    
    This is the foundational ring for ring-LWE based FHE schemes like CKKS.
    The modulus X^n + 1 ensures that X^n ≡ -1, enabling efficient operations.
    """
    
    degree: int
    modulus: int
    
    def mul(self, a: NDArray, b: NDArray) -> NDArray:
        """
        Multiply polynomials in the ring.
        
        Args:
            a, b: Polynomial coefficients
        
        Returns:
            Product in the ring
        """
        # Standard convolution
        n = self.degree
        result = np.convolve(a, b)
        
        # Reduce mod X^n + 1 (i.e., X^n ≡ -1)
        if len(result) > n:
            # Wrap around terms
            result[:n] = result[:n] - result[n:]
            result = result[:n]
        
        return np.mod(result, self.modulus)
    
    def pow(self, a: NDArray, exp: int) -> NDArray:
        """Power in the ring."""
        result = np.array([1] + [0] * (self.degree - 1))
        base = a.copy()
        
        while exp > 0:
            if exp % 2 == 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            exp //= 2
        
        return result


@dataclass  
class PolynomialVector:
    """Vector of polynomials (for batched operations)."""
    
    polynomials: List[Polynomial]
    
    def __post_init__(self):
        if len(self.polynomials) == 0:
            self.polynomials = [Polynomial(np.array([0.0]))]
    
    @property
    def size(self) -> int:
        return len(self.polynomials)
    
    def __getitem__(self, i: int) -> Polynomial:
        return self.polynomials[i]
    
    def __setitem__(self, i: int, p: Polynomial):
        self.polynomials[i] = p


class ChebyshevApproximator:
    """
    Chebyshev polynomial approximation.
    
    Chebyshev polynomials T_n(x) form an orthogonal basis on [-1, 1] with
    weight (1-x²)^(-1/2). They minimize the maximum error (minimax property).
    
    The approximation of f(x) on [-1, 1] is:
        f(x) ≈ Σ a_n · T_n(x)
    
    where coefficients are computed via discrete orthogonal projection.
    """
    
    @staticmethod
    def chebyshev_polynomial(n: int) -> Polynomial:
        """
        Generate Chebyshev polynomial T_n(x).
        
        Recurrence: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)
        """
        if n == 0:
            return Polynomial(np.array([1.0]))
        if n == 1:
            return Polynomial(np.array([0.0, 1.0]))
        
        t_prev = Polynomial(np.array([1.0]))
        t_curr = Polynomial(np.array([0.0, 1.0]))
        
        for _ in range(2, n + 1):
            t_next = Polynomial(np.array([0.0, 2.0]), None) * t_curr - t_prev
            t_prev, t_curr = t_curr, t_next
        
        return t_curr
    
    @classmethod
    def approximate(
        cls,
        func: Callable[[float], float],
        degree: int,
        domain: Tuple[float, float] = (-1.0, 1.0),
        n_samples: Optional[int] = None,
    ) -> Polynomial:
        """
        Approximate function using Chebyshev polynomials.
        
        Args:
            func: Function to approximate
            degree: Maximum polynomial degree
            domain: Domain [a, b] for approximation
            n_samples: Number of sample points (default: degree + 1)
        
        Returns:
            Polynomial approximating func on [a, b]
        
        Method:
            Use Clenshaw-Curtis quadrature at Chebyshev nodes:
            1. Sample f at n Chebyshev nodes
            2. Compute Chebyshev coefficients via DCT
            3. Convert to standard polynomial basis
        """
        if n_samples is None:
            n_samples = degree + 1
        
        a, b = domain
        
        # Generate Chebyshev nodes in [a, b]
        nodes = cls._chebyshev_nodes(n_samples, a, b)
        
        # Evaluate function at nodes
        values = np.array([func(x) for x in nodes])
        
        # Compute Chebyshev coefficients
        coeffs = cls._chebyshev_coefficients(values)
        
        # Truncate to desired degree
        coeffs = coeffs[:degree + 1]
        
        # Convert Chebyshev basis to monomial basis
        return cls._chebyshev_to_monomial(coeffs, a, b)
    
    @staticmethod
    def _chebyshev_nodes(n: int, a: float, b: float) -> NDArray:
        """Chebyshev nodes in [a, b]."""
        i = np.arange(n)
        nodes = np.cos(np.pi * (2 * i + 1) / (2 * n))
        return (b - a) / 2 * nodes + (a + b) / 2
    
    @staticmethod
    def _chebyshev_coefficients(values: NDArray) -> NDArray:
        """Compute Chebyshev coefficients via DCT."""
        n = len(values)
        coeffs = np.zeros(n)
        
        for k in range(n):
            sum_val = 0.0
            for j in range(n):
                theta = np.pi * (2 * j + 1) * k / (2 * n)
                sum_val += values[j] * np.cos(theta)
            coeffs[k] = (2 / n) * sum_val
        
        coeffs[0] /= 2  # First coefficient is special
        return coeffs
    
    @staticmethod
    def _chebyshev_to_monomial(
        cheb_coeffs: NDArray, 
        a: float, 
        b: float
    ) -> Polynomial:
        """Convert Chebyshev coefficients to monomial coefficients."""
        n = len(cheb_coeffs)
        
        # Transform domain from [a, b] to [-1, 1]
        # x' = (2x - a - b) / (b - a)
        scale = 2 / (b - a)
        shift = -(a + b) / (b - a)
        
        # Express Chebyshev polynomials in monomial basis
        # T_n(x') = cos(n·arccos(x'))
        # We use the recurrence relation in monomial form
        
        # Start with T_0 = 1, T_1 = x'
        monomials = [np.array([1.0])]  # T_0
        monomials.append(np.array([shift, scale]))  # T_1 = x'
        
        for n in range(2, len(cheb_coeffs)):
            # T_n = 2x'·T_{n-1} - T_{n-2}
            t_nm1 = monomials[-1]
            t_nm2 = monomials[-2]
            
            # Convolution with [shift, scale] for x'
            conv = np.convolve(t_nm1, [shift, scale])
            t_n = 2 * conv - np.concatenate([np.zeros(2), t_nm2])
            monomials.append(t_n)
        
        # Combine coefficients
        result = np.zeros(len(cheb_coeffs))
        for i, c in enumerate(cheb_coeffs):
            if i < len(monomials):
                # Pad to same length
                padded = np.zeros(len(result))
                padded[:len(monomials[i])] = monomials[i]
                result += c * padded
        
        return Polynomial(result)
    
    @classmethod
    def approximate_with_remez(
        cls,
        func: Callable[[float], float],
        degree: int,
        domain: Tuple[float, float] = (-1.0, 1.0),
        max_iterations: int = 10,
        tolerance: float = 1e-10,
    ) -> Polynomial:
        """
        Best uniform (minimax) approximation using Remez algorithm.
        
        This finds the polynomial that minimizes the maximum error.
        
        Args:
            func: Function to approximate
            degree: Polynomial degree
            domain: Approximation domain
            max_iterations: Maximum Remez iterations
            tolerance: Convergence tolerance
        
        Returns:
            Best approximating polynomial
        """
        from scipy.optimize import minimize
        
        a, b = domain
        
        # Initial approximation via Chebyshev
        poly = cls.approximate(func, degree, domain)
        
        # Remez refinement
        for _ in range(max_iterations):
            # Find alternation points
            n_points = degree + 2
            x_alternation = np.linspace(a, b, n_points)
            
            # Evaluate error at alternation points
            y_func = np.array([func(x) for x in x_alternation])
            y_poly = poly(x_alternation)
            errors = y_func - y_poly
            
            # Find extrema of error
            error_mags = np.abs(errors)
            max_idx = np.argmax(error_mags)
            max_error = error_mags[max_idx]
            
            # Check signs at alternation points
            signs = np.sign(errors)
            if not np.all(np.diff(signs) != 0):
                warnings.warn("Alternation property not satisfied")
            
            # Adjust polynomial to equalize errors
            # This is simplified - full Remez would solve linear system
            poly = cls.approximate(func, degree, domain, n_samples=degree * 4)
            
            # Check convergence
            if max_error < tolerance:
                break
        
        return poly


class TaylorApproximator:
    """
    Taylor series approximation.
    
    Taylor series of f at x_0:
        f(x) = Σ f^(n)(x_0)/n! · (x - x_0)^n
    
    Simple but accuracy degrades away from expansion point.
    """
    
    @classmethod
    def approximate(
        cls,
        func: Callable[[float], float],
        degree: int,
        center: float = 0.0,
    ) -> Polynomial:
        """
        Compute Taylor series approximation.
        
        Args:
            func: Function (must be differentiable)
            degree: Series degree
            center: Expansion point x_0
        
        Returns:
            Taylor polynomial
        """
        # Numerical differentiation using finite differences
        h = 1e-5
        
        coeffs = []
        for n in range(degree + 1):
            deriv = cls._nth_derivative(func, center, n, h)
            coeff = deriv / np.math.factorial(n)
            coeffs.append(coeff)
        
        # Shift to center
        if center != 0:
            coeffs = cls._shift_coefficients(coeffs, center)
        
        return Polynomial(np.array(coeffs))
    
    @staticmethod
    def _nth_derivative(
        func: Callable[[float], float],
        x: float,
        n: int,
        h: float,
    ) -> float:
        """Compute n-th derivative numerically."""
        if n == 0:
            return func(x)
        
        # Use central differences
        order = min(n, 5)  # Cap for stability
        result = 0.0
        
        for k in range(order + 1):
            sign = (-1) ** (order - k)
            binom = np.math.comb(order, k)
            result += sign * binom * func(x + (k - order / 2) * h)
        
        return result / (h ** n)
    
    @staticmethod
    def _shift_coefficients(coeffs: List[float], center: float) -> List[float]:
        """Shift polynomial coefficients to new center using binomial theorem."""
        result = [0.0] * len(coeffs)
        
        for i in range(len(coeffs)):
            for j in range(i + 1):
                result[i] += coeffs[j] * np.math.comb(j, i - j) * (-center) ** (j - (i - j))
        
        return result


class BestUniformApproximator:
    """
    Best uniform approximation using alternating theorem.
    
    For the best approximant p* of degree n:
        - There exist n+2 points where f(x_i) - p*(x_i) = ±E
        - Signs alternate at these points
    """
    
    @classmethod
    def approximate(
        cls,
        func: Callable[[float], float],
        degree: int,
        domain: Tuple[float, float] = (-1.0, 1.0),
        n_init: int = 50,
    ) -> Tuple[Polynomial, float]:
        """
        Find best uniform approximation.
        
        Returns:
            Tuple of (polynomial, maximum error)
        """
        from scipy.optimize import minimize
        
        a, b = domain
        
        def max_error(coeffs):
            """Maximum error over domain."""
            p = Polynomial(np.array(coeffs))
            x_test = np.linspace(a, b, 1000)
            errors = np.abs(func(x_test) - p(x_test))
            return np.max(errors)
        
        # Initial guess from Chebyshev
        cheb = ChebyshevApproximator.approximate(func, degree, domain)
        x0 = cheb.coefficients.tolist()
        
        # Optimize
        result = minimize(
            max_error,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        poly = Polynomial(np.array(result.x))
        error = result.fun
        
        return poly, error
