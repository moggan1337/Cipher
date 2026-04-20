"""Machine Learning module for encrypted computation."""

from .encrypted_model import EncryptedLinear, EncryptedLayer, EncryptedModel
from .approximation import PolynomialApproximator, ActivationApproximator
from .encrypted_training import EncryptedOptimizer, FederatedAveraging

__all__ = [
    "EncryptedLinear",
    "EncryptedLayer", 
    "EncryptedModel",
    "PolynomialApproximator",
    "ActivationApproximator",
    "EncryptedOptimizer",
    "FederatedAveraging",
]
