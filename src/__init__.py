"""
Cipher - Homomorphic Encryption ML Framework
A comprehensive framework for privacy-preserving machine learning using
Cheon-Kim-Kim-Song (CKKS) fully homomorphic encryption scheme.
"""

__version__ = "1.0.0"
__author__ = "Cipher Team"

from .core.keys import SecretKey, PublicKey, EvaluationKey
from .core.ckks import CKKS
from .core.context import FHEContext
from .ml.encrypted_model import EncryptedLinear, EncryptedModel
from .ml.approximation import PolynomialApproximator
from .polynomial.polynomial import Polynomial, PolynomialRing
from .bootstrapping.bootstrapping import Bootstrapping

__all__ = [
    "CKKS",
    "FHEContext",
    "SecretKey",
    "PublicKey", 
    "EvaluationKey",
    "EncryptedLinear",
    "EncryptedModel",
    "PolynomialApproximator",
    "Polynomial",
    "PolynomialRing",
    "Bootstrapping",
]
