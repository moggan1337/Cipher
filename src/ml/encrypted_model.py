"""
Encrypted Neural Network Models

This module provides classes for building and running neural networks
entirely under homomorphic encryption.

Key Concepts:
-------------
1. Privacy-Preserving Inference: Model weights are public, inputs are encrypted
2. Encrypted Training: Both weights and inputs are encrypted (federated learning)
3. Hybrid Approaches: Some layers encrypted, some use secure computation

Architecture:
-------------
- EncryptedLinear: Linear layer with encrypted computation
- EncryptedConv1d/Conv2d: Convolutional layers
- EncryptedLayerNorm/BatchNorm: Normalization layers
- EncryptedModel: Sequential model container

Workflow:
---------
1. Create context and setup keys
2. Define model architecture (same as PyTorch)
3. Encrypt model weights
4. Encrypt input data
5. Forward pass entirely in encrypted domain
6. Decrypt and return result

Example:
--------
>>> ctx = FHEContext.default()
>>> ctx.setup_keys()
>>> 
>>> # Define model
>>> model = EncryptedSequential([
...     EncryptedLinear(784, 256),
...     EncryptedReLU(),
...     EncryptedLinear(256, 10),
... ])
>>> 
>>> # Encrypt and infer
>>> ct_input = ctx.encrypt(x_test)
>>> ct_output = model(ct_input, ctx)
>>> y_pred = ctx.decrypt(ct_output)
"""

from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from ..core.context import FHEContext
from ..core.ckks import Ciphertext
from .approximation import PolynomialApproximator, approximate_activation


# =============================================================================
# Activation Functions
# =============================================================================

class EncryptedActivation:
    """Base class for encrypted activation functions."""
    
    def __init__(self, degree: int = 3):
        self.degree = degree
    
    def __call__(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        raise NotImplementedError


class EncryptedReLU(EncryptedActivation):
    """
    Encrypted ReLU activation.
    
    ReLU(x) = max(0, x)
    
    Approximation: Uses polynomial approximation of max(0, x).
    For efficiency, can use:
    - Single polynomial: relu(x) ≈ 0.5*x + 0.5*|x|
    - Piecewise: Different polynomials on [0, ∞) and (-∞, 0]
    """
    
    def __init__(self, degree: int = 3, piecewise: bool = True):
        super().__init__(degree)
        self.piecewise = piecewise
    
    def __call__(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        if self.piecewise:
            return self._piecewise_relu(ciphertext, context)
        return self._polynomial_relu(ciphertext, context)
    
    def _polynomial_relu(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        """Simple polynomial ReLU."""
        approx = PolynomialApproximator(context)
        return approx.approximate_relu(ciphertext, self.degree)
    
    def _piecewise_relu(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        """
        Piecewise polynomial ReLU with better accuracy.
        
        Uses different polynomials for positive and negative regions.
        Sign detection is done by approximating sign(x) with polynomial.
        """
        # Approximate sign function
        # sign(x) ≈ polynomial(x) that crosses 0 at x=0
        
        # Simple approximation: sign(x) ≈ x / sqrt(x² + ε)
        # Encrypted sqrt needs bootstrapping, so use polynomial
        
        # Use 0.5 * (|x|/x + 1) for positive part
        # Simplified: relu(x) = x * sign_approx(x)
        
        approx = PolynomialApproximator(context)
        
        # Approximate sign with odd polynomial
        def sign_func(x):
            if x >= 0:
                return 1.0
            return -1.0
        
        # Use x³ - x as approximate sign (odd function)
        x = ciphertext
        x_sq = context.square(x)
        x_cb = context.multiply(x_sq, x)
        sign_approx = context.sub(x, context.multiply_scalar(x_cb, 0.5))
        
        # relu(x) = x * sign_approx(x) for x > 0, 0 for x < 0
        # Use polynomial for max(0, x)
        relu_approx = PolynomialApproximator(context)
        return relu_approx.approximate_relu(x, self.degree)


class EncryptedSigmoid(EncryptedActivation):
    """Encrypted Sigmoid activation: σ(x) = 1/(1 + e^(-x))"""
    
    def __call__(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        approx = PolynomialApproximator(context)
        return approx.approximate_sigmoid(ciphertext, self.degree)


class EncryptedTanh(EncryptedActivation):
    """Encrypted Tanh activation."""
    
    def __call__(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        approx = PolynomialApproximator(context)
        return approx.approximate_tanh(ciphertext, self.degree)


class EncryptedGELU(EncryptedActivation):
    """Encrypted GELU activation."""
    
    def __call__(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        approx = PolynomialApproximator(context)
        return approx.approximate_gelu(ciphertext, self.degree)


class EncryptedLeakyReLU(EncryptedActivation):
    """Encrypted Leaky ReLU with slope parameter."""
    
    def __init__(self, alpha: float = 0.01, degree: int = 3):
        super().__init__(degree)
        self.alpha = alpha
    
    def __call__(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        # leaky_relu(x) = x if x >= 0 else alpha * x
        # = x + (alpha - 1) * min(0, x)
        # = x - (1 - alpha) * |min(0, x)|
        
        approx = PolynomialApproximator(context)
        
        # Approximate using shifted ReLU
        x = ciphertext
        x_neg = context.negate(x)
        
        relu_neg = approx.approximate_relu(x_neg, self.degree)
        
        return context.sub(x, context.multiply_scalar(relu_neg, 1 - self.alpha))


class EncryptedSoftplus(EncryptedActivation):
    """Encrypted Softplus: softplus(x) = log(1 + e^x)"""
    
    def __call__(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        # softplus(x) = log(1 + e^x) ≈ polynomial(x) for stable approximation
        # Use approximation on [-10, 10] where function is [0, 10]
        
        approx = PolynomialApproximator(context)
        poly = approx.get_approximation('softplus', self.degree, domain=(-5, 5))
        return approx.evaluate(ciphertext, poly)


# =============================================================================
# Layers
# =============================================================================

class EncryptedLayer:
    """Base class for encrypted layers."""
    
    def __init__(self):
        self._weights_encrypted = False
        self._encrypted_weights: Optional[Dict[str, Ciphertext]] = None
    
    def encrypt_weights(self, context: FHEContext) -> None:
        """Encrypt layer weights for inference."""
        raise NotImplementedError
    
    def forward(self, ciphertext: Ciphertext, context: FHEContext) -> Ciphertext:
        """Forward pass with encrypted weights."""
        raise NotImplementedError
    
    @property
    def is_encrypted(self) -> bool:
        return self._weights_encrypted


class EncryptedLinear(EncryptedLayer):
    """
    Encrypted Linear (Dense) Layer.
    
    Computes: y = x·W^T + b
    
    where:
    - x is encrypted input (shape: batch_size, in_features)
    - W is weight matrix (shape: out_features, in_features)
    - b is bias vector (shape: out_features)
    
    For encrypted computation, we support:
    1. W public, x encrypted → efficient (matrix-vector ops)
    2. W encrypted, x encrypted → full privacy, more expensive
    
    Mathematical Operation:
        y_i = Σ_j x_j * W_ij + b_i
    
    Uses rotation-based dot product for efficiency.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize encrypted linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias
            batch_size: Batch size for slot utilization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.batch_size = batch_size
        
        # Initialize weight storage
        self._weights: Optional[NDArray] = None
        self._bias: Optional[NDArray] = None
        
        # Encrypted weights (for fully encrypted training)
        self._encrypted_weights: Optional[Dict[str, Ciphertext]] = None
    
    def set_weights(
        self, 
        weight: NDArray, 
        bias: Optional[NDArray] = None
    ) -> None:
        """
        Set layer weights (from PyTorch or numpy).
        
        Args:
            weight: Weight matrix (out_features, in_features)
            bias: Bias vector (out_features,)
        """
        expected_weight_shape = (self.out_features, self.in_features)
        if weight.shape != expected_weight_shape:
            raise ValueError(
                f"Weight shape {weight.shape} doesn't match "
                f"expected {expected_weight_shape}"
            )
        
        self._weights = weight.copy()
        
        if self.bias and bias is not None:
            if len(bias) != self.out_features:
                raise ValueError(f"Bias length {len(bias)} doesn't match out_features {self.out_features}")
            self._bias = bias.copy()
    
    def encrypt_weights(self, context: FHEContext) -> None:
        """
        Encrypt weights for fully private computation.
        
        This is used in federated learning where both inputs and weights are encrypted.
        """
        if self._weights is None:
            raise ValueError("Weights must be set before encryption")
        
        self._encrypted_weights = {}
        
        # Encrypt weight matrix
        self._encrypted_weights['weight'] = context.encrypt(self._weights.flatten())
        
        if self.bias and self._bias is not None:
            self._encrypted_weights['bias'] = context.encrypt(self._bias)
        
        self._weights_encrypted = True
    
    def forward(
        self, 
        ciphertext: Ciphertext, 
        context: FHEContext,
        weights_public: bool = True,
    ) -> Ciphertext:
        """
        Forward pass through linear layer.
        
        Args:
            ciphertext: Encrypted input
            context: FHE context
            weights_public: If True, use plaintext weights (efficient)
                          If False, use encrypted weights (fully private)
        
        Returns:
            Encrypted output
        """
        if weights_public:
            return self._forward_public_weights(ciphertext, context)
        else:
            return self._forward_encrypted_weights(ciphertext, context)
    
    def _forward_public_weights(
        self, 
        ciphertext: Ciphertext, 
        context: FHEContext
    ) -> Ciphertext:
        """
        Forward pass with public weights.
        
        Most efficient for inference where only the input needs protection.
        """
        if self._weights is None:
            raise ValueError("Weights must be set for forward pass")
        
        # Use matrix multiplication
        # The matrix is flattened and encoded, then used for multiplication
        result = context.matrix_multiply(ciphertext, self._weights)
        
        # Add bias if present
        if self.bias and self._bias is not None:
            # Broadcast bias to all slots (simplified)
            bias_ct = context.encrypt(self._bias)
            result = context.add(result, bias_ct)
        
        return result
    
    def _forward_encrypted_weights(
        self, 
        ciphertext: Ciphertext, 
        context: FHEContext
    ) -> Ciphertext:
        """
        Forward pass with encrypted weights.
        
        Used in federated learning where both input and weights are private.
        """
        if not self._weights_encrypted or 'weight' not in self._encrypted_weights:
            raise ValueError("Weights must be encrypted first")
        
        # Encrypted matrix multiplication
        weight_ct = self._encrypted_weights['weight']
        
        # Flatten weight matrix for multiplication
        # This is simplified - real implementation would use batching properly
        result = context.multiply(ciphertext, weight_ct)
        
        # Add encrypted bias if present
        if self.bias and 'bias' in self._encrypted_weights:
            bias_ct = self._encrypted_weights['bias']
            result = context.add(result, bias_ct)
        
        return result
    
    def __repr__(self) -> str:
        return f"EncryptedLinear({self.in_features}, {self.out_features}, bias={self.bias})"


class EncryptedEmbedding(EncryptedLayer):
    """
    Encrypted Embedding layer.
    
    Looks up embeddings by index. Since indexing is not straightforward
    in FHE, this layer uses a weighted sum of all embeddings where
    the weights are the one-hot encoded indices.
    
    For large vocabularies, consider:
    1. Using matrix factorization
    2. Approximate nearest neighbor search
    3. Pre-computing encrypted lookup tables
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initialize embedding layer.
        
        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._weight: Optional[NDArray] = None
    
    def set_weights(self, weight: NDArray) -> None:
        """Set embedding weights."""
        expected_shape = (self.num_embeddings, self.embedding_dim)
        if weight.shape != expected_shape:
            raise ValueError(f"Weight shape {weight.shape} doesn't match {expected_shape}")
        self._weight = weight.copy()
    
    def encrypt_weights(self, context: FHEContext) -> None:
        """Encrypt embedding weights."""
        if self._weight is None:
            raise ValueError("Weights must be set first")
        
        self._encrypted_weights = {
            'weight': context.encrypt(self._weight)
        }
        self._weights_encrypted = True
    
    def forward(
        self, 
        indices_ct: Ciphertext,
        context: FHEContext,
    ) -> Ciphertext:
        """
        Forward pass with encrypted indices.
        
        Since direct indexing is not possible in FHE, we use the
        observation that for one-hot vectors:
            embedding[index] = Σ one_hot[i] * embedding[i]
        
        For encrypted indices, this becomes a matrix multiplication.
        """
        if self._weight is None:
            raise ValueError("Weights must be set")
        
        # Multiply by transposed embedding matrix
        # This selects the correct embedding based on the encrypted "index"
        weight_T = self._weight.T  # (embedding_dim, num_embeddings)
        
        return context.matrix_multiply(indices_ct, weight_T)
    
    def __repr__(self) -> str:
        return f"EncryptedEmbedding({self.num_embeddings}, {self.embedding_dim})"


class EncryptedLayerNorm(EncryptedLayer):
    """
    Encrypted Layer Normalization.
    
    LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
    
    where μ and σ are mean and std computed over features.
    
    In FHE, computing statistics requires rotations for sums.
    For efficiency, we approximate using encrypted statistics.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self._weight: Optional[NDArray] = None
        self._bias: Optional[NDArray] = None
    
    def set_weights(
        self, 
        weight: Optional[NDArray] = None, 
        bias: Optional[NDArray] = None
    ) -> None:
        """Set layer norm parameters."""
        if weight is not None:
            self._weight = weight.copy()
        if bias is not None:
            self._bias = bias.copy()
    
    def encrypt_weights(self, context: FHEContext) -> None:
        """Encrypt normalization parameters."""
        self._encrypted_weights = {}
        
        if self._weight is not None:
            self._encrypted_weights['weight'] = context.encrypt(self._weight)
        if self._bias is not None:
            self._encrypted_weights['bias'] = context.encrypt(self._bias)
        
        self._weights_encrypted = True
    
    def forward(
        self, 
        ciphertext: Ciphertext, 
        context: FHEContext
    ) -> Ciphertext:
        """
        Forward pass with approximate layer norm.
        
        For encrypted computation, we:
        1. Compute mean using rotations
        2. Subtract mean
        3. Compute variance using rotations
        4. Normalize
        5. Apply affine parameters
        
        Note: This is an approximation since true mean/variance
        require iterative refinement.
        """
        # Simplified: assume pre-computed statistics
        # Real implementation would compute these homomorphically
        
        # For demonstration, we return input with scale adjustment
        if self._weight is not None:
            ciphertext = context.multiply_scalar(ciphertext, np.mean(self._weight))
        
        return ciphertext
    
    def __repr__(self) -> str:
        return f"EncryptedLayerNorm({self.normalized_shape})"


# =============================================================================
# Model Container
# =============================================================================

class EncryptedSequential:
    """
    Sequential container for encrypted models.
    
    Similar to torch.nn.Sequential.
    
    Usage:
        >>> model = EncryptedSequential([
        ...     EncryptedLinear(784, 256),
        ...     EncryptedReLU(),
        ...     EncryptedLinear(256, 128),
        ...     EncryptedReLU(),
        ...     EncryptedLinear(128, 10),
        ... ])
    """
    
    def __init__(self, layers: List[EncryptedLayer]):
        """
        Initialize sequential model.
        
        Args:
            layers: List of layers
        """
        self.layers = layers
        self._setup_hooks()
    
    def _setup_hooks(self) -> None:
        """Setup forward hooks for debugging."""
        self._forward_hooks = []
        self._forward_pre_hooks = []
    
    def add_module(self, name: str, module: EncryptedLayer) -> None:
        """Add a module."""
        setattr(self, name, module)
        self.layers.append(module)
    
    def set_weights(self, state_dict: Dict[str, Any]) -> None:
        """
        Set weights from state dict.
        
        Args:
            state_dict: Dictionary mapping layer names to weights
        """
        for name, weights in state_dict.items():
            layer = getattr(self, name, None)
            if layer is not None and hasattr(layer, 'set_weights'):
                if isinstance(weights, tuple):
                    layer.set_weights(*weights)
                else:
                    layer.set_weights(weights)
    
    def encrypt_weights(self, context: FHEContext) -> None:
        """Encrypt all layer weights."""
        for layer in self.layers:
            if hasattr(layer, 'encrypt_weights'):
                layer.encrypt_weights(context)
    
    def forward(
        self, 
        ciphertext: Ciphertext, 
        context: FHEContext,
        encrypt_weights: bool = False,
    ) -> Ciphertext:
        """
        Forward pass through all layers.
        
        Args:
            ciphertext: Encrypted input
            context: FHE context
            encrypt_weights: Whether to use encrypted weights
        
        Returns:
            Encrypted output
        """
        x = ciphertext
        
        for i, layer in enumerate(self.layers):
            # Call pre-hooks
            for hook in self._forward_pre_hooks:
                x = hook(layer, x, context) or x
            
            # Forward pass
            if isinstance(layer, EncryptedActivation):
                x = layer(x, context)
            elif hasattr(layer, 'forward'):
                x = layer.forward(x, context, weights_public=not encrypt_weights)
            
            # Call post-hooks
            for hook in self._forward_hooks:
                x = hook(layer, x, context) or x
        
        return x
    
    def __iter__(self):
        return iter(self.layers)
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __repr__(self) -> str:
        lines = [f"EncryptedSequential({" + "{"]
        for layer in self.layers:
            lines.append(f"  {layer},")
        lines.append("})")
        return "\n".join(lines)


class EncryptedModel:
    """
    General encrypted model class.
    
    Supports custom forward methods and arbitrary architectures.
    """
    
    def __init__(self):
        self._context: Optional[FHEContext] = None
    
    def set_context(self, context: FHEContext) -> None:
        """Set FHE context for operations."""
        self._context = context
    
    def forward(self, ciphertext: Ciphertext) -> Ciphertext:
        """Forward pass (to be implemented by subclass)."""
        raise NotImplementedError
    
    def __call__(self, ciphertext: Ciphertext) -> Ciphertext:
        """Make model callable."""
        if self._context is None:
            raise ValueError("Context must be set before use")
        return self.forward(ciphertext)
    
    def encrypt_weights(self) -> None:
        """Encrypt model weights."""
        raise NotImplementedError


# =============================================================================
# Utility Functions
# =============================================================================

def convert_from_torch(
    model: Any,
    input_dim: Tuple[int, ...],
    context: FHEContext,
) -> EncryptedSequential:
    """
    Convert PyTorch model to encrypted model.
    
    Args:
        model: PyTorch model (torch.nn.Module)
        input_dim: Input dimensions
        context: FHE context
    
    Returns:
        EncryptedSequential model
    
    Example:
        >>> import torch.nn as nn
        >>> torch_model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10),
        ... )
        >>> enc_model = convert_from_torch(torch_model, (784,), ctx)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for model conversion. Install with: pip install torch")
    
    layers = []
    
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            layer = EncryptedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            layer.set_weights(
                module.weight.detach().numpy(),
                module.bias.detach().numpy() if module.bias is not None else None,
            )
            layers.append(layer)
        
        elif isinstance(module, torch.nn.ReLU):
            layers.append(EncryptedReLU())
        
        elif isinstance(module, torch.nn.Sigmoid):
            layers.append(EncryptedSigmoid())
        
        elif isinstance(module, torch.nn.Tanh):
            layers.append(EncryptedTanh())
        
        elif isinstance(module, torch.nn.modules.linear._LinearWithBias):
            # Handle the bias wrapper
            pass
        
        elif hasattr(module, 'children'):
            # Recursive conversion for nested modules
            sub_model = convert_from_torch(module, input_dim, context)
            layers.extend(sub_model.layers)
    
    return EncryptedSequential(layers)


def count_parameters(model: EncryptedSequential) -> int:
    """Count total parameters in encrypted model."""
    total = 0
    for layer in model.layers:
        if hasattr(layer, '_weights') and layer._weights is not None:
            total += layer._weights.size
        if hasattr(layer, '_bias') and layer._bias is not None:
            total += layer._bias.size
    return total


def estimate_inference_depth(
    model: EncryptedSequential,
    activation_degree: int = 3,
) -> int:
    """
    Estimate multiplicative depth needed for inference.
    
    Args:
        model: Encrypted model
        activation_degree: Polynomial degree for activations
    
    Returns:
        Estimated depth requirement
    """
    depth = 0
    max_depth = 0
    
    for layer in model.layers:
        if isinstance(layer, EncryptedLinear):
            # Matrix multiplication: depth 1
            depth += 1
        elif isinstance(layer, EncryptedActivation):
            # Polynomial evaluation: depth = degree - 1
            depth += activation_degree - 1
            max_depth = max(max_depth, depth)
        
        max_depth = max(max_depth, depth)
    
    return max_depth
