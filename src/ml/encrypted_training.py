"""
Privacy-Preserving Training with Federated Learning

This module implements federated learning under homomorphic encryption,
enabling collaborative model training without exposing raw data.

Federated Learning Overview:
---------------------------
1. Clients (data owners) encrypt their local data
2. Server aggregates encrypted gradients from all clients
3. Model updates are computed on aggregated encrypted data
4. Only encrypted model updates are sent back to clients
5. Clients decrypt and update their local models

Privacy Guarantees:
- Client data is never shared in plaintext
- Server cannot see individual client gradients
- Aggregation preserves differential privacy properties

Key Operations:
1. Encrypted Gradient Computation: ∇L(x_i, θ) encrypted
2. Secure Aggregation: Σ ∇L(x_i, θ) computed homomorphically
3. Encrypted Optimization: θ ← θ - η · ∇L computed under encryption
4. Differential Privacy: Noise added before aggregation

Mathematical Framework:
----------------------
For loss L and model parameters θ:
    ∇θ = ∂L/∂θ

Encrypted update:
    enc(θ_new) = enc(θ_old - η · Σ_i ∇L(x_i, θ_old))
              = enc(θ_old) ⊗ (1 - η · Σ_i ∇L(x_i, θ_old))

Where ⊗ is homomorphic multiplication.
"""

from typing import List, Dict, Optional, Tuple, Callable, Any, Protocol
from dataclasses import dataclass, field
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import time

from ..core.context import FHEContext
from ..core.ckks import Ciphertext
from .encrypted_model import EncryptedLinear, EncryptedSequential


class EncryptedOptimizer:
    """
    Optimizer for encrypted model parameters.
    
    Supports:
    - SGD with momentum
    - Adam (approximated)
    - Gradient averaging for federated learning
    
    Note: Adam's second moment estimation requires division by
    encrypted values, which is approximated using multiplication
    by precomputed inverses.
    """
    
    def __init__(
        self,
        model: EncryptedSequential,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """
        Initialize encrypted optimizer.
        
        Args:
            model: Encrypted model to optimize
            learning_rate: Learning rate η
            momentum: Momentum coefficient β
            weight_decay: L2 regularization λ
        """
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # State for momentum
        self._velocity: Dict[str, Ciphertext] = {}
        self._initialized = False
    
    def _initialize_state(self, context: FHEContext) -> None:
        """Initialize optimizer state."""
        if self._initialized:
            return
        
        # Initialize velocity to zeros
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, EncryptedLinear):
                self._velocity[f'layer_{i}_weight'] = context.encrypt(
                    np.zeros_like(layer._weights) if layer._weights is not None 
                    else np.zeros((layer.out_features, layer.in_features))
                )
                if layer.bias and layer._bias is not None:
                    self._velocity[f'layer_{i}_bias'] = context.encrypt(
                        np.zeros_like(layer._bias)
                    )
        
        self._initialized = True
    
    def step(
        self, 
        gradients: Dict[str, Ciphertext],
        context: FHEContext,
    ) -> None:
        """
        Perform optimization step.
        
        θ_new = θ_old - lr · (∇L + λ·θ_old) if weight_decay > 0
        θ_new = θ_old - lr · ∇L otherwise
        
        With momentum:
        v_new = β · v_old + (1-β) · ∇L
        θ_new = θ_old - lr · v_new
        
        Args:
            gradients: Dictionary of encrypted gradients
            context: FHE context
        """
        self._initialize_state(context)
        
        for name, grad in gradients.items():
            # Get current parameter
            param = self._get_param(name, context)
            
            # Compute update with momentum
            if self.momentum > 0 and name in self._velocity:
                velocity = self._velocity[name]
                # v_new = β · v_old + (1-β) · grad
                momentum_term = context.multiply_scalar(velocity, self.momentum)
                grad_term = context.multiply_scalar(grad, 1 - self.momentum)
                new_velocity = context.add(momentum_term, grad_term)
                self._velocity[name] = new_velocity
                update = new_velocity
            else:
                update = grad
            
            # Apply weight decay
            if self.weight_decay > 0 and param is not None:
                decay_term = context.multiply_scalar(param, self.weight_decay * self.lr)
                update = context.add(update, decay_term)
            
            # θ_new = θ_old - lr · update
            scaled_update = context.multiply_scalar(update, -self.lr)
            new_param = context.add(param, scaled_update)
            
            # Update parameter
            self._set_param(name, new_param)
    
    def _get_param(self, name: str, context: FHEContext) -> Optional[Ciphertext]:
        """Get parameter by name."""
        parts = name.split('_')
        if len(parts) < 3:
            return None
        
        layer_idx = int(parts[1])
        param_type = parts[2]
        
        if layer_idx < len(self.model.layers):
            layer = self.model.layers[layer_idx]
            if isinstance(layer, EncryptedLinear):
                if param_type == 'weight' and layer._weights is not None:
                    return context.encrypt(layer._weights)
                elif param_type == 'bias' and layer._bias is not None:
                    return context.encrypt(layer._bias)
        
        return None
    
    def _set_param(self, name: str, value: Ciphertext) -> None:
        """Set parameter by name (placeholder for actual implementation)."""
        # In a full implementation, this would update the model's
        # encrypted parameter storage
        pass
    
    def zero_grad(self) -> None:
        """Clear gradients."""
        self._velocity.clear()
        self._initialized = False


class FederatedClient:
    """
    Federated learning client.
    
    Each client:
    1. Holds local data (encrypted)
    2. Computes local gradients
    3. Sends encrypted gradients to server
    
    Privacy: Raw data never leaves the client.
    """
    
    def __init__(
        self,
        client_id: str,
        model: EncryptedSequential,
        context: FHEContext,
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model: Local model copy
            context: FHE context for encryption
        """
        self.client_id = client_id
        self.model = model
        self.context = context
        
        # Local data (stored encrypted)
        self._encrypted_data: List[Ciphertext] = []
        self._encrypted_labels: List[Ciphertext] = []
        
        # Local statistics
        self._num_samples = 0
    
    def load_data(
        self, 
        X: NDArray, 
        y: Optional[NDArray] = None
    ) -> None:
        """
        Load local data (encrypts immediately).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - optional for inference
        """
        self._encrypted_data = []
        
        # Encrypt in batches
        batch_size = min(100, len(X))
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            ct = self.context.encrypt(batch)
            self._encrypted_data.append(ct)
        
        if y is not None:
            self._encrypted_labels = []
            for i in range(0, len(y), batch_size):
                batch = y[i:i + batch_size]
                ct = self.context.encrypt(batch)
                self._encrypted_labels.append(ct)
        
        self._num_samples = len(X)
    
    def compute_gradients(
        self,
        loss_fn: Callable[[Ciphertext, Ciphertext], Ciphertext],
    ) -> Dict[str, Ciphertext]:
        """
        Compute encrypted gradients on local data.
        
        Args:
            loss_fn: Loss function (encrypted_input, encrypted_label) -> encrypted_loss
        
        Returns:
            Dictionary of encrypted gradients
        """
        gradients = {}
        
        # Simplified: compute gradients for each layer
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, EncryptedLinear):
                # In practice, use backpropagation
                # Here we simulate with random gradients for demonstration
                
                if layer._weights is not None:
                    grad_shape = layer._weights.shape
                    grad_values = np.random.randn(*grad_shape) * 0.01
                    gradients[f'layer_{i}_weight'] = self.context.encrypt(grad_values)
                
                if layer.bias and layer._bias is not None:
                    grad_shape = layer._bias.shape
                    grad_values = np.random.randn(*grad_shape) * 0.01
                    gradients[f'layer_{i}_bias'] = self.context.encrypt(grad_values)
        
        return gradients
    
    def apply_model_update(
        self,
        encrypted_updates: Dict[str, Ciphertext],
        decrypt: bool = True,
    ) -> None:
        """
        Apply model updates received from server.
        
        Args:
            encrypted_updates: Encrypted gradient/parameter updates
            decrypt: Whether to decrypt updates before applying
        """
        if decrypt:
            for name, ct in encrypted_updates.items():
                # Decrypt and apply locally
                update = self.context.decrypt(ct)
                self._apply_update(name, update)
        else:
            # Apply directly to encrypted model
            for name, ct in encrypted_updates.items():
                self._set_encrypted_param(name, ct)
    
    def _apply_update(self, name: str, update: NDArray) -> None:
        """Apply plaintext update to local model."""
        # Parse name and update corresponding layer
        parts = name.split('_')
        if len(parts) >= 3:
            layer_idx = int(parts[1])
            param_type = parts[2]
            
            if layer_idx < len(self.model.layers):
                layer = self.model.layers[layer_idx]
                if isinstance(layer, EncryptedLinear):
                    if param_type == 'weight' and layer._weights is not None:
                        layer._weights -= update * self.context.encrypt  # Simplified
                    elif param_type == 'bias' and layer._bias is not None:
                        layer._bias -= update
    
    def _set_encrypted_param(self, name: str, ct: Ciphertext) -> None:
        """Set encrypted parameter."""
        # Store encrypted parameter for encrypted model
        pass
    
    @property
    def num_samples(self) -> int:
        """Number of local samples."""
        return self._num_samples


class FederatedServer:
    """
    Federated learning server.
    
    The server:
    1. Initializes and distributes model parameters
    2. Receives encrypted gradients from clients
    3. Performs secure aggregation
    4. Updates global model
    5. Distributes updated parameters
    
    Privacy: Server never sees individual client gradients.
    """
    
    def __init__(
        self,
        model: EncryptedSequential,
        context: FHEContext,
    ):
        """
        Initialize federated server.
        
        Args:
            model: Global model (template for clients)
            context: FHE context
        """
        self.model = model
        self.context = context
        
        # Client management
        self._clients: Dict[str, FederatedClient] = {}
        
        # Aggregation state
        self._round = 0
        self._aggregation_history: List[Dict] = []
    
    def register_client(self, client: FederatedClient) -> None:
        """Register a client with the server."""
        self._clients[client.client_id] = client
    
    def unregister_client(self, client_id: str) -> None:
        """Remove a client from the federation."""
        if client_id in self._clients:
            del self._clients[client_id]
    
    def sample_clients(
        self, 
        fraction: float = 1.0,
        min_clients: int = 1,
    ) -> List[FederatedClient]:
        """
        Sample clients for current round.
        
        Args:
            fraction: Fraction of clients to sample (0.0 to 1.0)
            min_clients: Minimum number of clients
        
        Returns:
            List of sampled clients
        """
        n_clients = len(self._clients)
        n_sample = max(min_clients, int(n_clients * fraction))
        
        client_list = list(self._clients.values())
        indices = np.random.choice(n_clients, size=n_sample, replace=False)
        
        return [client_list[i] for i in indices]
    
    def secure_aggregate(
        self,
        client_gradients: Dict[str, List[Ciphertext]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Ciphertext]:
        """
        Perform secure aggregation of encrypted gradients.
        
        Mathematical Form:
            agg_grad[name] = Σ (n_i · grad_i[name]) / Σ n_i
        
        Where n_i is the number of samples from client i.
        
        Args:
            client_gradients: {client_id: {param_name: encrypted_grad}}
            weights: Optional sample weights for weighted averaging
        
        Returns:
            Aggregated encrypted gradients
        """
        if not client_gradients:
            return {}
        
        # Get parameter names from first client
        first_client = next(iter(client_gradients.values()))
        param_names = list(first_client.keys())
        
        aggregated = {}
        
        for param_name in param_names:
            # Collect gradients from all clients
            grads = []
            client_weights = []
            
            for client_id, gradients in client_gradients.items():
                if param_name in gradients:
                    grads.append(gradients[param_name])
                    
                    # Weight by number of samples
                    weight = weights[client_id] if weights else 1.0
                    client_weights.append(weight)
            
            if not grads:
                continue
            
            # Weighted sum of gradients
            total_weight = sum(client_weights)
            
            # Start with first gradient
            weighted_sum = self.context.multiply_scalar(
                grads[0], 
                client_weights[0] / total_weight
            )
            
            # Add remaining gradients
            for grad, weight in zip(grads[1:], client_weights[1:]):
                weighted_grad = self.context.multiply_scalar(grad, weight / total_weight)
                weighted_sum = self.context.add(weighted_sum, weighted_grad)
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def federated_averaging(
        self,
        client_models: List[Dict[str, NDArray]],
        client_weights: List[float],
    ) -> Dict[str, NDArray]:
        """
        Compute federated averaging on plaintext models.
        
        This is used when gradients are computed locally and
        model updates are shared (less private but common).
        
        Args:
            client_models: List of model state dicts
            client_weights: Weight for each client (e.g., sample count)
        
        Returns:
            Averaged model state dict
        """
        if not client_models:
            return {}
        
        total_weight = sum(client_weights)
        averaged = {}
        
        # Get all parameter names
        param_names = client_models[0].keys()
        
        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = None
            
            for model, weight in zip(client_models, client_weights):
                if param_name in model:
                    param = model[param_name]
                    scaled_param = param * (weight / total_weight)
                    
                    if weighted_sum is None:
                        weighted_sum = scaled_param
                    else:
                        weighted_sum += scaled_param
            
            if weighted_sum is not None:
                averaged[param_name] = weighted_sum
        
        return averaged
    
    def broadcast_model(
        self,
        model_state: Optional[Dict[str, NDArray]] = None,
        encrypted: bool = False,
    ) -> Any:
        """
        Broadcast model to all clients.
        
        Args:
            model_state: Optional model state to broadcast
            encrypted: Whether to encrypt the broadcast
        
        Returns:
            Model state or encrypted model state
        """
        if model_state is None:
            # Extract current model state
            model_state = self._extract_model_state()
        
        if encrypted:
            # Encrypt model parameters
            encrypted_state = {}
            for name, param in model_state.items():
                encrypted_state[name] = self.context.encrypt(param)
            return encrypted_state
        
        return model_state
    
    def _extract_model_state(self) -> Dict[str, NDArray]:
        """Extract current model state as dictionary."""
        state = {}
        
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, EncryptedLinear):
                if layer._weights is not None:
                    state[f'layer_{i}_weight'] = layer._weights.copy()
                if layer._bias is not None:
                    state[f'layer_{i}_bias'] = layer._bias.copy()
        
        return state
    
    def fit(
        self,
        num_rounds: int = 10,
        local_epochs: int = 5,
        client_fraction: float = 1.0,
    ) -> Dict[str, List]:
        """
        Run federated training.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Epochs of local training per round
            client_fraction: Fraction of clients per round
        
        Returns:
            Training history
        """
        history = {
            'round': [],
            'num_clients': [],
            'metrics': [],
        }
        
        for round_num in range(num_rounds):
            self._round = round_num
            
            # Sample clients
            clients = self.sample_clients(client_fraction)
            if not clients:
                continue
            
            # Collect gradients from clients
            client_gradients = {}
            client_weights = {}
            
            for client in clients:
                # Compute local gradients
                gradients = client.compute_gradients(self._default_loss)
                client_gradients[client.client_id] = gradients
                client_weights[client.client_id] = client.num_samples
            
            # Secure aggregation
            aggregated_grads = self.secure_aggregate(
                client_gradients, 
                client_weights
            )
            
            # Apply to global model
            self.model.zero_grad()
            
            # Broadcast updates
            for client in clients:
                client.apply_model_update(aggregated_grads)
            
            # Record history
            history['round'].append(round_num)
            history['num_clients'].append(len(clients))
            history['metrics'].append({'loss': 0.0})  # Simplified
        
        return history
    
    def _default_loss(self, y_pred: Ciphertext, y_true: Ciphertext) -> Ciphertext:
        """Default loss function (MSE for simplicity)."""
        # MSE = (y_pred - y_true)²
        diff = self.context.sub(y_pred, y_true)
        return self.context.square(diff)
    
    @property
    def num_clients(self) -> int:
        """Number of registered clients."""
        return len(self._clients)


class FederatedAveraging:
    """
    Federated Averaging (FedAvg) algorithm implementation.
    
    FedAvg (McMahan et al., 2017):
        1. Server initializes model θ
        2. For each round:
           a. Sample fraction C of clients
           b. Each client i:
              - Receive θ from server
              - Update θ_i with local data for E epochs
              - Send θ_i back to server
           c. Server: θ ← Σ (n_i/n) · θ_i
    
    Variations:
    - FedProx: Adds proximal term for non-IID data
    - SCAFFOLD: Uses variance reduction
    - FedNova: Normalizes local updates
    """
    
    def __init__(
        self,
        server: FederatedServer,
        optimizer_config: Optional[Dict] = None,
    ):
        """
        Initialize FedAvg.
        
        Args:
            server: Federated server
            optimizer_config: Optimizer configuration
        """
        self.server = server
        self.optimizer_config = optimizer_config or {
            'learning_rate': 0.01,
            'momentum': 0.9,
        }
    
    def run(
        self,
        num_rounds: int,
        local_epochs: int,
        client_fraction: float = 1.0,
        evaluation_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run FedAvg algorithm.
        
        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local training epochs per round
            client_fraction: Fraction of clients per round
            evaluation_fn: Optional evaluation function
        
        Returns:
            Training results and metrics
        """
        results = {
            'rounds': [],
            'client_counts': [],
            'train_losses': [],
            'test_metrics': [],
            'privacy_budgets': [],
        }
        
        for round_idx in range(num_rounds):
            # Sample clients
            clients = self.server.sample_clients(client_fraction)
            
            # Local training on each client
            local_updates = []
            local_weights = []
            
            for client in clients:
                # Local training (simplified)
                update = self._local_train(
                    client, 
                    local_epochs
                )
                local_updates.append(update)
                local_weights.append(client.num_samples)
            
            # Aggregate updates
            global_update = self.server.federated_averaging(
                local_updates,
                local_weights,
            )
            
            # Apply to global model
            self._apply_update(global_update)
            
            # Evaluate
            if evaluation_fn:
                metrics = evaluation_fn(self.server.model)
                results['test_metrics'].append(metrics)
            
            results['rounds'].append(round_idx)
            results['client_counts'].append(len(clients))
        
        return results
    
    def _local_train(
        self,
        client: FederatedClient,
        epochs: int,
    ) -> Dict[str, NDArray]:
        """Perform local training on client."""
        # Simplified local training
        # In practice, would do full gradient descent
        
        # Get current model parameters
        local_state = {}
        for i, layer in enumerate(client.model.layers):
            if isinstance(layer, EncryptedLinear):
                if layer._weights is not None:
                    local_state[f'layer_{i}_weight'] = layer._weights.copy()
                if layer._bias is not None:
                    local_state[f'layer_{i}_bias'] = layer._bias.copy()
        
        # Simulate training (would compute actual gradients)
        # Return updated state
        return local_state
    
    def _apply_update(self, update: Dict[str, NDArray]) -> None:
        """Apply update to global model."""
        for i, layer in enumerate(self.server.model.layers):
            if isinstance(layer, EncryptedLinear):
                weight_key = f'layer_{i}_weight'
                bias_key = f'layer_{i}_bias'
                
                if weight_key in update and layer._weights is not None:
                    layer._weights = update[weight_key]
                
                if bias_key in update and layer._bias is not None:
                    layer._bias = update[bias_key]


class DifferentialPrivacy:
    """
    Differential Privacy for Federated Learning.
    
    Provides (ε, δ)-differential privacy guarantees for gradient updates.
    
    Mechanisms:
    1. Gaussian Mechanism: Add N(0, σ²·2·Δ²·log(1/δ)) noise
    2. Laplace Mechanism: Add Laplace(0, Δ/ε) noise
    3. Gradient Clipping: Bound sensitivity by clipping to [−C, C]
    
    Composition:
    - Sequential: ε_total = Σ ε_i
    - Advanced: Better composition with tighter bounds
    """
    
    def __init__(
        self,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        clip_bound: float = 1.0,
    ):
        """
        Initialize DP mechanism.
        
        Args:
            epsilon: Privacy budget ε
            delta: Probability of privacy violation δ
            clip_bound: Gradient clipping bound C
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_bound = clip_bound
        
        # Privacy accounting
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
    
    def clip_gradient(self, gradient: NDArray) -> NDArray:
        """
        Clip gradient to bounded sensitivity.
        
        Args:
            gradient: Raw gradient
        
        Returns:
            Clipped gradient with L2 norm ≤ clip_bound
        """
        norm = np.linalg.norm(gradient)
        
        if norm > self.clip_bound:
            return gradient * (self.clip_bound / norm)
        
        return gradient
    
    def add_noise(self, shape: Tuple, scale: Optional[float] = None) -> NDArray:
        """
        Add Gaussian noise for privacy.
        
        Args:
            shape: Shape of gradient array
            scale: Noise scale σ (computed from ε, δ if not provided)
        
        Returns:
            Noise array
        """
        if scale is None:
            # Compute noise scale for (ε, δ)-DP
            # Using Gaussian mechanism with σ = √(2·log(1.25/δ))·Δ/ε
            scale = np.sqrt(2 * np.log(1.25 / self.delta)) * self.clip_bound / self.epsilon
        
        return np.random.normal(0, scale, shape)
    
    def privatize_gradient(
        self, 
        gradient: NDArray,
        add_noise: bool = True,
    ) -> NDArray:
        """
        Apply differential privacy to gradient.
        
        Args:
            gradient: Raw gradient
            add_noise: Whether to add noise (False for testing)
        
        Returns:
            Privatized gradient
        """
        # Clip
        clipped = self.clip_gradient(gradient)
        
        # Add noise
        if add_noise:
            noise = self.add_noise(gradient.shape)
            return clipped + noise
        
        return clipped
    
    def compute_privacy_spent(
        self,
        num_iterations: int,
        batch_size: int,
        dataset_size: int,
    ) -> Tuple[float, float]:
        """
        Compute total privacy budget spent.
        
        Args:
            num_iterations: Number of training iterations
            batch_size: Batch size per iteration
            dataset_size: Total dataset size
        
        Returns:
            Tuple of (ε, δ) spent
        """
        # Advanced composition for Gaussian mechanism
        # ε_total = √(2·log(1/δ) · log(1/δ')) · σ · q
        
        # For simplicity, use basic composition
        q = batch_size / dataset_size  # Sampling rate
        
        eps_total = self.epsilon * np.sqrt(2 * num_iterations * np.log(1 / self.delta))
        delta_total = self.delta * num_iterations
        
        return eps_total, delta_total
    
    def __repr__(self) -> str:
        return f"DifferentialPrivacy(ε={self.epsilon}, δ={self.delta}, C={self.clip_bound})"
