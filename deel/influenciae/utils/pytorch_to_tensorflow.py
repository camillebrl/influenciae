# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
# =====================================================================================
"""
Utilities for converting PyTorch models to TensorFlow for use with influenciae.

This module provides functions to convert simple PyTorch models (MLPs, linear layers)
to their TensorFlow/Keras equivalents, enabling the use of PyTorch-trained models
with the influenciae library.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from ..types import Dict, List, Optional, Tuple, Union

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch_available():
    """Raise an error if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for model conversion. "
            "Install it with: pip install torch"
        )


def convert_linear_weights(
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Convert PyTorch Linear layer weights to Keras Dense format.

    PyTorch Linear: weight shape (out_features, in_features)
    Keras Dense: weight shape (in_features, out_features)

    Parameters
    ----------
    weight
        PyTorch weight matrix of shape (out_features, in_features)
    bias
        Optional bias vector of shape (out_features,)

    Returns
    -------
    List of weights in Keras format: [weight.T, bias] or [weight.T]
    """
    keras_weight = weight.T  # Transpose for Keras
    if bias is not None:
        return [keras_weight, bias]
    return [keras_weight]


def mlp_pytorch_to_tensorflow(
    pytorch_state_dict: Dict[str, "torch.Tensor"],
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = None,
    layer_prefix: str = "net"
) -> Model:
    """
    Convert a PyTorch MLP state_dict to a Keras Model.

    Assumes the PyTorch MLP has the structure:
    - Linear layers named: {layer_prefix}.{idx}.weight, {layer_prefix}.{idx}.bias
    - Where idx are even numbers (0, 2, 4, ...) for linear layers
      (odd numbers being activations/dropout in nn.Sequential)

    Parameters
    ----------
    pytorch_state_dict
        The state_dict from the PyTorch model (can be dict of tensors or numpy arrays)
    input_dim
        Input dimension of the MLP
    hidden_dims
        List of hidden layer dimensions
    output_dim
        Output dimension (number of classes for classification)
    activation
        Activation function for hidden layers (default: "relu")
    output_activation
        Activation function for output layer (default: None for logits)
    layer_prefix
        Prefix used in PyTorch state_dict keys (default: "net")

    Returns
    -------
    A Keras Model equivalent to the PyTorch MLP

    Example
    -------
    >>> # PyTorch MLP: 100 -> 64 -> 13
    >>> checkpoint = torch.load("model.pt")
    >>> tf_model = mlp_pytorch_to_tensorflow(
    ...     checkpoint["model_state_dict"],
    ...     input_dim=100,
    ...     hidden_dims=[64],
    ...     output_dim=13
    ... )
    """
    _check_torch_available()

    # Convert tensors to numpy if needed
    state_dict = {}
    for key, value in pytorch_state_dict.items():
        if hasattr(value, 'numpy'):
            state_dict[key] = value.numpy()
        else:
            state_dict[key] = value

    # Build the Keras model
    inputs = layers.Input(shape=(input_dim,), name="input")
    x = inputs

    all_dims = [input_dim] + hidden_dims + [output_dim]
    layer_idx = 0  # PyTorch sequential index (0, 2, 4, ...)

    for i, (in_dim, out_dim) in enumerate(zip(all_dims[:-1], all_dims[1:])):
        is_output_layer = (i == len(all_dims) - 2)

        # Get weights from state_dict
        weight_key = f"{layer_prefix}.{layer_idx}.weight"
        bias_key = f"{layer_prefix}.{layer_idx}.bias"

        if weight_key not in state_dict:
            raise KeyError(
                f"Weight key '{weight_key}' not found in state_dict. "
                f"Available keys: {list(state_dict.keys())}"
            )

        weight = state_dict[weight_key]
        bias = state_dict.get(bias_key)

        # Determine activation
        if is_output_layer:
            act = output_activation
            layer_name = "output"
        else:
            act = activation
            layer_name = f"hidden_{i}"

        # Create Dense layer
        dense = layers.Dense(
            out_dim,
            activation=act,
            use_bias=(bias is not None),
            name=layer_name
        )

        # Build and set weights
        x = dense(x)
        dense.set_weights(convert_linear_weights(weight, bias))

        # Move to next linear layer in PyTorch sequential
        # (skip activation/dropout layers which are at odd indices)
        layer_idx += 2 if not is_output_layer else 0
        if is_output_layer:
            # For output layer, increment depends on structure
            # Try common patterns
            pass

    # Adjust layer_idx for output layer
    # Find the actual output layer index
    for key in state_dict.keys():
        if 'weight' in key:
            idx = int(key.split('.')[1])
            if idx > layer_idx - 2:
                layer_idx = idx

    # Set output layer weights if not already set
    output_weight_key = f"{layer_prefix}.{layer_idx}.weight"
    if output_weight_key in state_dict and layer_idx > 0:
        output_weight = state_dict[output_weight_key]
        output_bias = state_dict.get(f"{layer_prefix}.{layer_idx}.bias")
        # Find output layer and set weights
        for layer in inputs._keras_history[0].layers:
            if layer.name == "output":
                layer.set_weights(convert_linear_weights(output_weight, output_bias))
                break

    model = Model(inputs=inputs, outputs=x, name="converted_mlp")
    return model


def simple_mlp_pytorch_to_tensorflow(
    checkpoint_path: str,
    state_dict_key: str = "model_state_dict",
    config_keys: Optional[Dict[str, str]] = None
) -> Tuple[Model, Dict]:
    """
    Load a PyTorch MLP checkpoint and convert it to TensorFlow.

    This is a convenience function that handles common checkpoint formats
    where the checkpoint contains both the model state_dict and configuration.

    Parameters
    ----------
    checkpoint_path
        Path to the PyTorch checkpoint file (.pt or .pth)
    state_dict_key
        Key in the checkpoint dict containing the state_dict (default: "model_state_dict")
    config_keys
        Optional dict mapping config names to checkpoint keys.
        Default: {"emb_dim": "emb_dim", "hidden_dim": "hidden_dim", "num_classes": "num_classes"}

    Returns
    -------
    Tuple of (Keras Model, config dict with metadata from checkpoint)

    Example
    -------
    >>> tf_model, config = simple_mlp_pytorch_to_tensorflow("model.pt")
    >>> print(config["label_names"])  # If available in checkpoint
    """
    _check_torch_available()

    # Default config keys
    if config_keys is None:
        config_keys = {
            "input_dim": "emb_dim",
            "hidden_dim": "hidden_dim",
            "output_dim": "num_classes"
        }

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract config
    config = {}
    input_dim = checkpoint.get(config_keys.get("input_dim", "emb_dim"))
    hidden_dim = checkpoint.get(config_keys.get("hidden_dim", "hidden_dim"))
    output_dim = checkpoint.get(config_keys.get("output_dim", "num_classes"))

    if input_dim is None or hidden_dim is None or output_dim is None:
        raise ValueError(
            f"Could not find model dimensions in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    # Store all non-state_dict items in config
    for key, value in checkpoint.items():
        if key != state_dict_key:
            config[key] = value

    # Convert model
    state_dict = checkpoint[state_dict_key]

    # Infer architecture from state_dict
    hidden_dims = []
    layer_dims = []
    for key in sorted(state_dict.keys()):
        if 'weight' in key:
            weight = state_dict[key]
            if hasattr(weight, 'shape'):
                layer_dims.append(weight.shape)

    # Extract hidden dims (all except last layer's output)
    if layer_dims:
        for i, (out_dim, in_dim) in enumerate(layer_dims):
            if i < len(layer_dims) - 1:
                hidden_dims.append(out_dim)

    tf_model = mlp_pytorch_to_tensorflow(
        state_dict,
        input_dim=input_dim,
        hidden_dims=hidden_dims if hidden_dims else [hidden_dim],
        output_dim=output_dim
    )

    return tf_model, config


def verify_conversion(
    pytorch_model: "nn.Module",
    tf_model: Model,
    input_shape: Tuple[int, ...],
    num_samples: int = 10,
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> bool:
    """
    Verify that a converted TensorFlow model produces the same outputs as the PyTorch original.

    Parameters
    ----------
    pytorch_model
        The original PyTorch model
    tf_model
        The converted TensorFlow model
    input_shape
        Shape of a single input sample (without batch dimension)
    num_samples
        Number of random samples to test
    rtol
        Relative tolerance for comparison
    atol
        Absolute tolerance for comparison

    Returns
    -------
    True if outputs match within tolerance, raises AssertionError otherwise
    """
    _check_torch_available()

    pytorch_model.eval()

    # Generate random inputs
    np.random.seed(42)
    test_inputs = np.random.randn(num_samples, *input_shape).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        pt_inputs = torch.from_numpy(test_inputs)
        pt_outputs = pytorch_model(pt_inputs).numpy()

    # TensorFlow forward
    tf_outputs = tf_model.predict(test_inputs, verbose=0)

    # Compare
    np.testing.assert_allclose(
        tf_outputs, pt_outputs, rtol=rtol, atol=atol,
        err_msg="TensorFlow and PyTorch outputs do not match!"
    )

    return True
