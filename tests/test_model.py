import pytest
import torch
from mlbrs.model import Model


def test_model_forward_pass_shape():
    """Test forward pass produces correct output shape."""
    model = Model(
        conv1_args=[1, 32, 3, 1],
        conv2_args=[32, 64, 3, 1],
        fc1_args=[9216, 128],
        fc2_args=[128, 10],
    )
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    assert output.shape == (2, 10)


def test_model_forward_pass_is_log_softmax():
    """Test that forward pass returns log probabilities."""
    model = Model(
        conv1_args=[1, 32, 3, 1],
        conv2_args=[32, 64, 3, 1],
        fc1_args=[9216, 128],
        fc2_args=[128, 10],
    )
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    log_probs_sum = torch.exp(output).sum(dim=1)
    assert torch.allclose(log_probs_sum, torch.ones(2), atol=1e-5)


def test_model_dropout_rates():
    """Test that custom dropout rates are applied."""
    model = Model(
        conv1_args=[1, 32, 3, 1],
        conv2_args=[32, 64, 3, 1],
        fc1_args=[9216, 128],
        fc2_args=[128, 10],
        dropout_rate1=0.3,
        dropout_rate2=0.7,
    )
    assert model.dropout1.p == 0.3
    assert model.dropout2.p == 0.7


def test_model_dropout_effects_training_mode():
    """Test that dropout affects outputs in training mode but not eval."""
    model = Model(
        conv1_args=[1, 32, 3, 1],
        conv2_args=[32, 64, 3, 1],
        fc1_args=[9216, 128],
        fc2_args=[128, 10],
        dropout_rate1=0.9,
    )
    x = torch.randn(1, 1, 28, 28)

    model.train()
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2)

    model.eval()
    out3 = model(x)
    out4 = model(x)
    assert torch.allclose(out3, out4)


def test_model_from_config_with_custom_dropout():
    """Test creating model from config with custom dropout rates."""
    config = {
        "conv1_args": [1, 32, 3, 1],
        "conv2_args": [32, 64, 3, 1],
        "fc1_args": [9216, 128],
        "fc2_args": [128, 10],
        "dropout_rate1": 0.4,
        "dropout_rate2": 0.6,
    }
    model = Model.from_config(config)
    assert model.dropout1.p == 0.4
    assert model.dropout2.p == 0.6


def test_model_with_layer_kwargs():
    """Test that layer kwargs are properly applied."""
    model = Model(
        conv1_args=[1, 32, 3, 1],
        conv2_args=[32, 64, 3, 1],
        fc1_args=[9216, 128],
        fc2_args=[128, 10],
        conv1_kwargs={"padding": 1},
        fc1_kwargs={"bias": False},
    )
    assert model.conv1.padding == (1, 1)
    assert model.fc1.bias is None
