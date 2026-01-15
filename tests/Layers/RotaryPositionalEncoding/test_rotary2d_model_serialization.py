"""
Tests for model serialization/deserialization of Rotary Positional Encoding.

This module tests that models with RotaryPositionalEncoding2D layers can be saved
and loaded using keras.model.save() and keras.models.load_model().
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import keras

from src.DeepLearningUtils.Layers.RotaryPositionalEncoding.rotary_config import (
    RotaryPositionalEncodingConfig
)
from src.DeepLearningUtils.Layers.RotaryPositionalEncoding.rotary2d_keras import (
    RotaryPositionalEncoding2D
)


class TestRotaryModelSerialization:
    """Test suite for model serialization of RotaryPositionalEncoding2D."""

    def test_model_save_and_load(self) -> None:
        """Test saving and loading a model with RotaryPositionalEncoding2D using PyTorch backend."""
        # Create a temporary directory for saving the model
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "rotary_model.keras")

        try:
            # Create a simple model with RotaryPositionalEncoding2D
            config = RotaryPositionalEncodingConfig(dim=64, height=8, width=8)

            inputs = keras.Input(shape=(5, 64, 64))  # (seq_len, height*width, dim)
            x = RotaryPositionalEncoding2D(config=config)(inputs)
            model = keras.Model(inputs=inputs, outputs=x)

            # Create sample input
            np.random.seed(42)
            sample_input = np.random.randn(2, 5, 64, 64).astype(np.float32)

            # Get output before saving
            output_before = model.predict(sample_input, verbose=0)

            # Save the model
            model.save(model_path)
            print(f"Model saved to {model_path}")

            # Register custom object for loading
            custom_objects = {"RotaryPositionalEncoding2D": RotaryPositionalEncoding2D}

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )
            print("Model loaded successfully")

            # Get output after loading
            output_after = loaded_model.predict(sample_input, verbose=0)

            # Verify shapes match
            assert output_before.shape == output_after.shape, (
                f"Shape mismatch: Before save shape {output_before.shape}, "
                f"After load shape {output_after.shape}"
            )

            # Verify outputs are identical (should be exact since same backend)
            np.testing.assert_allclose(
                output_before,
                output_after,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Outputs differ after model save/load"
            )

            # Verify weights match
            original_weights = model.get_weights()
            loaded_weights = loaded_model.get_weights()

            assert len(original_weights) == len(loaded_weights), (
                f"Weight count mismatch: {len(original_weights)} vs {len(loaded_weights)}"
            )

            for i, (w_orig, w_loaded) in enumerate(zip(original_weights, loaded_weights)):
                np.testing.assert_allclose(
                    w_orig,
                    w_loaded,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"Weight {i} differs after load"
                )

            print("Model save/load test passed!")

        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_model_save_load_with_different_dimensions(self) -> None:
        """Test model save/load with different spatial dimensions."""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "rotary_model_16x16.keras")

        try:
            # Create model with different dimensions
            config = RotaryPositionalEncodingConfig(dim=128, height=16, width=16)

            inputs = keras.Input(shape=(10, 256, 128))  # (seq_len, height*width, dim)
            x = RotaryPositionalEncoding2D(config=config)(inputs)
            model = keras.Model(inputs=inputs, outputs=x)

            # Create sample input
            np.random.seed(123)
            sample_input = np.random.randn(1, 10, 256, 128).astype(np.float32)

            # Get output before saving
            output_before = model.predict(sample_input, verbose=0)

            # Save and load
            model.save(model_path)
            custom_objects = {"RotaryPositionalEncoding2D": RotaryPositionalEncoding2D}
            loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

            # Get output after loading
            output_after = loaded_model.predict(sample_input, verbose=0)

            # Verify
            np.testing.assert_allclose(
                output_before,
                output_after,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Outputs differ for 16x16 model"
            )

            print("Model save/load test with 16x16 dimensions passed!")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_model_in_larger_architecture(self) -> None:
        """Test model save/load when RotaryPositionalEncoding2D is part of a larger model."""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "complex_model.keras")

        try:
            # Create a more complex model
            config = RotaryPositionalEncodingConfig(dim=64, height=8, width=8)

            inputs = keras.Input(shape=(5, 64, 64))

            # Add some layers before rotary encoding
            x = keras.layers.Dense(64)(inputs)
            x = keras.layers.LayerNormalization()(x)

            # Apply rotary encoding
            x = RotaryPositionalEncoding2D(config=config)(x)

            # Add some layers after
            x = keras.layers.Dense(32)(x)
            outputs = keras.layers.Dense(64)(x)

            model = keras.Model(inputs=inputs, outputs=outputs)

            # Create sample input
            np.random.seed(456)
            sample_input = np.random.randn(2, 5, 64, 64).astype(np.float32)

            # Get output before saving
            output_before = model.predict(sample_input, verbose=0)

            # Save and load
            model.save(model_path)
            custom_objects = {"RotaryPositionalEncoding2D": RotaryPositionalEncoding2D}
            loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

            # Get output after loading
            output_after = loaded_model.predict(sample_input, verbose=0)

            # Verify
            np.testing.assert_allclose(
                output_before,
                output_after,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Outputs differ for complex model"
            )

            print("Complex model save/load test passed!")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_model_load_with_cuda_tensor(self) -> None:
        """Test that model loading works even with CUDA tensors (simulates the real error case)."""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "rotary_model_cuda.keras")

        try:
            # Create and save a model
            config = RotaryPositionalEncodingConfig(dim=384, height=16, width=16)

            inputs = keras.Input(shape=(1, 256, 384))  # Similar to the error case
            x = RotaryPositionalEncoding2D(config=config)(inputs)
            model = keras.Model(inputs=inputs, outputs=x)

            # Save the model
            model.save(model_path)
            print(f"Model saved to {model_path}")

            # Clear the model from memory
            del model

            # Register custom object for loading
            custom_objects = {"RotaryPositionalEncoding2D": RotaryPositionalEncoding2D}

            # Load the model - this should work without errors
            # The error happens during deserialization when Keras tries to infer output shapes
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )
            print("Model loaded successfully (this tests the deserialization issue)")

            # Now test with actual data
            np.random.seed(789)
            sample_input = np.random.randn(2, 1, 256, 384).astype(np.float32)
            output = loaded_model.predict(sample_input, verbose=0)

            assert output.shape == (2, 1, 256, 384), f"Unexpected output shape: {output.shape}"

            print("CUDA tensor test passed!")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_model_save_load_with_non_default_parameters(self) -> None:
        """Test that non-default parameters are preserved after save/load."""
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "rotary_non_default.keras")

        try:
            # Non-default parameters
            dim = 32
            height = 4
            width = 4
            theta = 100.0  # default is 10.0
            rotate = False  # default is True
            max_freq = 128  # default is 64

            # Create model with non-default parameters
            inputs = keras.Input(shape=(2, height * width, dim))
            x = RotaryPositionalEncoding2D(
                dim=dim,
                height=height,
                width=width,
                theta=theta,
                rotate=rotate,
                max_freq=max_freq
            )(inputs)
            model = keras.Model(inputs=inputs, outputs=x)

            # Create sample input
            np.random.seed(42)
            sample_input = np.random.randn(1, 2, height * width, dim).astype(np.float32)

            # Get output before saving
            output_before = model.predict(sample_input, verbose=0)

            # Save the model
            model.save(model_path)

            # Load the model
            custom_objects = {"RotaryPositionalEncoding2D": RotaryPositionalEncoding2D}
            loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)

            # Get output after loading
            output_after = loaded_model.predict(sample_input, verbose=0)

            # Verify outputs match
            np.testing.assert_allclose(
                output_before,
                output_after,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Outputs differ after save/load with non-default parameters"
            )

            # Find the loaded layer and verify parameters
            loaded_layer = None
            for layer in loaded_model.layers:
                if isinstance(layer, RotaryPositionalEncoding2D):
                    loaded_layer = layer
                    break

            assert loaded_layer is not None, "RotaryPositionalEncoding2D layer not found in loaded model"
            assert loaded_layer.config.dim == dim, f"Expected dim={dim}, got {loaded_layer.config.dim}"
            assert loaded_layer.config.height == height, f"Expected height={height}, got {loaded_layer.config.height}"
            assert loaded_layer.config.width == width, f"Expected width={width}, got {loaded_layer.config.width}"
            assert loaded_layer.config.theta == theta, f"Expected theta={theta}, got {loaded_layer.config.theta}"
            assert loaded_layer.config.rotate == rotate, f"Expected rotate={rotate}, got {loaded_layer.config.rotate}"
            assert loaded_layer.config.max_freq == max_freq, f"Expected max_freq={max_freq}, got {loaded_layer.config.max_freq}"

            print("Non-default parameters test passed!")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

