import pytest
import keras


@pytest.fixture(scope="function")
def keras_float32_policy():
    """
    Fixture to set Keras dtype policy to float32 for a test and restore it.

    It seems that some keras layers or operations
    maintain some kind of global mixed precision state and this
    fixture ensures that the tests run with float32 precision.
    """
    original_policy = keras.mixed_precision.dtype_policy().name
    keras.mixed_precision.set_dtype_policy("float32")
    yield
    keras.mixed_precision.set_dtype_policy(original_policy)