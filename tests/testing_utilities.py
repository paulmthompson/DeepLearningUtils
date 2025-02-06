import numpy as np

def assert_arrays_equal_with_nans(arr1, arr2, rtol=1e-05, atol=1e-08):
    """
    Asserts that two NumPy arrays are equal, handling NaNs correctly.

    Args:
        arr1: The first NumPy array.
        arr2: The second NumPy array.
        rtol: Relative tolerance (see np.allclose).
        atol: Absolute tolerance (see np.allclose).
    """

    # Check for shape mismatch first
    assert arr1.shape == arr2.shape, f"Shapes don't match: {arr1.shape} vs {arr2.shape}"

    # Check for NaNs
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)

    assert np.array_equal(nan_mask1, nan_mask2), "NaN locations don't match"

    # Check for Inf values
    inf_mask1 = np.isinf(arr1)
    inf_mask2 = np.isinf(arr2)
    assert np.array_equal(inf_mask1, inf_mask2), "Inf locations don't match"


    # Compare non-NaN values using np.allclose (for tolerance)
    #  We use a mask to compare ONLY the non-NaN and non-Inf values.
    non_nan_non_inf_mask = ~(nan_mask1 | inf_mask1)
    assert np.allclose(arr1[non_nan_non_inf_mask], arr2[non_nan_non_inf_mask], rtol=rtol, atol=atol), \
        "Arrays are not close within tolerance (excluding NaNs and Infs)"

    # Check Inf values using np.array_equal (for exact value match (+inf or -inf))
    assert np.array_equal(arr1[inf_mask1], arr2[inf_mask2]), \
        "Inf values are not equal"

    # Optional: Count NaNs (useful for debugging)
    num_nans = np.count_nonzero(nan_mask1)
    print(f"Number of NaNs found: {num_nans}")
