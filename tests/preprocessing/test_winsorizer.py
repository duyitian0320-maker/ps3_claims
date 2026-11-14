import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    # Create and fit Winsorizer
    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    winsorizer.fit(X)

    # Transform the data
    X_transformed = winsorizer.transform(X)

    # Calculate expected quantile thresholds
    lower = np.quantile(X, lower_quantile)
    upper = np.quantile(X, upper_quantile)

    # Check that values are clipped correctly
    assert np.all(X_transformed >= lower)
    assert np.all(X_transformed <= upper)

    # Check that the transformation matches expected output
    assert np.allclose(X_transformed, np.clip(X, lower, upper))

