
import hashlib
import numpy as np
from pandas.api.types import is_numeric_dtype

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.

def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

    df = df.copy()

    if not 0 <= training_frac <= 1:
        raise ValueError("training_frac must be between 0 and 1")

    # If ID is numeric, use modulo directly
    if is_numeric_dtype(df[id_column]):
        buckets = df[id_column] % 100

    # If ID is string/object, first hash it into an integer
    else:
        hashed_ids = df[id_column].astype(str).apply(
            lambda x: int(hashlib.sha256(x.encode()).hexdigest(), 16)
        )
        buckets = hashed_ids % 100

    df["sample"] = np.where(
        buckets < training_frac * 100,
        "train",
        "test"
    )

    return df
