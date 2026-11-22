

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_tweedie_deviance,
    mean_absolute_error,
    mean_squared_error,
    auc,
)


def evaluate_predictions(y_true, y_pred, sample_weight):
    """Calculate exposure-weighted evaluation metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(sample_weight)

    # 1. Bias
    bias = np.average(y_pred - y_true, weights=weights)

    # 2. Tweedie deviance
    deviance = mean_tweedie_deviance(
        y_true, y_pred,
        sample_weight=weights,
        power=1.5
    )

    # 3. Mean Absolute Error
    mae = mean_absolute_error(
        y_true, y_pred,
        sample_weight=weights
    )

    # 4. Root Mean Squared Error
    rmse = np.sqrt(
        mean_squared_error(
            y_true, y_pred,
            sample_weight=weights
        )
    )

    # 5. Gini coefficient (same definition as PS3)
    ranking = np.argsort(y_pred)

    ranked_exposure = weights[ranking]
    ranked_pure_premium = y_true[ranking]

    cumulative_claims = np.cumsum(
        ranked_pure_premium * ranked_exposure
    )
    cumulative_claims /= cumulative_claims[-1]

    cumulative_samples = np.linspace(
        0, 1, len(cumulative_claims)
    )

    gini = 1 - 2 * auc(
        cumulative_samples, cumulative_claims
    )

    # 6. Return results as DataFrame
    return pd.DataFrame(
        {
            "Value": [
                bias,
                deviance,
                mae,
                rmse,
                gini
            ]
        },
        index=[
            "Bias",
            "Tweedie Deviance",
            "MAE",
            "RMSE",
            "Gini"
        ]
    )
