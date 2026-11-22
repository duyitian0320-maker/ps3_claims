# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.metrics import make_scorer, mean_tweedie_deviance


from ps3.data import create_sample_split, load_transform

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
# Because different insurance policies may cover different lengths of time. Dividing by exposure makes the outcomes comparable. Later, we'll use exposure as a sample weight when fitting the Tweedie model.

# TODO: use your create_sample_split function here
df = create_sample_split(df, id_column="IDpol", training_frac=0.8)

train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]

# 1. Numeric pipeline: scaling followed by splines
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("splines", SplineTransformer(
        knots="quantile",
        include_bias=False
    )),
])

# 2. Apply different transformations by column
preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here
        ("num", numeric_pipeline, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)

preprocessor.set_output(transform="pandas")

# 3. Combine preprocessing and GLM
model_pipeline = Pipeline([("preprocess", preprocessor),
    # TODO: Define pipeline steps here
    ("estimate", GeneralizedLinearRegressor(
        family=TweedieDist,
        l1_ratio=1,
        fit_intercept=True
    )),    
])

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

model_pipeline = Pipeline([
    ("estimate", LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        random_state=42,
        verbosity=-1
    ))
])

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
cv = GridSearchCV(
    estimator=model_pipeline,
    param_grid={
        "estimate__learning_rate": [0.03, 0.1],
        "estimate__n_estimators": [100, 300],
    },
    scoring=make_scorer(
        mean_tweedie_deviance,
        greater_is_better=False,
        power=1.5
    ),
    cv=3,
    n_jobs=2,
    verbose=2
)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Optional: LightGBM frequency and severity models

# 1. Frequency model (Poisson)
y_freq = df_train["ClaimNb"] / df_train["Exposure"]

freq_model = LGBMRegressor(
    objective="poisson",
    random_state=42,
    verbosity=-1
)

freq_model.fit(
    X_train_t, y_freq,
    sample_weight=df_train["Exposure"]
)

# 2. Severity model (Gamma)
# Only policies with positive claims
positive = df_train["ClaimNb"] > 0

y_sev = (
    df_train.loc[positive, "ClaimAmountCut"]
    / df_train.loc[positive, "ClaimNb"]
)

sev_model = LGBMRegressor(
    objective="gamma",
    random_state=42,
    verbosity=-1
)

sev_model.fit(
    X_train_t.loc[positive], y_sev,
    sample_weight=df_train.loc[positive, "ClaimNb"]
)

# 3. Combine predictions
df_train["pp_freq_sev"] = (
    freq_model.predict(X_train_t)
    * sev_model.predict(X_train_t)
)

df_test["pp_freq_sev"] = (
    freq_model.predict(X_test_t)
    * sev_model.predict(X_test_t)
)

# 4. Compare deviance
for name, data in [
    ("train", df_train),
    ("test", df_test)
]:
    loss = TweedieDist.deviance(
        data["PurePremium"],
        data["pp_freq_sev"],
        sample_weight=data["Exposure"]
    ) / data["Exposure"].sum()

    print(f"{name} deviance: {loss:.4f}")
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
    ("LGBM Frequency-Severity", df_test["pp_freq_sev"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%


# Conclusion:
# Adding splines improves the benchmark GLM's predictive performance.
# LightGBM further reduces test deviance, especially after tuning.
# The optional frequency-severity LightGBM achieves the lowest
# test deviance (73.2263), outperforming the tuned Tweedie
# LightGBM (73.3544).
# However, its larger train-test gap suggests some remaining
# overfitting. Further tuning could improve generalisation.

# ============================================================
# D100 — Problem Set 4: Model Evaluation & Interpretability
# ============================================================


# ============================================================
# PS4 - Ex 1: Monotonicity Constraints
# ============================================================

# Step 1: Average claim frequency by BonusMalus
bonus_malus_group = (
    df.groupby("BonusMalus")
    .agg(
        total_claims=("ClaimNb", "sum"),
        total_exposure=("Exposure", "sum")
    )
    .reset_index()
)

bonus_malus_group["claim_frequency"] = (
    bonus_malus_group["total_claims"]
    / bonus_malus_group["total_exposure"]
)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    bonus_malus_group["BonusMalus"],
    bonus_malus_group["claim_frequency"],
    marker="o",
    markersize=3
)

plt.xlabel("BonusMalus")
plt.ylabel("Average Claim Frequency (Exposure Weighted)")
plt.title("Claim Frequency by BonusMalus")
plt.grid(alpha=0.3)
plt.show()



# %% [markdown]
# ### Ex 1 — Step 1: Interpretation
#
# Claim frequency generally increases with BonusMalus, but the relationship
# is not strictly monotonic, particularly at higher BonusMalus scores.
# The fluctuations may reflect limited exposure and sampling noise.
#
# Without a monotonicity constraint, LightGBM may learn these irregular
# patterns and predict lower claim risk for higher BonusMalus scores,
# holding other features constant. A monotonicity constraint ensures
# that predicted risk is non-decreasing as BonusMalus increases.
# %%

# %%
# Step 2: Create constrained LightGBM

from sklearn.base import clone

# Constraint vector aligned with feature order
monotone_constraints = [
    1 if feature == "BonusMalus" else 0
    for feature in X_train_t.columns
]

print("Monotonicity constraints:", monotone_constraints)

# Clone existing LightGBM pipeline
constrained_lgbm = clone(model_pipeline)

# Apply increasing constraint to BonusMalus
constrained_lgbm.set_params(
    estimate__monotone_constraints=monotone_constraints
)

print(constrained_lgbm)

# %%
# Step 3: Cross-validation and prediction

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_tweedie_deviance
from sklearn.base import clone

# Same hyperparameter grid for both models
param_grid = {
    "estimate__n_estimators": [100, 200],
    "estimate__learning_rate": [0.05, 0.1],
    "estimate__num_leaves": [15, 31]
}

# Tweedie deviance scorer
tweedie_scorer = make_scorer(
    mean_tweedie_deviance,
    power=1.5,
    greater_is_better=False
)

# Unconstrained LightGBM
unconstrained_cv = GridSearchCV(
    clone(model_pipeline),
    param_grid=param_grid,
    scoring=tweedie_scorer,
    cv=3,
    n_jobs=-1
)

# Constrained LightGBM
constrained_cv = GridSearchCV(
    constrained_lgbm,
    param_grid=param_grid,
    scoring=tweedie_scorer,
    cv=3,
    n_jobs=-1
)

# Fit both models using exposure weights
unconstrained_cv.fit(
    X_train_t, y_train_t,
    estimate__sample_weight=w_train_t
)

constrained_cv.fit(
    X_train_t, y_train_t,
    estimate__sample_weight=w_train_t
)

print("Unconstrained best parameters:")
print(unconstrained_cv.best_params_)

print("\nConstrained best parameters:")
print(constrained_cv.best_params_)

print("\nCross-validation deviance (unconstrained):",
      -unconstrained_cv.best_score_)

print("Cross-validation deviance (constrained):",
      -constrained_cv.best_score_)
# %%
import joblib

joblib.dump(unconstrained_cv, "unconstrained_cv.joblib")

print("Unconstrained model saved!")
print("Best parameters:", unconstrained_cv.best_params_)
print("CV deviance:", -unconstrained_cv.best_score_)
# %%
# %%
# PS4 Ex 1 — Step 3: Constrained LightGBM cross-validation

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

# Reduce internal parallelism
constrained_model = clone(constrained_lgbm)
constrained_model.set_params(estimate__n_jobs=2)

constrained_cv = GridSearchCV(
    estimator=constrained_model,
    param_grid=param_grid,  # Same grid as unconstrained model
    scoring=tweedie_scorer,
    cv=3,
    n_jobs=1,
    verbose=2
)

constrained_cv.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t
)

print("Constrained best parameters:", constrained_cv.best_params_)
print("Constrained CV deviance:", -constrained_cv.best_score_)

# %%
joblib.dump(constrained_cv, "constrained_cv.joblib")
print("Constrained model saved!")
# %%
print("Unconstrained CV deviance:", -unconstrained_cv.best_score_)
print("Constrained CV deviance:", -constrained_cv.best_score_)

# %%
# PS4 Ex 1 - Exposure-weighted test evaluation

from sklearn.metrics import mean_tweedie_deviance

# Predict using the best estimators
pp_t_lgbm_unconstrained = (
    unconstrained_cv.best_estimator_.predict(X_test_t)
)

pp_t_lgbm_constrained = (
    constrained_cv.best_estimator_.predict(X_test_t)
)

# Save predictions as required
df_test["pp_t_lgbm_constrained"] = pp_t_lgbm_constrained

# Calculate exposure-weighted Tweedie deviance
dev_unconstrained = mean_tweedie_deviance(
    y_test_t,
    pp_t_lgbm_unconstrained,
    sample_weight=w_test_t,
    power=1.5
)

dev_constrained = mean_tweedie_deviance(
    y_test_t,
    pp_t_lgbm_constrained,
    sample_weight=w_test_t,
    power=1.5
)

print(f"Unconstrained test deviance: {dev_unconstrained:.4f}")
print(f"Constrained test deviance:   {dev_constrained:.4f}")
print(f"Difference: {dev_constrained - dev_unconstrained:.4f}")


# %% [markdown]
# ### Ex 1 — Step 3: Model Comparison
#
# The constrained LightGBM achieves an exposure-weighted test
# Tweedie deviance of 73.6128, compared with 73.3879 for the
# unconstrained model (approximately 0.31% higher).
#
# The monotonicity constraint slightly reduces predictive accuracy,
# but ensures that predicted risk is non-decreasing with BonusMalus,
# holding other features constant.
#
# **Cross-validation and computational adjustments:**
#
# Initially, GridSearchCV used n_jobs=-1, allowing parallel model
# fitting across available CPU cores. However, the VS Code/WSL
# Interactive Window lost its connection during execution,
# potentially due to excessive computational or memory usage.
#
# To improve stability, GridSearchCV was changed to n_jobs=1,
# and LightGBM was limited to two threads (estimate__n_jobs=2).
# These adjustments reduce parallel processing without changing
# the hyperparameter search space or the underlying model.
#
# **Exposure weighting:**
#
# Exposure weights were used during model fitting, but the
# GridSearchCV scoring function did not automatically receive
# the validation weights. Therefore, CV scores were unweighted.
#
# The final test comparison explicitly uses exposure-weighted
# Tweedie deviance, which is more appropriate for insurance
# risk modelling. Fully weighted cross-validation would provide
# a more consistent model-selection procedure.
#
# Overall, imposing monotonicity produces a small loss in
# predictive accuracy while ensuring economically sensible
# model behaviour.

# %%
# PS4 - Ex 2: Learning Curve

import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.base import clone

# Refit the best constrained estimator from Ex 1
lgbm_learning = clone(constrained_cv.best_estimator_)

lgbm_learning.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[
        (X_train_t, y_train_t),
        (X_test_t, y_test_t)
    ],
    estimate__eval_names=["Training", "Test"],
    estimate__eval_sample_weight=[
        w_train_t,
        w_test_t
    ],
    estimate__eval_metric="tweedie"
)

print("Learning curve model fitted successfully!")

# %%
# Plot convergence of Tweedie deviance

ax = lgb.plot_metric(
    lgbm_learning.named_steps["estimate"],
    metric="tweedie",
    figsize=(10, 5)
)

ax.set_title("Learning Curve — Constrained LightGBM")
ax.set_xlabel("Boosting Iterations")
ax.set_ylabel("Tweedie Deviance")
ax.grid(alpha=0.3)

plt.show()


import numpy as np

results = lgbm_learning.named_steps["estimate"].evals_result_

print(results.keys())
print(results["Test"].keys())

# %%
# Identify the iteration with minimum test deviance

test_deviance = results["Test"]["tweedie"]
train_deviance = results["Training"]["tweedie"]

best_iteration = np.argmin(test_deviance) + 1

print(f"Best boosting iteration: {best_iteration}")
print(f"Minimum test deviance: {min(test_deviance):.4f}")
print(f"Test deviance at 100 iterations: {test_deviance[-1]:.4f}")
print(f"Training deviance at 100 iterations: {train_deviance[-1]:.4f}")

# %% [markdown]
# ### Ex 2 — Learning Curve Interpretation
#
# Both training and test Tweedie deviance initially decrease,
# indicating successful convergence of the constrained LightGBM.
#
# Test deviance reaches its minimum of 44.7941 at iteration 71,
# before increasing slightly to 44.8545 at iteration 100.
# Meanwhile, training deviance continues decreasing to 43.1969.
#
# This suggests mild overfitting beyond 71 iterations, as additional
# trees improve in-sample fit without improving generalisation.
#
# The estimator is reasonably well tuned, but the selected 100
# boosting iterations are slightly excessive. Early stopping or
# tuning n_estimators could improve predictive performance.
#
# The improvement from stopping earlier is relatively small,
# suggesting that overfitting is not severe.
#
# Note: The test set is used here for diagnostic purposes.
# In practice, early stopping should be based on a separate
# validation set to avoid test-set information leakage.

# %%
# PS4 Ex 3 - Test evaluation function


# %%
# Reload updated evaluation function

import importlib
import evaluation._evaluate_predictions as eval_module

importlib.reload(eval_module)

from evaluation._evaluate_predictions import evaluate_predictions

print("Evaluation function updated!")

# %%
# Compare constrained and unconstrained LightGBM
metrics_unconstrained = evaluate_predictions(
    y_test_t, pp_t_lgbm_unconstrained, w_test_t
)

metrics_constrained = evaluate_predictions(
    y_test_t, pp_t_lgbm_constrained, w_test_t
)

metrics_comparison = pd.concat(
    [
        metrics_unconstrained.rename(
            columns={"Value": "Unconstrained LGBM"}
        ),
        metrics_constrained.rename(
            columns={"Value": "Constrained LGBM"}
        )
    ],
    axis=1
)

print(metrics_comparison.round(4))

# %% [markdown]
# ### Ex 3 — Model Evaluation and Comparison
#
# The evaluation function computes exposure-weighted bias, Tweedie
# deviance, MAE and RMSE, together with the Gini coefficient using
# the same Lorenz curve definition as PS3.
#
# The unconstrained LightGBM performs slightly better in terms of
# predictive accuracy, with lower Tweedie deviance (73.3879 vs 73.6128),
# MAE (246.5490 vs 248.1539) and RMSE (3847.4438 vs 3847.5788).
#
# It also achieves a higher Gini coefficient (0.2564 vs 0.2464),
# indicating slightly better risk discrimination.
#
# However, the constrained model has a smaller absolute bias
# (-7.7009 vs -8.9301), suggesting better average calibration.
#
# Overall, imposing monotonicity on BonusMalus results in a small
# reduction in predictive accuracy and discrimination, but ensures
# economically sensible model behaviour.
#
# The reusable evaluation function allows consistent comparison
# of future models without repeatedly calculating individual metrics.

# %%
# ============================================================
# PS4 - Ex 4: Partial Dependence Plots
# ============================================================

import dalex as dx
import numpy as np
import pandas as pd

# Use the same sample for both models
X_pdp = X_train_t.sample(n=300, random_state=42)
y_pdp = y_train_t.loc[X_pdp.index]

# Constrained LightGBM
explainer_constrained = dx.Explainer(
    constrained_cv.best_estimator_,
    data=X_pdp,
    y=y_pdp,
    predict_function=lambda model, data: model.predict(data),
    label="Constrained LGBM",
    model_type="regression",
    verbose=False
)

# Unconstrained LightGBM
explainer_unconstrained = dx.Explainer(
    unconstrained_cv.best_estimator_,
    data=X_pdp,
    y=y_pdp,
    predict_function=lambda model, data: model.predict(data),
    label="Unconstrained LGBM",
    model_type="regression",
    verbose=False
)

print("Both explainers created successfully!")

# %%
# Step 2: PDP for BonusMalus

pdp_constrained = explainer_constrained.model_profile(
    variables=["BonusMalus"],
    type="partial",
    N=100,
    random_state=42
)

pdp_unconstrained = explainer_unconstrained.model_profile(
    variables=["BonusMalus"],
    type="partial",
    N=100,
    random_state=42
)

# Compare the two models
pdp_constrained.plot(
    objects=[pdp_unconstrained],
    variables=["BonusMalus"],
    title="PDP Comparison - BonusMalus"
)



# %%
# ============================================================
# PS4 - Ex 4: Partial Dependence Plots
# ============================================================

import dalex as dx
import pandas as pd
import numpy as np

numerical_features = ["BonusMalus", "Density"]

categorical_features = [
    "VehBrand", "VehGas", "Region", "Area",
    "DrivAge", "VehAge", "VehPower"
]

# Use the same observations for both models
X_pdp = X_train_t.sample(n=300, random_state=42).copy()
y_pdp = y_train_t.loc[X_pdp.index]

# Store the original categorical data types
category_dtypes = {
    col: X_train_t[col].dtype
    for col in categorical_features
}

# DALEX receives ordinary category labels
X_pdp_dalex = X_pdp.copy()

for col in categorical_features:
    X_pdp_dalex[col] = (
        X_pdp_dalex[col].astype(str).astype(object)
    )


# Restore original data types before predicting
def pdp_predict(model, data):
    data = data.copy()

    for col in categorical_features:
        dtype = category_dtypes[col]

        mapping = {
            str(value): value
            for value in dtype.categories
        }

        data[col] = (
            data[col]
            .map(mapping)
            .astype(dtype)
        )

    return model.predict(data)


# Verify that the conversion preserves predictions
original_pred = constrained_cv.best_estimator_.predict(X_pdp)
converted_pred = pdp_predict(
    constrained_cv.best_estimator_, X_pdp_dalex
)

np.testing.assert_allclose(original_pred, converted_pred)
print("Prediction compatibility verified!")

# %%
# Define DALEX explainers

explainer_constrained = dx.Explainer(
    constrained_cv.best_estimator_,
    data=X_pdp_dalex,
    y=y_pdp,
    predict_function=pdp_predict,
    label="Constrained LGBM",
    model_type="regression",
    verbose=False
)

explainer_unconstrained = dx.Explainer(
    unconstrained_cv.best_estimator_,
    data=X_pdp_dalex,
    y=y_pdp,
    predict_function=pdp_predict,
    label="Unconstrained LGBM",
    model_type="regression",
    verbose=False
)

print("Both DALEX explainers ready!")

# %%
# Step 3: Numerical PDP comparison

pdp_num_constrained = explainer_constrained.model_profile(
    variables=numerical_features,
    type="partial",
    variable_type="numerical",
    N=100,
    grid_points=30,
    random_state=42
)

pdp_num_unconstrained = explainer_unconstrained.model_profile(
    variables=numerical_features,
    type="partial",
    variable_type="numerical",
    N=100,
    grid_points=30,
    random_state=42
)

pdp_num_constrained.plot(
    objects=[pdp_num_unconstrained],
    variables=numerical_features,
    facet_ncol=2,
    title="PDP Comparison - Numerical Features"
)

# %%
# Step 4A: Vehicle and location PDPs

features_group1 = [
    "VehBrand", "VehGas", "Region", "Area"
]

pdp_cat1_constrained = explainer_constrained.model_profile(
    variables=features_group1,
    type="partial",
    variable_type="categorical",
    N=100,
    random_state=42
)

pdp_cat1_unconstrained = explainer_unconstrained.model_profile(
    variables=features_group1,
    type="partial",
    variable_type="categorical",
    N=100,
    random_state=42
)

pdp_cat1_constrained.plot(
    objects=[pdp_cat1_unconstrained],
    variables=features_group1,
    facet_ncol=2,
    title="PDP Comparison - Vehicle and Location"
)

# %%
# Step 4B: Age and vehicle power PDPs

features_group2 = [
    "DrivAge", "VehAge", "VehPower"
]

pdp_cat2_constrained = explainer_constrained.model_profile(
    variables=features_group2,
    type="partial",
    variable_type="categorical",
    N=100,
    random_state=42
)

pdp_cat2_unconstrained = explainer_unconstrained.model_profile(
    variables=features_group2,
    type="partial",
    variable_type="categorical",
    N=100,
    random_state=42
)

pdp_cat2_constrained.plot(
    objects=[pdp_cat2_unconstrained],
    variables=features_group2,
    facet_ncol=2,
    title="PDP Comparison - Age and Vehicle Power"
)
# %%
# Compare categorical PDP values between models

for feature in categorical_features:
    if feature in ["VehBrand", "VehGas", "Region", "Area"]:
        result_c = pdp_cat1_constrained.result
        result_u = pdp_cat1_unconstrained.result
    else:
        result_c = pdp_cat2_constrained.result
        result_u = pdp_cat2_unconstrained.result

    c = result_c.loc[
        result_c["_vname_"] == feature,
        ["_x_", "_yhat_"]
    ]
    u = result_u.loc[
        result_u["_vname_"] == feature,
        ["_x_", "_yhat_"]
    ]

    comparison = c.merge(u, on="_x_", suffixes=("_c", "_u"))
    max_diff = (
        comparison["_yhat__c"] - comparison["_yhat__u"]
    ).abs().max()

    print(f"{feature}: maximum PDP difference = {max_diff:.4f}")

# %% [markdown]
# ### Ex 4 — Partial Dependence Plots: Interpretation
#
# **Methodology**
#
# Partial Dependence Plots (PDPs) illustrate how average model
# predictions change with a particular feature, averaging over
# the remaining observed characteristics.
#
# We use DALEX explainers to compare the constrained and
# unconstrained LightGBM models across all nine features.
# Numerical and categorical variables are analysed separately,
# preserving the original categorical encoding used in training.
# Both models are evaluated on the same sampled observations
# to ensure comparability and reduce computational cost.
#
# **1. Numerical Features**
#
# - **BonusMalus:** Both models show a generally increasing
#   relationship between BonusMalus and predicted insurance risk.
#   However, the unconstrained model exhibits local decreases,
#   particularly around BonusMalus 60–65.
#   In contrast, the constrained model produces a non-decreasing
#   PDP, consistent with the monotonicity constraint imposed in Ex 1.
#   Both models exhibit a sharp increase around BonusMalus 95–105,
#   followed by a plateau at higher values.
#
# - **Density:** Predicted risk initially increases with Density
#   before becoming almost constant over much of the remaining
#   range. The two models produce nearly identical PDPs, suggesting
#   that constraining BonusMalus has little effect on the average
#   learned relationship with Density.
#
# **2. Categorical Features**
#
# - **VehBrand:** Predicted risk varies substantially across vehicle
#   brands. The constrained and unconstrained models show broadly
#   similar patterns, although some category-level differences remain.
#
# - **VehGas:** Diesel is associated with higher average predicted
#   risk than Regular in the displayed PDPs. The two models show
#   relatively similar effects.
#
# - **Region:** Considerable variation in predicted risk is observed
#   across regions, indicating that geographical characteristics
#   contribute to the models' risk predictions.
#
# - **Area:** Differences across area categories are relatively small,
#   with broadly similar relationships between the two models.
#
# - **DrivAge:** Predicted risk differs substantially across the
#   encoded driver-age groups, suggesting that driver characteristics
#   are important for distinguishing insurance risk.
#
# - **VehAge:** The differences across vehicle-age categories are
#   comparatively modest.
#
# - **VehPower:** Some vehicle-power categories are associated with
#   noticeably higher predicted risk than others, although the
#   overall patterns remain similar between the two models.
#
# **3. Comparison Between Constrained and Unconstrained Models**
#
# The main difference occurs in BonusMalus, where the constraint
# eliminates local decreases in predicted risk.
#
# The maximum absolute differences between the categorical PDPs
# are largest for VehBrand (9.22), Region (8.41), and DrivAge (7.38).
# Smaller maximum differences are observed for VehPower (4.56),
# Area (4.25), VehAge (3.49), and VehGas (3.29).
#
# These values represent the maximum difference in average
# predictions across category levels, rather than overall feature
# importance or average model differences.
#
# Although only BonusMalus is explicitly constrained, its restriction
# can indirectly affect the learned relationships with other features
# as LightGBM adjusts its tree structure and interactions.
#
# **4. Limitations**
#
# PDPs describe relationships learned by the fitted models rather
# than causal effects. They may conceal individual heterogeneity
# and feature interactions because predictions are averaged.
#
# Furthermore, varying one feature independently of correlated
# predictors may generate unrealistic combinations of characteristics.
# The PDPs are also estimated using a sample of observations,
# so their precise shapes may vary with the chosen sample.
#
# **Conclusion**
#
# Imposing an increasing monotonicity constraint on BonusMalus
# successfully prevents counterintuitive decreases in predicted
# insurance risk as BonusMalus increases, holding other features
# constant.
#
# The two models otherwise exhibit broadly similar marginal
# relationships across the remaining predictors, although some
# categorical effects differ.
#
# Combined with Ex 1 and Ex 3, the findings suggest that
# monotonicity constraints improve consistency with insurance-pricing
# logic at a relatively small cost to predictive accuracy and
# risk discrimination.


# %%
# ============================================================
# PS4 - Ex 5: Shapley Values
# ============================================================

# Step 1: Select the first test observation
observation = X_test_t.iloc[[0]].copy()

print("Selected policyholder:")
print(observation.T)

print("\nActual pure premium:", y_test_t.iloc[0])

# Compare the constrained LightGBM prediction
prediction_lgbm = (
    constrained_cv.best_estimator_.predict(observation)[0]
)

print("Constrained LGBM prediction:", prediction_lgbm)


# %%
# Step 2: Create constrained LightGBM Explainer

# Convert categorical features for DALEX
observation_dalex = observation.copy()

for col in categorical_features:
    observation_dalex[col] = (
        observation_dalex[col].astype(str).astype(object)
    )

# Reuse the reference data and prediction function from Ex 4
shap_explainer_lgbm = dx.Explainer(
    constrained_cv.best_estimator_,
    data=X_pdp_dalex,
    y=y_pdp,
    predict_function=pdp_predict,
    label="Constrained LGBM",
    model_type="regression",
    verbose=False
)

print("LightGBM Explainer ready!")


# %%
# Step 3: Calculate LightGBM SHAP values

shap_lgbm = shap_explainer_lgbm.predict_parts(
    observation_dalex,
    type="shap",
    N=100,
    B=25,
    random_state=42
)

print("LightGBM SHAP values calculated!")

# %%
# Step 4: Create GLM Explainer

shap_explainer_glm = dx.Explainer(
    t_glm1,
    data=X_pdp_dalex,
    y=y_pdp,
    predict_function=pdp_predict,
    label="GLM Benchmark",
    model_type="regression",
    verbose=False
)

print("GLM Explainer ready!")

# Verify GLM prediction compatibility
glm_pred_dalex = pdp_predict(t_glm1, observation_dalex)[0]

print("GLM prediction through DALEX adapter:", glm_pred_dalex)
print("Original GLM prediction:", t_glm1.predict(observation)[0])

# %%
# Step 5: Calculate GLM SHAP values

shap_glm = shap_explainer_glm.predict_parts(
    observation_dalex,
    type="shap",
    N=100,
    B=25,
    random_state=42
)

print("GLM SHAP values calculated!")

# %%
# Step 6: Compare SHAP decompositions

shap_lgbm.plot(
    objects=[shap_glm],
    title="SHAP Comparison - Constrained LightGBM vs GLM"
)

# %% [markdown]
# ### Ex 5 — SHAP Interpretation
#
# **Methodology**
#
# SHAP values decompose an individual prediction into a baseline
# prediction and additive feature contributions. Positive SHAP
# values increase predicted risk relative to the baseline, while
# negative values decrease it.
#
# We use DALEX to compare SHAP decompositions for the first test
# observation using the constrained LightGBM and original Tweedie
# GLM benchmark. Both models use the same reference observations
# and SHAP approximation settings.
#
# **Comparison of feature contributions**
#
# - **BonusMalus:** The largest negative contributor in both models.
#   BonusMalus = 50 reduces predicted risk by approximately 46.06
#   in LightGBM and 55.61 in GLM, reflecting favourable claim history.
#
# - **VehPower:** Both models assign positive contributions, with
#   a larger contribution in GLM (approximately +27 vs +18).
#
# - **DrivAge:** Positive in both models, with GLM assigning a
#   larger contribution (+19.22 vs +13.45).
#
# - **Area:** A particularly notable difference. Area E increases
#   predicted risk by approximately 22.13 in GLM, compared with
#   only 3.27 in LightGBM.
#
# - **Density:** LightGBM assigns a slightly larger positive
#   contribution (+9.45 vs +7.01).
#
# - **VehBrand and VehGas:** Both contribute negatively to the
#   prediction, with larger absolute contributions in GLM.
#
# **Interpretation**
#
# Both models identify similar directions of feature contributions,
# particularly the risk-reducing effect of a low BonusMalus score.
# However, the magnitudes differ considerably across features.
#
# This reflects their different modelling structures: GLM uses
# parametrically specified relationships, while LightGBM captures
# nonlinearities and feature interactions through boosted trees.
#
# Consequently, two models may produce similar overall predictions
# while attributing those predictions differently across features.
#
# **Limitations**
#
# SHAP explains predictions relative to a model-specific baseline,
# not causal effects. DALEX estimates SHAP values using sampled
# feature orderings, so contributions may vary with the number
# of simulations and the reference dataset.
#
# **Conclusion**
#
# SHAP provides local interpretability by identifying which features
# increase or decrease a particular policyholder's predicted risk.
# Compared with PDPs in Ex 4, which describe average relationships,
# SHAP reveals how the two models explain individual predictions.
