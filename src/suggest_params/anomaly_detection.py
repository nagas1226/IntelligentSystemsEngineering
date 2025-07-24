from typing import Any, Dict

import optuna


def suggest_lgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Suggest parameters for LightGBM model using Optuna trial.

    Args:
        trial: Optuna trial object

    Returns:
        Dict[str, Any]: Suggested parameters for LightGBM
    """
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "max_depth": trial.suggest_int("max_depth", -1, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    }
