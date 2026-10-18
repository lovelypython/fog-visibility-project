"""Run chronological fog forecasting comparisons and save tables/figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


def make_models(seed: int) -> dict[str, object]:
    # Scaling is fit within training data only. Target scaling is likewise fit on y_train.
    return {
        "SVR": TransformedTargetRegressor(
            regressor=Pipeline([("scale", StandardScaler()), ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf"))]),
            transformer=StandardScaler(),
        ),
        "Random Forest": Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=500, min_samples_leaf=2, max_features=0.8,
                random_state=seed, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scale", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="reg:squarederror", random_state=seed, n_jobs=-1,
            )),
        ]),
    }


def load_frame(config: dict, project_root: Path) -> tuple[pd.DataFrame, list[str]]:
    data_path = (project_root / config["observations_csv"]).resolve()
    frame = pd.read_csv(data_path)
    time_col, target = config.get("time_column", "timestamp"), config["target"]
    predictors = config["predictors"]
    required = [time_col, target, *predictors]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Observations CSV lacks columns: {sorted(missing)}")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise", utc=True)
    frame = frame.sort_values(time_col).drop_duplicates(time_col, keep="last").reset_index(drop=True)
    numeric = [target, *predictors]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")

    feature_columns: list[str] = []
    lag_steps = int(config.get("lag_steps", 5))
    if lag_steps < 1:
        raise ValueError("lag_steps must be at least 1")
    for column in numeric:
        for lag in range(1, lag_steps + 1):
            feature = f"{column}_lag{lag}"
            frame[feature] = frame[column].shift(lag)
            feature_columns.append(feature)

    features_file = config.get("cnn_features_csv")
    if features_file:
        features_path = (project_root / features_file).resolve()
        image_features = pd.read_csv(features_path)
        if time_col not in image_features:
            raise ValueError(f"CNN features CSV must include {time_col!r}")
        image_features[time_col] = pd.to_datetime(image_features[time_col], errors="raise", utc=True)
        cnn_columns = [c for c in image_features if c.startswith("cnn_")]
        if not cnn_columns:
            raise ValueError("CNN features CSV contains no cnn_* feature columns")
        frame = frame.merge(image_features[[time_col, *cnn_columns]], on=time_col, how="left", validate="one_to_one")
        # Forecasting at t uses only images available through t-1.
        for column in cnn_columns:
            for lag in range(1, lag_steps + 1):
                feature = f"{column}_lag{lag}"
                frame[feature] = frame[column].shift(lag)
                feature_columns.append(feature)

    frame = frame.dropna(subset=[*feature_columns, target]).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError(f"Only {len(frame)} usable rows remain; more timestamp-aligned data are needed")
    return frame, feature_columns


def run(config_path: Path) -> None:
    config_path = config_path.resolve()
    root = config_path.parent.parent if config_path.parent.name == "config" else Path.cwd()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame, features = load_frame(config, root)
    target = config["target"]
    time_col = config.get("time_column", "timestamp")
    fraction = float(config.get("train_fraction", 0.70))
    if not 0.5 <= fraction < 1:
        raise ValueError("train_fraction must be in [0.5, 1)")
    split = int(len(frame) * fraction)
    train, test = frame.iloc[:split], frame.iloc[split:]
    if test.empty:
        raise ValueError("Test partition is empty")
    X_train, X_test = train[features], test[features]
    y_train, y_test = train[target], test[target]

    out = (root / config.get("output_dir", "outputs")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({time_col: test[time_col].astype(str), "actual": y_test.to_numpy()})
    metrics: list[dict[str, object]] = []
    model_names = list(make_models(int(config.get("random_seed", 42))))
    fitted_models = make_models(int(config.get("random_seed", 42)))
    for name, model in fitted_models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        mse = mean_squared_error(y_test, predicted)
        metrics.append({"model": name, "MSE": mse, "RMSE": float(np.sqrt(mse)),
                        "MAE": mean_absolute_error(y_test, predicted),
                        "train_rows": len(train), "test_rows": len(test)})

    score_frame = pd.DataFrame(metrics).sort_values("MAE")
    score_frame.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(predictions[time_col]), predictions["actual"], label="Observed", linewidth=2)
    for name in model_names:
        ax.plot(pd.to_datetime(predictions[time_col]), predictions[name], label=name, alpha=0.85)
    ax.set(title="Fog evolution forecast: chronological test period", xlabel="Time", ylabel=target)
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(out / "forecast.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(score_frame["model"], score_frame["MAE"])
    axes[0].set(title="Mean absolute error", ylabel="MAE (target units)")
    axes[1].bar(score_frame["model"], score_frame["RMSE"])
    axes[1].set(title="Root mean squared error", ylabel="RMSE (target units)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "error_comparison.png", dpi=180); plt.close(fig)

    xgb_model = fitted_models["XGBoost"].named_steps["model"]
    importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values().tail(20)
    importances.to_csv(out / "xgboost_feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot.barh(ax=ax)
    ax.set(title="XGBoost: top 20 feature importances", xlabel="Gain-based importance")
    fig.tight_layout(); fig.savefig(out / "xgboost_feature_importance.png", dpi=180); plt.close(fig)

    frame[[time_col, target, *config["predictors"]]].describe().T.to_csv(out / "descriptive_statistics.csv")
    print(f"Train: {len(train)} rows; test: {len(test)} rows")
    print(score_frame.to_string(index=False))
    print(f"Saved predictions, metrics, descriptive statistics, and figures to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/example.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
