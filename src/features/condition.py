import polars as pl

from src.features.base_encoder import BaseEncoder
from src.features.target_encoding import TargetEncoder, TargetEncoderConfig


class ConditionEncoder(BaseEncoder):
    def __init__(
        self,
        use_target_encoding: bool = True,
        target_encoder_config: None | TargetEncoderConfig = None,
    ):
        super().__init__(use_target_encoding, target_encoder_config)

        # 数値に変換
        self.numerical_conversion_expr = (
            pl.when(pl.col("condition") == "salvage")
            .then(0)
            .when(pl.col("condition") == "fair")
            .then(1)
            .when(pl.col("condition") == "good")
            .then(2)
            .when(pl.col("condition") == "excellent")
            .then(3)
            .when(pl.col("condition") == "like new")
            .then(4)
            .when(pl.col("condition") == "new")
            .then(5)
            .otherwise(None)
            .cast(pl.Float32)
            .alias("condition_numerical")
        )

    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> "ConditionEncoder":
        if self.use_target_encoding:
            self.target_encoder = TargetEncoder(
                smoothing=self.target_encoder_config.smoothing,
                min_samples_leaf=self.target_encoder_config.min_samples_leaf,
                noise_level=self.target_encoder_config.noise_level,
            )
            self.target_encoder.fit(X.select("condition").to_numpy(), y.to_numpy())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        result = X.with_columns(self.numerical_conversion_expr)

        if self.use_target_encoding:
            result = result.with_columns(
                pl.Series(
                    name="condition_te",
                    values=self.target_encoder.transform(
                        X.select("condition").to_numpy()
                    ),
                )
            )

        return result.drop("condition")

    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame) -> pl.DataFrame:
        self.fit(X, y)
        return self.transform(X)
