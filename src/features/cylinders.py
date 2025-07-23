from typing import Union

import polars as pl

from src.features.base_encoder import BaseEncoder
from src.features.target_encoding import TargetEncoder, TargetEncoderConfig


class CylindersEncoder(BaseEncoder):
    def __init__(
        self,
        use_target_encoding: bool = True,
        target_encoder_config: Union[TargetEncoderConfig, None] = None,
    ):
        super().__init__(use_target_encoding, target_encoder_config)

        # シリンダー数の抽出と変換
        self.cylinder_expr = (
            pl.when(pl.col("cylinders").str.contains(r"\d+"))
            .then(pl.col("cylinders").str.extract(r"(\d+)").cast(pl.Float32))
            .otherwise(None)
            .alias("cylinders_numerical")
        )

    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> "CylindersEncoder":
        if self.use_target_encoding:
            self.target_encoder = TargetEncoder(
                smoothing=self.target_encoder_config.smoothing,
                min_samples_leaf=self.target_encoder_config.min_samples_leaf,
                noise_level=self.target_encoder_config.noise_level,
            )
            self.target_encoder.fit(
                X.select("cylinders").to_numpy(), y.select("price").to_numpy()
            )
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        result = X.with_columns(self.cylinder_expr)

        # ターゲットエンコーディング
        if self.use_target_encoding:
            result = result.with_columns(
                pl.Series(
                    name="cylinders_te",
                    values=self.target_encoder.transform(
                        result.select("cylinders").to_numpy()
                    ),
                )
            )

        return result.drop("cylinders")
