from typing import Union

import polars as pl
from sklearn.preprocessing import LabelEncoder

from src.config.preprocess import TargetEncoderConfig
from src.features.base_encoder import BaseEncoder
from src.features.target_encoding import TargetEncoder


class ManufacturerEncoder(BaseEncoder):
    def __init__(
        self,
        use_grouping: bool = True,
        use_label_encoding: bool = True,
        use_target_encoding: bool = True,
        target_encoder_config: Union[TargetEncoderConfig, None] = None,
    ):
        super().__init__(use_target_encoding, target_encoder_config)
        self.use_grouping = use_grouping
        self.use_label_encoding = use_label_encoding

        # プレミアムメーカーのフラグ
        self.premium_expr = (
            pl.when(pl.col("manufacturer").is_in(["ferrari", "tesla", "ram"]))
            .then(1)
            .otherwise(0)
            .alias("is_premium_manufacturer")
        )

        # 潜在的に高価格なメーカーのフラグ
        self.potential_expr = (
            pl.when(
                pl.col("manufacturer").is_in(["porsche", "jaguar", "ford", "chevrolet"])
            )
            .then(1)
            .otherwise(0)
            .alias("is_potentially_overpriced_manufacturer")
        )

    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> "ManufacturerEncoder":
        # カテゴリカル変数のエンコーディング
        if self.use_grouping:
            combined_df = pl.concat([X, y], how="horizontal")
            major_manufacturers = (
                combined_df.group_by("manufacturer")
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") >= 200)
                .select("manufacturer")
                .to_series()
                .to_list()
            )
            self.manufacturer_expr = (
                pl.when(pl.col("manufacturer").is_in(major_manufacturers))
                .then(pl.col("manufacturer"))
                .otherwise(pl.lit("other_manufacturers"))
                .alias("manufacturer")
            )
            X = X.with_columns(self.manufacturer_expr)

        if self.use_label_encoding:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(X.select("manufacturer").to_numpy().ravel())

        if self.use_target_encoding:
            self.target_encoder = TargetEncoder(
                smoothing=self.target_encoder_config.smoothing,
                min_samples_leaf=self.target_encoder_config.min_samples_leaf,
                noise_level=self.target_encoder_config.noise_level,
            )
            self.target_encoder.fit(X.select("manufacturer").to_numpy(), y.to_numpy())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        result = X.with_columns(self.premium_expr, self.potential_expr)

        if self.use_grouping:
            result = result.with_columns(self.manufacturer_expr)

        # ラベルエンコーディング
        if self.use_label_encoding:
            result = result.with_columns(
                pl.Series(
                    name="manufacturer_label",
                    values=self.label_encoder.transform(
                        result.select("manufacturer").to_numpy().ravel()
                    ),
                )
            )

        # ターゲットエンコーディング
        if self.use_target_encoding:
            result = result.with_columns(
                pl.Series(
                    name="manufacturer_te",
                    values=self.target_encoder.transform(
                        result.select("manufacturer").to_numpy()
                    ),
                )
            )

        return result.drop("manufacturer")
