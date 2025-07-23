from typing import Union

import polars as pl
from sklearn.preprocessing import LabelEncoder

from src.features.base_encoder import BaseEncoder
from src.features.target_encoding import TargetEncoder, TargetEncoderConfig


class PaintColorEncoder(BaseEncoder):
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

    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> "PaintColorEncoder":
        if self.use_grouping:
            combined_df = pl.concat([X, y], how="horizontal")
            major_colors = (
                combined_df.group_by("paint_color")
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") >= 500)
                .select("paint_color")
                .to_series()
                .to_list()
            )
            self.paint_color_expr = (
                pl.when(pl.col("paint_color").is_in(major_colors))
                .then(pl.col("paint_color"))
                .otherwise(pl.lit("other_colors"))
                .alias("paint_color")
            )
            X = X.with_columns(self.paint_color_expr)

        if self.use_label_encoding:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(X.to_numpy().ravel())

        if self.use_target_encoding:
            self.target_encoder = TargetEncoder(
                smoothing=self.target_encoder_config.smoothing,
                min_samples_leaf=self.target_encoder_config.min_samples_leaf,
                noise_level=self.target_encoder_config.noise_level,
            )
            self.target_encoder.fit(X.select("paint_color").to_numpy(), y.to_numpy())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        result = X.clone()

        if self.use_grouping:
            result = result.with_columns(self.paint_color_expr)

        if self.use_label_encoding:
            result = result.with_columns(
                pl.Series(
                    name="paint_color_label",
                    values=self.label_encoder.transform(
                        result.select("paint_color").to_numpy().ravel()
                    ),
                )
            )

        # ターゲットエンコーディング
        if self.use_target_encoding:
            result = result.with_columns(
                pl.Series(
                    name="paint_color_te",
                    values=self.target_encoder.transform(
                        result.select("paint_color").to_numpy()
                    ),
                )
            )

        return result.drop("paint_color")
