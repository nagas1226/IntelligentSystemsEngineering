from typing import Union

import polars as pl
from sklearn.preprocessing import LabelEncoder

from src.features.base_encoder import BaseEncoder
from src.features.target_encoding import TargetEncoder, TargetEncoderConfig


class StateEncoder(BaseEncoder):
    def __init__(
        self,
        use_grouping: bool = True,
        use_top_tier_flag: bool = True,
        use_label_encoding: bool = True,
        use_target_encoding: bool = True,
        target_encoder_config: Union[TargetEncoderConfig, None] = None,
    ):
        super().__init__(use_target_encoding, target_encoder_config)
        self.use_grouping = use_grouping
        self.use_top_tier_flag = use_top_tier_flag
        self.use_label_encoding = use_label_encoding

    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> "StateEncoder":
        combined_df = pl.concat([X, y], how="horizontal")
        if self.use_grouping:
            major_states = (
                combined_df.group_by("state")
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") >= 300)
                .select("state")
                .to_series()
                .to_list()
            )
            self.state_expr = (
                pl.when(pl.col("state").is_in(major_states))
                .then(pl.col("state"))
                .otherwise(pl.lit("other_states"))
                .alias("state")
            )
            X = X.with_columns(self.state_expr)

        if self.use_top_tier_flag:
            top_states = (
                combined_df.group_by("state")
                .agg(pl.median("price").alias("median_price"))
                .sort("median_price", descending=True)
                .head(10)
                .select("state")
                .to_series()
                .to_list()
            )
            self.top_states_expr = (
                pl.when(pl.col("state").is_in(top_states))
                .then(1)
                .otherwise(0)
                .alias("is_top_10_state")
            )
            X = X.with_columns(self.top_states_expr)

        if self.use_label_encoding:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(X.select("state").to_numpy().ravel())

        if self.use_target_encoding:
            self.target_encoder = TargetEncoder(
                smoothing=self.target_encoder_config.smoothing,
                min_samples_leaf=self.target_encoder_config.min_samples_leaf,
                noise_level=self.target_encoder_config.noise_level,
            )
            self.target_encoder.fit(X.select("state").to_numpy(), y.to_numpy())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        result = X.clone()
        if self.use_top_tier_flag:
            result = result.with_columns(self.top_states_expr)

        if self.use_grouping:
            result = result.with_columns(self.state_expr)

        if self.use_label_encoding:
            result = result.with_columns(
                pl.Series(
                    name="state_label",
                    values=self.label_encoder.transform(
                        result.select("state").to_numpy().ravel()
                    ),
                )
            )

        # ターゲットエンコーディング
        if self.use_target_encoding:
            result = result.with_columns(
                pl.Series(
                    name="state_te",
                    values=self.target_encoder.transform(
                        result.select("state").to_numpy()
                    ),
                )
            )

        return result.drop("state")
