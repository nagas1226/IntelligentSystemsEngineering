from typing import Union

import polars as pl
from sklearn.preprocessing import LabelEncoder

from src.features.base_encoder import BaseEncoder
from src.features.target_encoding import TargetEncoder, TargetEncoderConfig


class DriveEncoder(BaseEncoder):
    def __init__(
        self,
        use_label_encoding: bool = True,
        use_target_encoding: bool = True,
        target_encoder_config: Union[TargetEncoderConfig, None] = None,
    ):
        super().__init__(use_target_encoding, target_encoder_config)
        self.use_label_encoding = use_label_encoding

    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> "DriveEncoder":
        if self.use_label_encoding:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(X.to_numpy().ravel())

        if self.use_target_encoding:
            self.target_encoder = TargetEncoder(
                smoothing=self.target_encoder_config.smoothing,
                min_samples_leaf=self.target_encoder_config.min_samples_leaf,
                noise_level=self.target_encoder_config.noise_level,
            )
            self.target_encoder.fit(X.to_numpy(), y.select("price").to_numpy())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        result = X.clone()
        if self.use_label_encoding:
            result = result.with_columns(
                pl.Series(
                    name="drive_label",
                    values=self.label_encoder.transform(
                        result.select("drive").to_numpy().ravel()
                    ),
                )
            )

        # ターゲットエンコーディング
        if self.use_target_encoding:
            result = result.with_columns(
                pl.Series(
                    name="drive_te",
                    values=self.target_encoder.transform(
                        result.select("drive").to_numpy()
                    ),
                )
            )

        return result.drop("drive")
