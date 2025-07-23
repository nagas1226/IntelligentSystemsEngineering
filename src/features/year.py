import polars as pl

from src.features.base_encoder import BaseEncoder


class YearEncoder(BaseEncoder):
    def __init__(self, use_1987_flag: bool = True, use_1975_flag: bool = True):
        # YearEncoder doesn't use target encoding, so pass False
        super().__init__(use_target_encoding=False, target_encoder_config=None)
        self.use_1987_flag = use_1987_flag
        self.use_1975_flag = use_1975_flag

        self.flag_1987_expr = (
            pl.when(pl.col("year") >= 1987)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_1987_or_later")
        )

        self.flag_1975_expr = (
            pl.when(pl.col("year") >= 1975)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_1975_or_later")
        )

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "YearEncoder":
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        result = X.clone()

        if self.use_1987_flag:
            result = result.with_columns(self.flag_1987_expr)
        if self.use_1975_flag:
            result = result.with_columns(self.flag_1975_expr)
        return result
