import polars as pl

from src.features.base_encoder import BaseEncoder
from src.features.condition import ConditionEncoder
from src.features.cylinders import CylindersEncoder
from src.features.drive import DriveEncoder
from src.features.fuel import FuelEncoder
from src.features.manufacturer import ManufacturerEncoder
from src.features.paint_color import PaintColorEncoder
from src.features.state import StateEncoder
from src.features.transmission import TransmissionEncoder
from src.features.type import TypeEncoder
from src.features.year import YearEncoder


class Preprocessor:
    def __init__(
        self,
        condition_encoder_config: dict,
        cylinder_encoder_config: dict,
        drive_encoder_config: dict,
        fuel_encoder_config: dict,
        manufacturer_encoder_config: dict,
        paint_color_encoder_config: dict,
        state_encoder_config: dict,
        transmission_encoder_config: dict,
        type_encoder_config: dict,
        year_encoder_config: dict,
        price_upper_bound: int = 40_000,
        price_lower_bound: int = 1_000,
        remove_outliers_val: bool = True,
    ):
        self.encoders: dict[str, BaseEncoder] = {
            "condition": ConditionEncoder(**(condition_encoder_config)),
            "cylinders": CylindersEncoder(**(cylinder_encoder_config)),
            "drive": DriveEncoder(**(drive_encoder_config)),
            "fuel": FuelEncoder(**(fuel_encoder_config)),
            "manufacturer": ManufacturerEncoder(**(manufacturer_encoder_config)),
            "paint_color": PaintColorEncoder(**(paint_color_encoder_config)),
            "state": StateEncoder(**(state_encoder_config)),
            "transmission": TransmissionEncoder(**(transmission_encoder_config)),
            "type": TypeEncoder(**(type_encoder_config)),
            "year": YearEncoder(**(year_encoder_config)),
        }

        self.price_upper_bound = price_upper_bound
        self.price_lower_bound = price_lower_bound

        self._remove_outliers_val = remove_outliers_val

    def run(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        # Exclude outliers from the training data
        train_df_excluded = self._remove_outliers(
            train_df, self.price_upper_bound, self.price_lower_bound
        )

        if self._remove_outliers_val:
            # Exclude outliers from the validation and test data
            val_df = self._remove_outliers(
                val_df, self.price_upper_bound, self.price_lower_bound
            )

        # feature engineering
        train_df_preprocessed, val_df_preprocessed, test_df_preprocessed = (
            self._feature_engineering(train_df_excluded, val_df, test_df)
        )

        return train_df_preprocessed, val_df_preprocessed, test_df_preprocessed

    def _fit_encoders(self, train_df: pl.DataFrame) -> None:
        # Fit all encoders on the training data
        for col, encoder in self.encoders.items():
            encoder.fit(train_df.select(col), train_df.select("price"))

    def _transform(self, df: pl.DataFrame) -> pl.DataFrame:
        # Transform the dataframe using the fitted encoders
        encoded_dfs = []
        for col, encoder in self.encoders.items():
            encoded_df = encoder.transform(df.select(col))
            encoded_dfs.append(encoded_df)

        encoded_df = pl.concat(encoded_dfs, how="horizontal")

        transformed_df = pl.concat(
            [df.select(["price", "odometer"]), encoded_df], how="horizontal"
        )
        return transformed_df

    def _feature_engineering(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        # fit encoders
        self._fit_encoders(train_df)

        # transform dataframes
        train_df_transformed = self._transform(train_df)
        val_df_transformed = self._transform(val_df)
        test_df_transformed = self._transform(test_df)

        return train_df_transformed, val_df_transformed, test_df_transformed

    def _remove_outliers(
        self,
        df: pl.DataFrame,
        price_upper_bound: int = 40_000,
        price_lower_bound: int = 1_000,
    ) -> pl.DataFrame:
        return df.filter(
            (pl.col("price") < price_upper_bound)
            & (pl.col("price") > price_lower_bound)
        )
