"""
Default configurations for all encoders and preprocessing.
"""

from src.config.base_config import (
    ConditionEncoderConfig,
    CylinderEncoderConfig,
    DriveEncoderConfig,
    FuelEncoderConfig,
    ManufacturerEncoderConfig,
    PaintColorEncoderConfig,
    PreprocessorConfig,
    StateEncoderConfig,
    TransmissionEncoderConfig,
    TypeEncoderConfig,
    YearEncoderConfig,
)
from src.features.target_encoding import TargetEncoderConfig


def get_default_config() -> PreprocessorConfig:
    """Get the default configuration for all encoders."""

    # Default target encoder config
    default_target_config = TargetEncoderConfig(
        smoothing=1.0,
        min_samples_leaf=1,
        noise_level=0.1,
    )

    return PreprocessorConfig(
        condition_encoder_config=ConditionEncoderConfig(
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        cylinder_encoder_config=CylinderEncoderConfig(
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        drive_encoder_config=DriveEncoderConfig(
            use_label_encoding=True,
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        fuel_encoder_config=FuelEncoderConfig(
            use_label_encoding=True,
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        manufacturer_encoder_config=ManufacturerEncoderConfig(
            use_label_encoding=True,
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        paint_color_encoder_config=PaintColorEncoderConfig(
            use_grouping=True,
            use_label_encoding=True,
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        state_encoder_config=StateEncoderConfig(
            use_grouping=True,
            use_top_tier_flag=True,
            use_label_encoding=True,
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        transmission_encoder_config=TransmissionEncoderConfig(
            use_label_encoding=True,
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        type_encoder_config=TypeEncoderConfig(
            use_grouping=True,
            use_label_encoding=True,
            use_target_encoding=True,
            target_encoder_config=default_target_config,
        ),
        year_encoder_config=YearEncoderConfig(
            use_1987_flag=True,
            use_1975_flag=True,
        ),
        price_upper_bound=40_000,
        price_lower_bound=1_000,
        remove_outliers_val=True,
    )


def get_search_space() -> dict:
    """
    Define the hyperparameter search space for all encoders.
    Returns a dictionary with parameter names and their possible values.
    """
    return {
        # Target encoding parameters
        "target_smoothing": [0.5, 1.0, 2.0, 5.0, 10.0],
        "target_min_samples_leaf": [1, 5, 10, 20, 50],
        "target_noise_level": [0.0, 0.01, 0.05, 0.1, 0.2],
        # Price bounds
        "price_upper_bound": [35_000, 40_000, 45_000, 50_000],
        "price_lower_bound": [500, 1_000, 1_500, 2_000],
        # Boolean flags for each encoder
        "condition_use_target_encoding": [True, False],
        "cylinder_use_target_encoding": [True, False],
        "drive_use_label_encoding": [True, False],
        "drive_use_target_encoding": [True, False],
        "fuel_use_label_encoding": [True, False],
        "fuel_use_target_encoding": [True, False],
        "manufacturer_use_label_encoding": [True, False],
        "manufacturer_use_target_encoding": [True, False],
        "paint_color_use_grouping": [True, False],
        "paint_color_use_label_encoding": [True, False],
        "paint_color_use_target_encoding": [True, False],
        "state_use_grouping": [True, False],
        "state_use_top_tier_flag": [True, False],
        "state_use_label_encoding": [True, False],
        "state_use_target_encoding": [True, False],
        "transmission_use_label_encoding": [True, False],
        "transmission_use_target_encoding": [True, False],
        "type_use_grouping": [True, False],
        "type_use_label_encoding": [True, False],
        "type_use_target_encoding": [True, False],
        "year_use_1987_flag": [True, False],
        "year_use_1975_flag": [True, False],
    }


def create_config_from_params(params: dict) -> PreprocessorConfig:
    """
    Create a PreprocessorConfig from hyperparameter search parameters.

    Args:
        params: Dictionary of hyperparameters from the search

    Returns:
        PreprocessorConfig: Configuration object for the preprocessor
    """
    # Create target encoder config from params
    target_config = TargetEncoderConfig(
        smoothing=params.get("target_smoothing", 1.0),
        min_samples_leaf=params.get("target_min_samples_leaf", 1),
        noise_level=params.get("target_noise_level", 0.1),
    )

    return PreprocessorConfig(
        condition_encoder_config=ConditionEncoderConfig(
            use_target_encoding=params.get("condition_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("condition_use_target_encoding", True)
            else None,
        ),
        cylinder_encoder_config=CylinderEncoderConfig(
            use_target_encoding=params.get("cylinder_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("cylinder_use_target_encoding", True)
            else None,
        ),
        drive_encoder_config=DriveEncoderConfig(
            use_label_encoding=params.get("drive_use_label_encoding", True),
            use_target_encoding=params.get("drive_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("drive_use_target_encoding", True)
            else None,
        ),
        fuel_encoder_config=FuelEncoderConfig(
            use_label_encoding=params.get("fuel_use_label_encoding", True),
            use_target_encoding=params.get("fuel_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("fuel_use_target_encoding", True)
            else None,
        ),
        manufacturer_encoder_config=ManufacturerEncoderConfig(
            use_label_encoding=params.get("manufacturer_use_label_encoding", True),
            use_target_encoding=params.get("manufacturer_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("manufacturer_use_target_encoding", True)
            else None,
        ),
        paint_color_encoder_config=PaintColorEncoderConfig(
            use_grouping=params.get("paint_color_use_grouping", True),
            use_label_encoding=params.get("paint_color_use_label_encoding", True),
            use_target_encoding=params.get("paint_color_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("paint_color_use_target_encoding", True)
            else None,
        ),
        state_encoder_config=StateEncoderConfig(
            use_grouping=params.get("state_use_grouping", True),
            use_top_tier_flag=params.get("state_use_top_tier_flag", True),
            use_label_encoding=params.get("state_use_label_encoding", True),
            use_target_encoding=params.get("state_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("state_use_target_encoding", True)
            else None,
        ),
        transmission_encoder_config=TransmissionEncoderConfig(
            use_label_encoding=params.get("transmission_use_label_encoding", True),
            use_target_encoding=params.get("transmission_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("transmission_use_target_encoding", True)
            else None,
        ),
        type_encoder_config=TypeEncoderConfig(
            use_grouping=params.get("type_use_grouping", True),
            use_label_encoding=params.get("type_use_label_encoding", True),
            use_target_encoding=params.get("type_use_target_encoding", True),
            target_encoder_config=target_config
            if params.get("type_use_target_encoding", True)
            else None,
        ),
        year_encoder_config=YearEncoderConfig(
            use_1987_flag=params.get("year_use_1987_flag", True),
            use_1975_flag=params.get("year_use_1975_flag", True),
        ),
        price_upper_bound=params.get("price_upper_bound", 40_000),
        price_lower_bound=params.get("price_lower_bound", 1_000),
        remove_outliers_val=True,
    )
