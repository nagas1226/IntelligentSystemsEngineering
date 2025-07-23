"""
Base configuration classes for encoders and preprocessing.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.features.target_encoding import TargetEncoderConfig


@dataclass
class EncoderConfig:
    """Base configuration for all encoders."""

    use_target_encoding: bool = True
    target_encoder_config: Optional[TargetEncoderConfig] = None


@dataclass
class ConditionEncoderConfig(EncoderConfig):
    """Configuration for ConditionEncoder."""

    pass


@dataclass
class CylinderEncoderConfig(EncoderConfig):
    """Configuration for CylindersEncoder."""

    pass


@dataclass
class DriveEncoderConfig(EncoderConfig):
    """Configuration for DriveEncoder."""

    use_label_encoding: bool = True


@dataclass
class FuelEncoderConfig(EncoderConfig):
    """Configuration for FuelEncoder."""

    use_label_encoding: bool = True


@dataclass
class ManufacturerEncoderConfig(EncoderConfig):
    """Configuration for ManufacturerEncoder."""

    use_label_encoding: bool = True


@dataclass
class PaintColorEncoderConfig(EncoderConfig):
    """Configuration for PaintColorEncoder."""

    use_grouping: bool = True
    use_label_encoding: bool = True


@dataclass
class StateEncoderConfig(EncoderConfig):
    """Configuration for StateEncoder."""

    use_grouping: bool = True
    use_top_tier_flag: bool = True
    use_label_encoding: bool = True


@dataclass
class TransmissionEncoderConfig(EncoderConfig):
    """Configuration for TransmissionEncoder."""

    use_label_encoding: bool = True


@dataclass
class TypeEncoderConfig(EncoderConfig):
    """Configuration for TypeEncoder."""

    use_grouping: bool = True
    use_label_encoding: bool = True


@dataclass
class YearEncoderConfig:
    """Configuration for YearEncoder."""

    use_1987_flag: bool = True
    use_1975_flag: bool = True


@dataclass
class PreprocessorConfig:
    """Configuration for the Preprocessor."""

    condition_encoder_config: ConditionEncoderConfig
    cylinder_encoder_config: CylinderEncoderConfig
    drive_encoder_config: DriveEncoderConfig
    fuel_encoder_config: FuelEncoderConfig
    manufacturer_encoder_config: ManufacturerEncoderConfig
    paint_color_encoder_config: PaintColorEncoderConfig
    state_encoder_config: StateEncoderConfig
    transmission_encoder_config: TransmissionEncoderConfig
    type_encoder_config: TypeEncoderConfig
    year_encoder_config: YearEncoderConfig
    price_upper_bound: int = 40_000
    price_lower_bound: int = 1_000
    remove_outliers_val: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format for Preprocessor initialization."""
        return {
            "condition_encoder_config": self.condition_encoder_config.__dict__,
            "cylinder_encoder_config": self.cylinder_encoder_config.__dict__,
            "drive_encoder_config": self.drive_encoder_config.__dict__,
            "fuel_encoder_config": self.fuel_encoder_config.__dict__,
            "manufacturer_encoder_config": self.manufacturer_encoder_config.__dict__,
            "paint_color_encoder_config": self.paint_color_encoder_config.__dict__,
            "state_encoder_config": self.state_encoder_config.__dict__,
            "transmission_encoder_config": self.transmission_encoder_config.__dict__,
            "type_encoder_config": self.type_encoder_config.__dict__,
            "year_encoder_config": self.year_encoder_config.__dict__,
            "price_upper_bound": self.price_upper_bound,
            "price_lower_bound": self.price_lower_bound,
            "remove_outliers_val": self.remove_outliers_val,
        }
