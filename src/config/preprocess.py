"""
Pydantic-based configuration classes for the preprocessing pipeline.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TargetEncoderConfig(BaseModel):
    """Configuration for target encoding."""

    smoothing: float = Field(
        default=1.0,
        ge=0.0,
        description="Smoothing parameter for target encoding. Higher values lead to more regularization.",
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        description="Minimum number of samples required to compute target encoding for a category.",
    )
    noise_level: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Level of noise to add to target encoding to prevent overfitting.",
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"


# エンコーダー別設定クラス
class ConditionEncoderConfig(BaseModel):
    """Configuration for condition encoder."""

    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class CylinderEncoderConfig(BaseModel):
    """Configuration for cylinder encoder."""

    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class DriveEncoderConfig(BaseModel):
    """Configuration for drive encoder."""

    use_label_encoding: bool = Field(
        default=True, description="Whether to use label encoding."
    )
    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class FuelEncoderConfig(BaseModel):
    """Configuration for fuel encoder."""

    use_label_encoding: bool = Field(
        default=True, description="Whether to use label encoding."
    )
    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class ManufacturerEncoderConfig(BaseModel):
    """Configuration for manufacturer encoder."""

    use_grouping: bool = Field(
        default=True, description="Whether to group rare categories."
    )
    use_label_encoding: bool = Field(
        default=True, description="Whether to use label encoding."
    )
    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class PaintColorEncoderConfig(BaseModel):
    """Configuration for paint color encoder."""

    use_grouping: bool = Field(
        default=True, description="Whether to group rare categories."
    )
    use_label_encoding: bool = Field(
        default=True, description="Whether to use label encoding."
    )
    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class StateEncoderConfig(BaseModel):
    """Configuration for state encoder."""

    use_grouping: bool = Field(
        default=True, description="Whether to group rare categories."
    )
    use_top_tier_flag: bool = Field(
        default=True, description="Whether to use top tier flag."
    )
    use_label_encoding: bool = Field(
        default=True, description="Whether to use label encoding."
    )
    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class TransmissionEncoderConfig(BaseModel):
    """Configuration for transmission encoder."""

    use_label_encoding: bool = Field(
        default=True, description="Whether to use label encoding."
    )
    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class TypeEncoderConfig(BaseModel):
    """Configuration for type encoder."""

    use_grouping: bool = Field(
        default=True, description="Whether to group rare categories."
    )
    use_label_encoding: bool = Field(
        default=True, description="Whether to use label encoding."
    )
    use_target_encoding: bool = Field(
        default=True, description="Whether to use target encoding."
    )
    target_encoder_config: Optional[TargetEncoderConfig] = Field(
        default_factory=TargetEncoderConfig
    )

    class Config:
        validate_assignment = True
        extra = "forbid"


class YearEncoderConfig(BaseModel):
    """Configuration for year encoder."""

    use_1987_flag: bool = Field(
        default=True, description="Whether to create a flag for year 1987."
    )
    use_1975_flag: bool = Field(
        default=True, description="Whether to create a flag for year 1975."
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"


class PreprocessorConfig(BaseModel):
    """Main configuration for the preprocessing pipeline."""

    # Categorical encoder configurations
    condition_encoder_config: ConditionEncoderConfig = Field(
        default_factory=ConditionEncoderConfig
    )
    cylinder_encoder_config: CylinderEncoderConfig = Field(
        default_factory=CylinderEncoderConfig
    )
    drive_encoder_config: DriveEncoderConfig = Field(default_factory=DriveEncoderConfig)
    fuel_encoder_config: FuelEncoderConfig = Field(default_factory=FuelEncoderConfig)
    manufacturer_encoder_config: ManufacturerEncoderConfig = Field(
        default_factory=ManufacturerEncoderConfig
    )
    paint_color_encoder_config: PaintColorEncoderConfig = Field(
        default_factory=PaintColorEncoderConfig
    )
    state_encoder_config: StateEncoderConfig = Field(default_factory=StateEncoderConfig)
    transmission_encoder_config: TransmissionEncoderConfig = Field(
        default_factory=TransmissionEncoderConfig
    )
    type_encoder_config: TypeEncoderConfig = Field(default_factory=TypeEncoderConfig)
    year_encoder_config: YearEncoderConfig = Field(default_factory=YearEncoderConfig)

    # Price filtering parameters
    price_upper_bound: float = Field(
        default=40000.0, gt=0.0, description="Upper bound for price filtering."
    )
    price_lower_bound: float = Field(
        default=1000.0, ge=0.0, description="Lower bound for price filtering."
    )
    remove_outliers_val: bool = Field(
        default=False, description="Whether to remove outliers from validation set."
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"

    def to_dict(self) -> dict:
        """Convert config to dictionary format for backward compatibility.

        Unlike the standard dict() method, this preserves TargetEncoderConfig objects
        as objects rather than converting them to dictionaries.
        """
        result = {}

        # Convert each field manually
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "target_encoder_config"):
                # For encoder configs, create a dict but preserve target_encoder_config as object
                encoder_dict = {}
                for key, value in field_value.__dict__.items():
                    if key == "target_encoder_config":
                        # Keep the TargetEncoderConfig object as-is
                        encoder_dict[key] = value
                    else:
                        encoder_dict[key] = value
                result[field_name] = encoder_dict
            else:
                # For non-encoder configs, use regular conversion
                if hasattr(field_value, "dict"):
                    result[field_name] = field_value.dict()
                else:
                    result[field_name] = field_value

        return result

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PreprocessorConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
