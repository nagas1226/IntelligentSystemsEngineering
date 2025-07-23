from typing import Literal

import optuna

from src.config.preprocess import (
    ConditionEncoderConfig,
    CylinderEncoderConfig,
    DriveEncoderConfig,
    FuelEncoderConfig,
    ManufacturerEncoderConfig,
    PaintColorEncoderConfig,
    PreprocessorConfig,
    StateEncoderConfig,
    TargetEncoderConfig,
    TransmissionEncoderConfig,
    TypeEncoderConfig,
)


def suggest_preprocessor_config(
    trial: optuna.Trial, task: Literal["regression", "classification"]
) -> PreprocessorConfig:
    """OptunaトライアルからPreprocessorConfigを生成"""
    if task not in ["regression", "classification"]:
        raise ValueError("task must be either 'regression' or 'classification'")

    if task == "regression":
        remove_outliers_val = False
        price_upper_bound = trial.suggest_int(
            "price_upper_bound", 10_000, 60_000, step=500
        )
        price_lower_bound = trial.suggest_int("price_lower_bound", 0, 5_000, step=100)

    else:
        remove_outliers_val = False
        price_upper_bound = float("inf")
        price_lower_bound = 0.0

    target_encoder_config = suggest_target_encoding_params(trial)

    return PreprocessorConfig(
        condition_encoder_config=create_condition_config(target_encoder_config),
        cylinder_encoder_config=create_cylinder_config(target_encoder_config),
        drive_encoder_config=create_drive_config(target_encoder_config),
        fuel_encoder_config=create_fuel_config(target_encoder_config),
        manufacturer_encoder_config=create_manufacturer_config(target_encoder_config),
        paint_color_encoder_config=create_paint_color_config(target_encoder_config),
        state_encoder_config=create_state_config(target_encoder_config),
        transmission_encoder_config=create_transmission_config(target_encoder_config),
        type_encoder_config=create_type_config(target_encoder_config),
        price_lower_bound=price_lower_bound,
        price_upper_bound=price_upper_bound,
        remove_outliers_val=remove_outliers_val,
    )


def create_condition_config(target_encoder_config: TargetEncoderConfig):
    """ConditionEncoderConfigをOptunaトライアルから生成"""
    return ConditionEncoderConfig(target_encoder_config=target_encoder_config)


def create_cylinder_config(target_encoder_config: TargetEncoderConfig):
    """CylinderEncoderConfigをOptunaトライアルから生成"""
    return CylinderEncoderConfig(target_encoder_config=target_encoder_config)


def create_drive_config(target_encoder_config: TargetEncoderConfig):
    """DriveEncoderConfigをOptunaトライアルから生成"""
    return DriveEncoderConfig(target_encoder_config=target_encoder_config)


def create_fuel_config(target_encoder_config: TargetEncoderConfig):
    """FuelEncoderConfigをOptunaトライアルから生成"""
    return FuelEncoderConfig(target_encoder_config=target_encoder_config)


def create_manufacturer_config(target_encoder_config: TargetEncoderConfig):
    """ManufacturerEncoderConfigをOptunaトライアルから生成"""
    return ManufacturerEncoderConfig(target_encoder_config=target_encoder_config)


def create_paint_color_config(target_encoder_config: TargetEncoderConfig):
    """PaintColorEncoderConfigをOptunaトライアルから生成"""
    return PaintColorEncoderConfig(target_encoder_config=target_encoder_config)


def create_state_config(target_encoder_config: TargetEncoderConfig):
    """StateEncoderConfigをOptunaトライアルから生成"""
    return StateEncoderConfig(target_encoder_config=target_encoder_config)


def create_transmission_config(target_encoder_config: TargetEncoderConfig):
    """TransmissionEncoderConfigをOptunaトライアルから生成"""
    return TransmissionEncoderConfig(target_encoder_config=target_encoder_config)


def create_type_config(target_encoder_config: TargetEncoderConfig):
    """TypeEncoderConfigをOptunaトライアルから生成"""
    return TypeEncoderConfig(target_encoder_config=target_encoder_config)


def suggest_target_encoding_params(trial: optuna.Trial) -> TargetEncoderConfig:
    """Optunaトライアルからターゲットエンコーディングのパラメータを提案"""
    return TargetEncoderConfig(
        smoothing=trial.suggest_float("smoothing", 0.01, 1.0, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        noise_level=trial.suggest_float("noise_level", 0.01, 1.0, log=True),
    )
