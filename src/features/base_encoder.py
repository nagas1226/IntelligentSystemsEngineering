"""
Base encoder class for feature encoders.

All feature encoders should inherit from this base class to maintain
a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Optional

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.preprocess import TargetEncoderConfig


class BaseEncoder(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for feature encoders.

    This class defines the common interface that all feature encoders
    should implement. It provides common functionality for target encoding
    configuration and enforces the sklearn transformer interface.

    Args:
        use_target_encoding: Whether to use target encoding
        target_encoder_config: Configuration for target encoding
    """

    def __init__(
        self,
        use_target_encoding: bool = True,
        target_encoder_config: Optional[TargetEncoderConfig] = None,
    ):
        self.use_target_encoding = use_target_encoding
        self.target_encoder_config = target_encoder_config

        if use_target_encoding and target_encoder_config is None:
            raise ValueError(
                "When use_target_encoding is True, target_encoder_config must be provided."
            )

    @abstractmethod
    def fit(self, X: pl.DataFrame, y: pl.DataFrame) -> "BaseEncoder":
        """
        Fit the encoder to the training data.

        Args:
            X: Input features (DataFrame)
            y: Target variable (DataFrame)

        Returns:
            self: Fitted encoder instance
        """
        pass

    @abstractmethod
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the input data using the fitted encoder.

        Args:
            X: Input features to transform (DataFrame)

        Returns:
            pl.DataFrame: Transformed features
        """
        pass

    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame) -> pl.DataFrame:
        """
        Fit the encoder and transform the data in one step.

        Args:
            X: Input features (DataFrame or DataFrame)
            y: Target variable (DataFrame)

        Returns:
            pl.DataFrame: Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
