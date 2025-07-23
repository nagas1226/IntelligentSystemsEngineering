import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=1.0, min_samples_leaf=1, noise_level=0.01):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.global_mean = None
        self.category_encoding_map = {}

    def fit(
        self,
        X: np.ndarray | pl.Series,
        y: np.ndarray | pl.Series,
    ):
        """
        学習データでカテゴリ統計を計算し、エンコーディングマップを作成
        """
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        if isinstance(X, pl.Series):
            X: np.ndarray = X.to_numpy()
        if isinstance(y, pl.Series):
            y: np.ndarray = y.to_numpy()

        X = X.flatten()
        self.global_mean = np.mean(y)

        # カテゴリごとのエンコード値を事前計算
        unique_categories = np.unique(X)
        for category in unique_categories:
            mask = X == category
            category_values = y[mask]
            category_mean = np.mean(category_values)
            category_count = len(category_values)

            # 件数が少ない場合は全体平均
            if category_count < self.min_samples_leaf:
                encoded_value = self.global_mean
            else:
                # Smoothing formula
                encoded_value = (
                    category_count * category_mean + self.smoothing * self.global_mean
                ) / (category_count + self.smoothing)

            self.category_encoding_map[category] = encoded_value

        return self

    def transform(self, X: np.ndarray | pl.Series) -> np.ndarray:
        """
        カテゴリをtarget encodingで変換（未知カテゴリにも対応）
        """
        if isinstance(X, pl.Series) or isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        X = X.flatten()

        # ベクトル化された変換
        result = np.array(
            [
                self.category_encoding_map.get(category, self.global_mean)
                for category in X
            ],
            dtype=float,
        )

        # 軽微なノイズを追加してoverfittingを防ぐ
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=len(result))
            result += noise

        return result

    def fit_transform(self, X, y):
        """
        fit -> transform の組み合わせ
        """
        self.fit(X, y)
        return self.transform(X)
