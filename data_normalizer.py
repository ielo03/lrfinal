import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataNormalizer:
    def __init__(self, columns_to_normalize, log_transform_columns=None):
        """
        Initialize the DataNormalizer.

        Args:
            columns_to_normalize (list): Columns to normalize.
            log_transform_columns (list, optional): Columns to log-transform before normalization.
        """
        self.columns_to_normalize = columns_to_normalize
        self.log_transform_columns = log_transform_columns or []
        self.scaler = StandardScaler()

    def fit(self, df):
        # Apply log transformation if specified
        df = df.copy()
        for col in self.log_transform_columns:
            df[col] = df[col].apply(lambda x: np.log1p(x))  # log(1 + x) to handle zeros
        self.scaler.fit(df[self.columns_to_normalize])

    def transform(self, df):
        # Apply log transformation if specified
        df = df.copy()
        for col in self.log_transform_columns:
            df[col] = df[col].apply(lambda x: np.log1p(x))
        normalized_data = pd.DataFrame(
            self.scaler.transform(df[self.columns_to_normalize]),
            columns=self.columns_to_normalize,
            index=df.index
        )
        return pd.concat([normalized_data, df.drop(columns=self.columns_to_normalize)], axis=1)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        original_data = pd.DataFrame(
            self.scaler.inverse_transform(df[self.columns_to_normalize]),
            columns=self.columns_to_normalize,
            index=df.index
        )
        # Reverse log transformation if specified
        for col in self.log_transform_columns:
            original_data[col] = original_data[col].apply(lambda x: np.expm1(x))  # Reverse log(1 + x)
        return pd.concat([original_data, df.drop(columns=self.columns_to_normalize)], axis=1)