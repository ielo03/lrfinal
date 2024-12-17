import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataNormalizer:
    def __init__(self, columns_to_normalize=None, log_transform_columns=None):
        self.columns_to_normalize = list(columns_to_normalize) if columns_to_normalize is not None else []
        self.log_transform_columns = log_transform_columns or []
        self.scaler = StandardScaler()

    def fit(self, df):
        df_copy = df.copy()

        # Log transform specified columns
        for col in self.log_transform_columns:
            df_copy[col] = np.log1p(df_copy[col])  # log(1 + x) transformation

        # Fit scaler on all columns, including log-transformed ones
        self.scaler.fit(df_copy[self.columns_to_normalize])

    def transform(self, df):
        df_copy = df.copy()

        # Log transform specified columns
        for col in self.log_transform_columns:
            df_copy[col] = np.log1p(df_copy[col])  # log(1 + x) transformation

        # Apply Z-score normalization to all specified columns
        df_copy[self.columns_to_normalize] = self.scaler.transform(df_copy[self.columns_to_normalize])

        return df_copy

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def reverse_transform(self, df):
        df_copy = df.copy()

        # Reverse Z-score normalization
        df_copy[self.columns_to_normalize] = self.scaler.inverse_transform(df_copy[self.columns_to_normalize])

        # Reverse log transformation for specified columns
        for col in self.log_transform_columns:
            df_copy[col] = np.expm1(df_copy[col])  # Reverse of log1p -> exp(x) - 1

        return df_copy

    def reverse_predictions(self, predictions, column_name):
        # Ensure predictions are in the correct shape (n_samples, 1)
        predictions = np.array(predictions).reshape(-1, 1)

        # Create a dummy DataFrame with zeros for all columns the scaler expects
        dummy_df = pd.DataFrame(
            np.zeros((predictions.shape[0], len(self.columns_to_normalize))),
            columns=self.columns_to_normalize
        )

        # Replace the target column with the predictions
        dummy_df[column_name] = predictions.flatten()

        # Reverse Z-score normalization
        denormalized = self.scaler.inverse_transform(dummy_df)

        # Extract the denormalized target column
        denormalized_target = denormalized[:, self.columns_to_normalize.index(column_name)]

        # Reverse log transformation if applied
        if column_name in self.log_transform_columns:
            denormalized_target = np.expm1(denormalized_target)

        return denormalized_target