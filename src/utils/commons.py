import pandas as pd
import numpy as np

def reduce_mem_usage(df):
    """
    Iterates through all columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Initial memory usage: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # Use float32 instead of float16 to avoid compatibility issues with LightGBM
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Final memory usage: {end_mem:.2f} MB')
    print(f'Decreased by: {(start_mem - end_mem) / start_mem * 100:.1f}%')
    return df

def extract_data(transaction_path, identity_path):
    """ Loads transaction and identity datasets from CSV files. """
    print(f"Loading data from: {transaction_path} and {identity_path}...")
    df_transaction = pd.read_csv(transaction_path)
    df_identity = pd.read_csv(identity_path)
    print(f"Data successfully loaded! Shapes: {df_transaction.shape}, {df_identity.shape}")
    return df_transaction, df_identity

def get_redundant_features(df, prefix='V', threshold=0.90):
    """ Identifies highly correlated features based on a specific prefix. """
    cols = [c for c in df.columns if c.startswith(prefix)]
    corr_matrix = df[cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Found {len(to_drop)} redundant '{prefix}' columns with correlation > {threshold}.")
    return to_drop

def add_group_stats(df, group_cols, target_col='TransactionAmt', stats=['mean', 'std']):
    """
    Automates group-based statistical feature engineering.
    """
    for col in group_cols:
        for stat in stats:
            new_col_name = f"{target_col}_to_{stat}_{col}"
            df[new_col_name] = df[target_col] / df.groupby([col])[target_col].transform(stat)

    print(f"Created {len(group_cols) * len(stats)} new statistical features.")
    return df

def add_frequency_encoding(df, cols):
    """
    Performs frequency encoding on specified columns.
    Creates a new feature '{column}_count' for each input column.
    """
    for col in cols:
        new_col_name = f"{col}_count"
        df[new_col_name] = df[col].map(df[col].value_counts(dropna=False))

    print(f"Frequency encoding completed for: {cols}")
    return df


def merge_datasets(df_transaction, df_identity, join_col='TransactionID'):
    """
    Performs a left join between transaction and identity dataframes.
    Ensures that all transactions are kept.
    """
    print(f"🔗 Merging datasets on {join_col}...")
    df_merged = df_transaction.merge(df_identity, on=join_col, how='left')

    print(f"Merge complete! New shape: {df_merged.shape}")
    return df_merged