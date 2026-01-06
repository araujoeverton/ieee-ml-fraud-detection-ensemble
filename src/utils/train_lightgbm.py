import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ==========================================
# 1. GLOBAL CONFIGURATIONS
# ==========================================
INPUT_PATH = Path('../input/train_featured.parquet')
MODELS_PATH = Path('../models')
PLOTS_PATH = Path('../plots')

# Ensure directories exist
MODELS_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH.mkdir(exist_ok=True, parents=True)

# LightGBM Hyperparameters
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'is_unbalance': True,
    'learning_rate': 0.02,
    'num_leaves': 128,
    'max_depth': -1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'n_jobs': -1,
    'seed': 42,
    'verbosity': -1
}


# ==========================================
# 2. FUNCTIONS
# ==========================================

def train_robust_model():
    """
    Trains the model using TimeSeriesSplit and returns feature importance
    and last fold data for financial analysis.
    """
    print("[INFO] Loading dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    # 1. Temporal Sorting
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    # Capture raw values for financial analysis
    if 'TransactionAmt' in df.columns:
        transaction_amt = df['TransactionAmt']
    elif 'TransactionAmt_Log' in df.columns:
        transaction_amt = np.expm1(df['TransactionAmt_Log'])
    else:
        raise ValueError("Column TransactionAmt or TransactionAmt_Log required.")

    # 2. Split Features and Target
    cols_to_drop = ['isFraud', 'TransactionID', 'TransactionDT', 'TransactionAmt']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df['isFraud']

    del df
    gc.collect()

    print(f"[INFO] Dataset Shape: {X.shape}")

    # 3. Time-Based Cross-Validation (5 Folds)
    folds = TimeSeriesSplit(n_splits=5)

    aucs = []
    feature_importance_df = pd.DataFrame()
    last_fold_data = {}

    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"\n[INFO] Starting Fold {fold + 1}...")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=500)
            ]
        )

        val_preds = model.predict(X_val)
        score = roc_auc_score(y_val, val_preds)
        aucs.append(score)
        print(f"[RESULT] Fold {fold + 1} AUC: {score:.4f}")

        # Save Importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = X.columns
        fold_importance["importance"] = model.feature_importance(importance_type='gain')
        fold_importance["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

        # Save data for the last fold
        if fold == 4:
            joblib.dump(model, MODELS_PATH / 'lgbm_model_final.joblib')
            last_fold_data = {
                'X_val': X_val,
                'y_true': y_val.values,
                'y_prob': val_preds,
                'amounts': transaction_amt.iloc[val_idx].values
            }

        del X_train, y_train, X_val, y_val, dtrain, dval, model
        gc.collect()

    print("-" * 30)
    print(f"[FINAL] Average AUC (CV): {np.mean(aucs):.4f}")
    print("-" * 30)

    return feature_importance_df, last_fold_data


def plot_feature_importance(importance_df):
    """Generates a plot of the Top 50 features."""
    cols = (importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)

    best_features = importance_df.loc[importance_df.feature.isin(cols)]

    plt.figure(figsize=(12, 10))
    sns.barplot(
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        data=best_features.sort_values(by="importance", ascending=False),
        palette="viridis"
    )
    plt.title('LightGBM Feature Importance (Avg Gain over Folds)')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()

    output_file = PLOTS_PATH / '1_feature_importance.png'
    plt.savefig(output_file)
    print(f"[PLOT] Feature importance plot saved at: {output_file}")


def optimize_financial_threshold(y_true, y_prob, amounts):
    """Calculates the best financial threshold."""
    thresholds = np.linspace(0.5, 0.99, 100)
    results = []

    ADMIN_COST = 5.0
    INSULT_RATE = 0.02

    total_fraud_value = amounts[y_true == 1].sum()
    print(f"\n[INFO] Total Fraud Value in Validation Set: ${total_fraud_value:,.2f}")

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)

        saved_fraud = amounts[tp_mask].sum()
        intervention_cost = (fp_mask.sum() * ADMIN_COST) + (amounts[fp_mask].sum() * INSULT_RATE)

        results.append(saved_fraud - intervention_cost)

    best_idx = np.argmax(results)
    best_thresh = thresholds[best_idx]
    best_savings = results[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, results, color='#2a9d8f', linewidth=3)
    plt.axvline(best_thresh, color='#e76f51', linestyle='--', label=f'Best Threshold: {best_thresh:.2f}')
    plt.title('Net Savings by Probability Threshold')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Net Savings ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_file = PLOTS_PATH / '2_financial_threshold_curve.png'
    plt.savefig(output_file)

    print(f"[RESULT] Optimal Threshold: {best_thresh:.2f}")
    print(f"[RESULT] Max Potential Savings: ${best_savings:,.2f}")
    print(f"[PLOT] Financial curve saved at: {output_file}")

    return best_thresh


def analyze_error_patterns(last_fold_data, threshold):
    """
    Compares Normal Legit transactions vs False Positives.
    """
    print("\n[INFO] Starting False Positive Analysis...")

    X_val = last_fold_data['X_val']
    y_true = last_fold_data['y_true']
    y_prob = last_fold_data['y_prob']

    df_analysis = X_val.copy()
    df_analysis['isFraud'] = y_true
    df_analysis['prob'] = y_prob

    # Define groups
    # TN: Legit classified as Legit
    # FP: Legit classified as Fraud (Error)
    tn_mask = (y_true == 0) & (y_prob <= threshold)
    fp_mask = (y_true == 0) & (y_prob > threshold)

    df_tn = df_analysis[tn_mask]
    df_fp = df_analysis[fp_mask]

    print(f"[INFO] True Negatives (Correct Legit): {len(df_tn)}")
    print(f"[INFO] False Positives (Incorrect Legit): {len(df_fp)}")

    if len(df_fp) == 0:
        print("[WARNING] No False Positives found with this threshold.")
        return

    # Plot 1: Hour of Day
    if 'hour' in df_analysis.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df_tn['hour'], label='Legit (TN)', fill=True, color='green', alpha=0.3)
        sns.kdeplot(df_fp['hour'], label='False Positives (FP)', fill=True, color='red', alpha=0.3)
        plt.title('Error Analysis: Time of Day Distribution')
        plt.legend()
        plt.savefig(PLOTS_PATH / '3_error_analysis_hour.png')
        print("[PLOT] Hour analysis saved.")

    # Plot 2: Transaction Amount (Log)
    if 'TransactionAmt_Log' in df_analysis.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df_tn['TransactionAmt_Log'], label='Legit (TN)', fill=True, color='green', alpha=0.3)
        sns.kdeplot(df_fp['TransactionAmt_Log'], label='False Positives (FP)', fill=True, color='red', alpha=0.3)
        plt.title('Error Analysis: Transaction Amount (Log)')
        plt.legend()
        plt.savefig(PLOTS_PATH / '3_error_analysis_amount.png')
        print("[PLOT] Amount analysis saved.")

        # Plot 3: Card Frequency
        if 'card1_count' in df_analysis.columns:
            plt.figure(figsize=(10, 6))

            plot_data = pd.DataFrame({
                'Count': pd.concat([df_tn['card1_count'], df_fp['card1_count']], ignore_index=True),
                'Type': ['Legit (TN)'] * len(df_tn) + ['False Positives (FP)'] * len(df_fp)
            })

            sns.boxplot(
                data=plot_data,
                x='Type',
                y='Count',
                hue='Type',
                palette={'Legit (TN)': 'green', 'False Positives (FP)': 'red'}
            )
            # -----------------------------------------------------------

            plt.title('Error Analysis: Card Frequency (card1_count)')
            plt.yscale('log')
            plt.xlabel('')

            plt.savefig(PLOTS_PATH / '3_error_analysis_card1_count.png')
            print("[PLOT] Frequency analysis saved.")