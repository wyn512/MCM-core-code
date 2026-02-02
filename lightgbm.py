from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve, average_precision_score

@dataclass(frozen=True)
class ModelConfig:
    fill_method: str = 'group_mean'

    n_estimators: int = 1000
    learning_rate: float = 0.1
    max_depth: int = 5
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_seed: int = 42

    test_size: float = 0.2
    random_state: int = 42

def load_data(file_path: Path):
    df = pd.read_csv(file_path)
    df = df[df['active']].copy()
    return df

def fill_missing_values(df: pd.DataFrame, method: str = 'group_mean') :
    df_filled = df.copy()
    continuous_features = ['judge_mean', 'judge_sd', 'judge_z', 
                          'judge_mean_std', 'personal_dynamic', 
                          'vote_estimate', 'vote_std']
    for feature in continuous_features:
        if feature in df_filled.columns:
            df_filled[feature] = df_filled.groupby(['season', 'week'])[feature].transform(
                lambda x: x.fillna(x.mean())
            )
            if df_filled[feature].isnull().any():
                df_filled[feature] = df_filled[feature].fillna(df_filled[feature].mean())
    return df_filled

def calculate_weekly_percentage(df: pd.DataFrame, feature: str) :
    def percentage_transform(group):
        if group.sum() == 0:
            return pd.Series([0.0] * len(group), index=group.index)
        return (group / group.sum()) * 100
    return df.groupby(['season', 'week'])[feature].transform(percentage_transform)


def apply_weekly_percentage_transform(df: pd.DataFrame):
    df_transformed = df.copy()
    features_to_transform = ['judge_mean', 'judge_sd', 'judge_z', 
                           'weeks_participated_so_far', 'vote_estimate']
    for feature in features_to_transform:
        if feature in df_transformed.columns:
            new_feature_name = f"{feature}_pct"
            df_transformed[new_feature_name] = calculate_weekly_percentage(df_transformed, feature)
    return df_transformed

def create_features(df: pd.DataFrame):
    df_features = df.copy()
    industry_dummies = pd.get_dummies(df_features['celebrity_industry'], prefix='industry', drop_first=True)
    df_features = pd.concat([df_features, industry_dummies], axis=1)
    df_features['is_us_contestant'] = df_features['celebrity_homecountry_region'].apply(
        lambda x: 1 if str(x).lower() == 'united states' else 0
    )
    popular_states = {'california', 'texas', 'new york', 'florida'}
    df_features['is_popular_state'] = df_features['celebrity_homestate'].apply(
        lambda x: 1 if str(x).lower() in popular_states else 0
    )
    df_features['age_std'] = (df_features['celebrity_age_during_season'] - 
                              df_features['celebrity_age_during_season'].mean()) / \
                             df_features['celebrity_age_during_season'].std()
    df_features['week_std'] = (df_features['week'] - df_features['week'].mean()) / \
                             df_features['week'].std()
    df_features['progress'] = df_features.groupby(['celebrity_name', 'season']).apply(
        lambda group: group['week'] / group['week'].max()
    ).reset_index(level=[0, 1], drop=True)
    df_features['performance_trend'] = df_features.groupby(['celebrity_name', 'season'])['judge_mean'].transform(
        lambda x: x.rolling(window=min(3, len(x)), min_periods=1).mean()
    )
    return df_features

def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_columns = []
    continuous_features = ['judge_mean_pct', 'judge_sd_pct', 'judge_z_pct', 
                          'weeks_participated_so_far_pct', 'vote_estimate_pct',
                          'age_std', 'week_std', 'progress', 'performance_trend']
    encoded_features = [col for col in df.columns if col.startswith('industry_')]
    binary_features = ['is_us_contestant', 'is_popular_state']
    for feature in continuous_features + encoded_features + binary_features:
        if feature in df.columns:
            feature_columns.append(feature)
    X = df[feature_columns]
    y = df['vote_estimate']
    return X, y, feature_columns

def build_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series, 
                        config: ModelConfig = ModelConfig()) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.random_seed,
        objective='regression',
        metric='rmse'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    return model

def evaluate_model(model: lgb.LGBMRegressor, X_test: pd.DataFrame, 
                  y_test: pd.Series) -> Dict[str, float]:
    metrics = {}
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['rmse'] = rmse

    r2 = model.score(X_test, y_test)
    metrics['r2'] = r2

    spearman_corr, spearman_pvalue = stats.spearmanr(y_test, y_pred)
    metrics['spearman_correlation'] = spearman_corr
    metrics['spearman_pvalue'] = spearman_pvalue

    return metrics


def estimate_votes_with_lightgbm(df: pd.DataFrame, model: lgb.LGBMRegressor, 
                                 feature_columns: List[str]):
    df_estimated = df.copy()
    X = df_estimated[feature_columns]
    df_estimated['vote_estimate_lgbm'] = model.predict(X)
    residual_std = np.std(model.predict(X) - df_estimated['vote_estimate'])
    df_estimated['vote_std_lgbm'] = residual_std
    df_estimated['certainty_lgbm'] = 1.0 - (df_estimated['vote_std_lgbm'] / 
                                          np.abs(df_estimated['vote_estimate_lgbm']))
    df_estimated['certainty_lgbm'] = df_estimated['certainty_lgbm'].clip(0, 1)
    df_estimated['predicted_rank_lgbm'] = df_estimated.groupby(['season', 'week'])['vote_estimate_lgbm'].rank(
        method='average', ascending=False
    )
    return df_estimated


def calculate_elimination_consistency(df: pd.DataFrame):
    metrics = {}
    consistency_count = 0
    total_count = 0
    elimination_weeks = df[df['eliminated_this_week']].groupby(['season', 'week']).size().index
    
    for (season, week) in elimination_weeks:
        group = df[(df['season'] == season) & (df['week'] == week)]
        if len(group) > 1:
            lowest_vote_idx = group['vote_estimate_lgbm'].idxmin()
            eliminated = group.loc[lowest_vote_idx, 'eliminated_this_week']
            if eliminated:
                consistency_count += 1
            total_count += 1
    
    if total_count > 0:
        metrics['elimination_consistency'] = consistency_count / total_count
    else:
        metrics['elimination_consistency'] = 0.0

    metrics['average_certainty'] = df['certainty_lgbm'].mean()
    metrics['certainty_std'] = df['certainty_lgbm'].std()
    
    return metrics


def evaluate_elimination_prediction(df: pd.DataFrame, output_dir: Optional[Path] = None):
    metrics = {}

    def calculate_relative_vote(group):
        min_vote = group['vote_estimate_lgbm'].min()
        max_vote = group['vote_estimate_lgbm'].max()
        range_vote = max_vote - min_vote if max_vote > min_vote else 1
        group['relative_vote'] = (group['vote_estimate_lgbm'] - min_vote) / range_vote
        return group
    
    df_eval = df.groupby(['season', 'week']).apply(calculate_relative_vote).reset_index(drop=True)
    df_eval['elimination_prob'] = 1 - df_eval['relative_vote']

    y_true = df_eval['eliminated_this_week'].astype(int)
    y_score = df_eval['elimination_prob']

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    metrics['roc_auc'] = roc_auc

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    metrics['average_precision'] = average_precision

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Elimination Prediction')
        plt.legend(loc="lower right")
        roc_path = output_dir / 'roc_curve.png'
        plt.savefig(roc_path)
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (PR) Curve for Elimination Prediction')
        plt.legend(loc="best")
        pr_path = output_dir / 'pr_curve.png'
        plt.savefig(pr_path)
        plt.close()
        
        print(f"ROC curve saved to: {roc_path}")
        print(f"PR curve saved to: {pr_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to save results')
    parser.add_argument('--config', help='Path to model config JSON')
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        cfg = ModelConfig(**config_dict)
    else:
        cfg = ModelConfig()

    df = load_data(in_path)
    df_filled = fill_missing_values(df, method=cfg.fill_method)
    df_transformed = apply_weekly_percentage_transform(df_filled)
    df_features = create_features(df_transformed)
    X, y, feature_columns = prepare_model_data(df_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    model = build_lightgbm_model(X_train, y_train, cfg)
    evaluation_metrics = evaluate_model(model, X_test, y_test)
    df_results = estimate_votes_with_lightgbm(df_features, model, feature_columns)
    consistency_metrics = calculate_elimination_consistency(df_results)
    output_dir = out_path.parent / 'evaluation_plots'
    elimination_metrics = evaluate_elimination_prediction(df_results, output_dir)
    df_results.to_csv(out_path, index=False)

    metrics_path = out_path.with_suffix('.metrics.json')
    all_metrics = {
        'evaluation': evaluation_metrics,
        'consistency': consistency_metrics,
        'elimination_prediction': elimination_metrics,
        'model_config': cfg.__dict__,
        'feature_columns': feature_columns
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main())