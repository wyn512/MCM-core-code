from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import statsmodels.api as sm
from scipy import stats

@dataclass(frozen=True)
class ModelConfig:
    beta_prior_mu: float = 0.0
    beta_prior_sigma: float = 3.0
    lambda_prior_alpha: float = 1.0
    lambda_prior_beta: float = 1.0
    sigma_prior_mu: float = 0.0
    sigma_prior_sigma: float = 5.0
    n_draws: int = 1000
    n_tune: int = 500
    chains: int = 2
    random_seed: int = 42

def load_cleaned_data(file_path: Path):
    df = pd.read_csv(file_path)
    df = df[df['active']].copy()
    df['judge_mean_std'] = (df['judge_mean'] - df['judge_mean'].mean()) / df['judge_mean'].std()
    df['personal_dynamic'] = df.groupby(['celebrity_name', 'season']).apply(
        lambda group: (group['weeks_participated_so_far'] / group['weeks_participated_so_far'].max()) * 
        (1 + group['judge_mean_std'].cumsum() / len(group))
    ).reset_index(level=[0, 1], drop=True)
    df['personal_dynamic'] = (df['personal_dynamic'] - df['personal_dynamic'].mean()) / df['personal_dynamic'].std()
    popular_states = {'california', 'texas', 'new york', 'florida'}
    df['region_match'] = df.apply(
        lambda row:
            1.0 if str(row['celebrity_homecountry_region']).lower() == 'united states' and 
            str(row['celebrity_homestate']).lower() in popular_states else
            0.8 if str(row['celebrity_homecountry_region']).lower() == 'united states' else
            0.5,
        axis=1
    )

    df['industry_actor'] = df['celebrity_industry'].apply(lambda x: 1.0 if 'actor' in str(x).lower() else 0.0)
    df['industry_athlete'] = df['celebrity_industry'].apply(lambda x: 1.0 if 'athlete' in str(x).lower() else 0.0)
    df['industry_singer'] = df['celebrity_industry'].apply(lambda x: 1.0 if 'singer' in str(x).lower() else 0.0)
    df['industry_model'] = df['celebrity_industry'].apply(lambda x: 1.0 if 'model' in str(x).lower() else 0.0)
    df['age_std'] = (df['celebrity_age_during_season'] - df['celebrity_age_during_season'].mean()) / df['celebrity_age_during_season'].std()
    df['judge_sd_std'] = (df['judge_sd'] - df['judge_sd'].mean()) / df['judge_sd'].std()
    df['week_std'] = (df['week'] - df['week'].mean()) / df['week'].std()
    df['relative_performance'] = df.groupby(['season', 'week'])['judge_mean_std'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['performance_trend'] = df.groupby(['celebrity_name', 'season'])['judge_mean_std'].transform(
        lambda x: x.rolling(window=min(3, len(x)), min_periods=1).mean()
    )
    df['judge_industry_interaction'] = df['judge_mean_std'] * (df['industry_actor'] + df['industry_athlete'] + df['industry_singer'] + df['industry_model'])
    df['judge_region_interaction'] = df['judge_mean_std'] * df['region_match']
    df['week_trend_interaction'] = df['week_std'] * df['performance_trend']
    return df

def create_bayesian_gam_model(
    df: pd.DataFrame,
    cfg: ModelConfig = ModelConfig()
):
    with pm.Model() as model:
        Sw = pm.Data('Sw', df['judge_mean_std'].values)
        I = pm.Data('I', df['personal_dynamic'].values)
        R = pm.Data('R', df['region_match'].values)
        industry_actor = pm.Data('industry_actor', df['industry_actor'].values)
        industry_athlete = pm.Data('industry_athlete', df['industry_athlete'].values)
        industry_singer = pm.Data('industry_singer', df['industry_singer'].values)
        industry_model = pm.Data('industry_model', df['industry_model'].values)
        week_std = pm.Data('week_std', df['week_std'].values)
        relative_performance = pm.Data('relative_performance', df['relative_performance'].values)

        beta0 = pm.Normal('beta0', mu=0.0, sigma=1.5)
        beta1 = pm.Normal('beta1', mu=1.5, sigma=0.8)
        beta2 = pm.Normal('beta2', mu=0.8, sigma=0.8)
        beta3 = pm.Normal('beta3', mu=0.4, sigma=0.4)
        beta4 = pm.Normal('beta4', mu=0.3, sigma=0.4)
        beta5 = pm.Normal('beta5', mu=0.4, sigma=0.4)
        beta6 = pm.Normal('beta6', mu=0.3, sigma=0.4)
        beta7 = pm.Normal('beta7', mu=0.2, sigma=0.4)
        beta8 = pm.Normal('beta8', mu=0.0, sigma=0.4)
        beta9 = pm.Normal('beta9', mu=0.8, sigma=0.4)

        f1 = beta2 * I + pm.Normal('gamma1', mu=0, sigma=0.3) * I**2
        f3 = beta1 * Sw + pm.Normal('gamma3', mu=0, sigma=0.3) * Sw**2
        mu = (
            beta0 + 
            f3 +
            f1 +
            beta3 * R +
            beta4 * industry_actor +
            beta5 * industry_athlete +
            beta6 * industry_singer +
            beta7 * industry_model +
            beta8 * week_std +
            beta9 * relative_performance
        )

        sigma = pm.HalfNormal('sigma', sigma=cfg.sigma_prior_sigma)
        y_obs = df['judge_mean_std'].copy()
        y_obs[df['eliminated_this_week']] -= 2.0
        y_obs = (y_obs - y_obs.mean()) / y_obs.std()
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_obs)
    return model


def fit_model(
    model: pm.Model,
    cfg: ModelConfig = ModelConfig()
) -> pm.backends.base.MultiTrace:
    with model:
        trace = pm.sample(
            draws=cfg.n_draws,
            tune=cfg.n_tune,
            chains=cfg.chains,
            random_seed=cfg.random_seed,
            return_inferencedata=False
        )
    return trace

def estimate_votes(
    df: pd.DataFrame,
    trace: pm.backends.base.MultiTrace
):
    beta0_mean = trace['beta0'].mean()
    beta1_mean = trace['beta1'].mean()
    beta2_mean = trace['beta2'].mean()
    beta3_mean = trace['beta3'].mean()
    beta4_mean = trace['beta4'].mean()
    beta5_mean = trace['beta5'].mean()
    beta6_mean = trace['beta6'].mean()
    beta7_mean = trace['beta7'].mean()
    beta8_mean = trace['beta8'].mean()
    beta9_mean = trace['beta9'].mean()
    gamma1_mean = trace['gamma1'].mean()
    gamma3_mean = trace['gamma3'].mean()
    sigma_mean = trace['sigma'].mean()

    f1_mean = beta2_mean * df['personal_dynamic'] + gamma1_mean * df['personal_dynamic']**2
    f3_mean = beta1_mean * df['judge_mean_std'] + gamma3_mean * df['judge_mean_std']**2

    df['vote_estimate'] = (
        beta0_mean + 
        f3_mean +
        f1_mean +
        beta3_mean * df['region_match'] +
        beta4_mean * df['industry_actor'] +
        beta5_mean * df['industry_athlete'] +
        beta6_mean * df['industry_singer'] +
        beta7_mean * df['industry_model'] +
        beta8_mean * df['week_std'] +
        beta9_mean * df['relative_performance']
    )

    df['vote_std'] = sigma_mean
    df['certainty'] = 1.0 - (df['vote_std'] / np.abs(df['vote_estimate']))
    df['certainty'] = df['certainty'].clip(0, 1)  # 确保在[0,1]范围内
    df['predicted_rank'] = df.groupby(['season', 'week'])['vote_estimate'].rank(
        method='average', ascending=False
    )
    return df

def calculate_consistency_metrics(
    df: pd.DataFrame
):
    metrics = {}
    consistency_count = 0
    total_count = 0
    elimination_weeks = df[df['eliminated_this_week']].groupby(['season', 'week']).size().index
    
    for (season, week) in elimination_weeks:
        group = df[(df['season'] == season) & (df['week'] == week)]
        if len(group) > 1:
            lowest_vote_idx = group['vote_estimate'].idxmin()
            lowest_vote_contestant = group.loc[lowest_vote_idx, 'celebrity_name']
            eliminated = group.loc[lowest_vote_idx, 'eliminated_this_week']
            if eliminated:
                consistency_count += 1
            total_count += 1
    
    if total_count > 0:
        metrics['elimination_consistency'] = consistency_count / total_count
    else:
        metrics['elimination_consistency'] = 0.0

    metrics['average_certainty'] = df['certainty'].mean()
    metrics['certainty_std'] = df['certainty'].std()
    return metrics

def perform_model_validation(
    df: pd.DataFrame,
    model: pm.Model,
    trace: pm.backends.base.MultiTrace
):
    validation_metrics = {}

    beta0_mean = trace['beta0'].mean()
    beta1_mean = trace['beta1'].mean()
    beta2_mean = trace['beta2'].mean()
    beta3_mean = trace['beta3'].mean()
    beta4_mean = trace['beta4'].mean()
    beta5_mean = trace['beta5'].mean()
    beta6_mean = trace['beta6'].mean()
    beta7_mean = trace['beta7'].mean()
    beta8_mean = trace['beta8'].mean()
    beta9_mean = trace['beta9'].mean()
    gamma1_mean = trace['gamma1'].mean()
    gamma3_mean = trace['gamma3'].mean()
    f1_mean = beta2_mean * df['personal_dynamic'] + gamma1_mean * df['personal_dynamic']**2
    f3_mean = beta1_mean * df['judge_mean_std'] + gamma3_mean * df['judge_mean_std']**2
    y_pred = (
        beta0_mean + 
        f3_mean +
        f1_mean +
        beta3_mean * df['region_match'] +
        beta4_mean * df['industry_actor'] +
        beta5_mean * df['industry_athlete'] +
        beta6_mean * df['industry_singer'] +
        beta7_mean * df['industry_model'] +
        beta8_mean * df['week_std'] +
        beta9_mean * df['relative_performance']
    )
    y_obs = df['judge_mean_std'].copy()
    y_obs[df['eliminated_this_week']] -= 2.0
    y_obs = (y_obs - y_obs.mean()) / y_obs.std()
    residuals = y_obs - y_pred
    bg_test = sm.stats.diagnostic.acorr_breusch_godfrey(
        sm.OLS(residuals, sm.add_constant(df['judge_mean_std'])).fit(),
        nlags=4
    )
    validation_metrics['bg_test_statistic'] = bg_test[0]
    validation_metrics['bg_test_pvalue'] = bg_test[1]
    validation_metrics['bg_test_df'] = bg_test[2]

    try:
        with model:
            ppc = pm.sample_posterior_predictive(trace, var_names=['y'], random_seed=42)

            y_observed = y_obs.values
            y_predicted = ppc.posterior_predictive['y'].values
            p_values = []
            for i, obs in enumerate(y_observed):
                posterior_pred = y_predicted[:, i]
                p_value = (posterior_pred < obs).mean()
                p_value = 2 * min(p_value, 1 - p_value)
                p_values.append(p_value)
            ks_stat, ks_pvalue = stats.kstest(
                y_observed,
                y_predicted.flatten()
            )
            validation_metrics['ppp_test_pvalue'] = ks_pvalue
            validation_metrics['ppp_ks_statistic'] = ks_stat
            validation_metrics['average_pp_value'] = np.mean(p_values)
    except Exception as e:
        print(f"Error performing PPP test: {e}")
        validation_metrics['ppp_test_pvalue'] = None
        validation_metrics['ppp_ks_statistic'] = None
        validation_metrics['average_pp_value'] = None
    return validation_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to cleaned data CSV')
    parser.add_argument('--out', required=True, help='Path to write results')
    parser.add_argument('--config', help='Path to model config JSON')
    parser.add_argument('--celebrity', help='Name of celebrity to generate rank density plot for')
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        cfg = ModelConfig(**config_dict)
    else:
        cfg = ModelConfig()

    df = load_cleaned_data(in_path)
    model = create_bayesian_gam_model(df, cfg)
    trace = fit_model(model, cfg)
    df_results = estimate_votes(df, trace)
    consistency_metrics = calculate_consistency_metrics(df_results)
    validation_metrics = perform_model_validation(df_results, model, trace)
    output_dir = out_path.parent / 'model_plots'
    df_results.to_csv(out_path, index=False)
    metrics_path = out_path.with_suffix('.metrics.json')
    all_metrics = {
        'consistency': consistency_metrics,
        'validation': validation_metrics,
        'model_config': cfg.__dict__
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
