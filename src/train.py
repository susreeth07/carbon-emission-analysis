"""
Carbon Emission Analysis - Phase 3 ML Model Comparison & Controlled Improvement
Author: Machine Learning Engineering Pipeline
Description:
    Trains and compares Baseline Ridge Regression and Random Forest Regressor
    on the Carbon Majors dataset (emissions_medium_granularity.csv).
    Ensures zero data leakage, consistent 70/30 train/test evaluation,
    outputs comparison metrics (CSV & JSON), and generates diagnostic plots
    including feature importances.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_validate_data(data_path: str):
    """
    Loads dataset and validates columns, reporting any rows removed due to invalid/missing target.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    initial_rows = len(df)
    print(f"[1/7] Loaded dataset from '{data_path}' with shape: {df.shape}")

    feature_cols = ['year', 'production_value', 'commodity', 'parent_type']
    target_col = 'total_emissions_MtCO2e'
    expected_cols = feature_cols + [target_col]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Drop only if essential target data or features are missing/invalid
    valid_mask = df[target_col].notnull() & (df[target_col] >= 0)
    for col in feature_cols:
        valid_mask = valid_mask & df[col].notnull()

    df_cleaned = df[valid_mask].copy()
    rows_removed = initial_rows - len(df_cleaned)

    if rows_removed > 0:
        print(f"      Rows removed: {rows_removed} due to invalid or missing data.")
    else:
        print("      Data integrity check passed: 0 missing or invalid rows found. No rows removed.")

    return df_cleaned, feature_cols, target_col


def get_preprocessor():
    """
    Constructs the ColumnTransformer for feature transformations.
    Fitted exclusively on training data to prevent data leakage.
    """
    log_feature = ['production_value']
    cat_features = ['commodity', 'parent_type']
    pass_features = ['year']

    return ColumnTransformer(
        transformers=[
            (
                'log_transform',
                FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
                log_feature
            ),
            (
                'one_hot_encode',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                cat_features
            ),
            (
                'numeric_passthrough',
                'passthrough',
                pass_features
            )
        ]
    )


def build_models():
    """
    Creates the dictionary of pipelines to be compared.
    """
    models = {
        "Ridge Regression (Baseline)": Pipeline(steps=[
            ('preprocessor', get_preprocessor()),
            ('regressor', Ridge(alpha=1.0))
        ]),
        "Random Forest Regressor": Pipeline(steps=[
            ('preprocessor', get_preprocessor()),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        ])
    }
    return models


def generate_evaluation_plots(y_test, predictions_dict, output_plots_dir):
    """
    Generates predicted-vs-actual and residual comparison plots.
    """
    os.makedirs(output_plots_dir, exist_ok=True)

    # 1. Predicted vs. Actual Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    models = list(predictions_dict.keys())

    for idx, model_name in enumerate(models):
        y_pred = predictions_dict[model_name]
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.3, color='#1f77b4' if idx == 0 else '#2ca02c', edgecolors='none')
        min_v = min(y_test.min(), y_pred.min())
        max_v = max(y_test.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='Ideal Fit (y = x)')
        ax.set_title(f'{model_name}\nPredicted vs. Actual', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Emissions (MtCO2e)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Predicted Emissions (MtCO2e)', fontsize=11)
        ax.legend(frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    pred_vs_act_path = os.path.join(output_plots_dir, 'predicted_vs_actual.png')
    plt.savefig(pred_vs_act_path, dpi=300)
    plt.close()

    # 2. Residuals Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, model_name in enumerate(models):
        y_pred = predictions_dict[model_name]
        residuals = y_test - y_pred
        ax = axes[idx]
        ax.scatter(y_pred, residuals, alpha=0.3, color='#ff7f0e' if idx == 0 else '#9467bd', edgecolors='none')
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Residual Line')
        ax.set_title(f'{model_name}\nResiduals vs. Predicted', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Emissions (MtCO2e)', fontsize=11)
        ax.set_ylabel('Residuals (Actual - Predicted) [MtCO2e]', fontsize=11)
        ax.legend(frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    residual_path = os.path.join(output_plots_dir, 'residual_plot.png')
    plt.savefig(residual_path, dpi=300)
    plt.close()

    return pred_vs_act_path, residual_path


def generate_rf_feature_importance(rf_pipeline, output_plots_dir):
    """
    Extracts and visualizes impurity-based feature importances from the Random Forest model.
    """
    preprocessor = rf_pipeline.named_steps['preprocessor']
    regressor = rf_pipeline.named_steps['regressor']

    raw_names = preprocessor.get_feature_names_out()
    # Clean up prefixes for readable display
    clean_names = [
        name.replace('log_transform__', '')
            .replace('one_hot_encode__', '')
            .replace('numeric_passthrough__', '')
            .replace('commodity_', 'Commodity: ')
            .replace('parent_type_', 'Parent: ')
            .replace('production_value', 'log1p(Production Value)')
            .replace('year', 'Year')
        for name in raw_names
    ]

    importances = regressor.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': clean_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color='#2b5c8f', edgecolor='black', alpha=0.85)
    ax.set_xlabel('MDI Feature Importance (Mean Decrease in Impurity)', fontsize=11, fontweight='bold')
    ax.set_title('Random Forest Feature Importance (Impurity-Based)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, axis='x')

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        if width > 0.005:
            ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                    ha='left', va='center', fontsize=9)

    plt.tight_layout()
    fi_plot_path = os.path.join(output_plots_dir, 'rf_feature_importance.png')
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()

    return feat_df.sort_values(by='Importance', ascending=False), fi_plot_path


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Model Comparison & Controlled Improvement")
    parser.add_argument("--data_path", type=str, default="data/emissions_medium_granularity.csv")
    parser.add_argument("--output_models_dir", type=str, default="outputs/models")
    parser.add_argument("--output_metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--output_plots_dir", type=str, default="outputs/plots")
    args = parser.parse_args()

    print("=" * 75)
    print("  PHASE 3: MODEL COMPARISON & CONTROLLED IMPROVEMENT")
    print("=" * 75)

    # 1. Load & validate dataset
    df, feature_cols, target_col = load_and_validate_data(args.data_path)
    X = df[feature_cols]
    y = df[target_col]

    # 2. Strict 70/30 split using random_state=42 and shuffle=True
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )
    print(f"[2/7] Split data: Train = {len(X_train)} rows (70.0%), Test = {len(X_test)} rows (30.0%)")
    print(f"      Selected Features: {feature_cols}")
    print(f"      Target Column:     '{target_col}'")

    # 3. Model instantiation
    models = build_models()
    metrics_summary = []
    predictions_dict = {}

    os.makedirs(args.output_models_dir, exist_ok=True)
    os.makedirs(args.output_metrics_dir, exist_ok=True)

    # 4. Train and evaluate models on identical data
    for name, pipeline in models.items():
        print(f"[3/7] Training pipeline: '{name}' on training data...")
        pipeline.fit(X_train, y_train)

        print(f"      Evaluating '{name}' on untouched test set...")
        y_pred = pipeline.predict(X_test)
        predictions_dict[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics_summary.append({
            "Model": name,
            "MAE_MtCO2e": round(mae, 4),
            "RMSE_MtCO2e": round(rmse, 4),
            "R2_Score": round(r2, 4),
            "Test_Samples": len(y_test)
        })

        # Save individual model pipeline
        file_slug = "ridge_pipeline.joblib" if "Ridge" in name else "random_forest_pipeline.joblib"
        model_save_path = os.path.join(args.output_models_dir, file_slug)
        joblib.dump(pipeline, model_save_path)
        print(f"      Saved model to: {model_save_path}")

    # 5. Create comparison DataFrame and save as CSV
    comparison_df = pd.DataFrame(metrics_summary)
    comparison_csv_path = os.path.join(args.output_metrics_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)

    # Save detailed JSON metrics
    metrics_json_path = os.path.join(args.output_metrics_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # 6. Generate diagnostic & comparison plots
    print("[5/7] Generating evaluation plots (Predicted vs. Actual, Residuals)...")
    p_path, r_path = generate_evaluation_plots(y_test, predictions_dict, args.output_plots_dir)

    print("[6/7] Generating Random Forest feature importance plot...")
    rf_pipeline = models["Random Forest Regressor"]
    feat_importance_df, fi_plot_path = generate_rf_feature_importance(rf_pipeline, args.output_plots_dir)

    # 7. Model selection recommendation based on test metrics
    best_model_row = comparison_df.sort_values(by="RMSE_MtCO2e", ascending=True).iloc[0]
    selected_model = best_model_row["Model"]

    print("[7/7] Execution and evaluation finished successfully!")
    print("-" * 75)
    print("MODEL COMPARISON TABLE (TEST SET - ORIGINAL MtCO2e UNITS):")
    print(comparison_df.to_string(index=False))
    print("-" * 75)
    print(f"PREFERRED MODEL SELECTED: '{selected_model}'")
    print(f"  • Test MAE:   {best_model_row['MAE_MtCO2e']} MtCO2e (Lower is better)")
    print(f"  • Test RMSE:  {best_model_row['RMSE_MtCO2e']} MtCO2e (Lower is better)")
    print(f"  • Test R²:    {best_model_row['R2_Score']} (Higher is better)")
    print("-" * 75)
    print("GENERATED ARTIFACTS AND OUTPUT PATHS:")
    print(f"  • Model Comparison CSV:       {os.path.abspath(comparison_csv_path)}")
    print(f"  • Evaluation Metrics JSON:     {os.path.abspath(metrics_json_path)}")
    print(f"  • Ridge Model Pipeline:       {os.path.abspath(os.path.join(args.output_models_dir, 'ridge_pipeline.joblib'))}")
    print(f"  • Random Forest Pipeline:     {os.path.abspath(os.path.join(args.output_models_dir, 'random_forest_pipeline.joblib'))}")
    print(f"  • Predicted vs. Actual Plot:  {os.path.abspath(p_path)}")
    print(f"  • Residual Plot:              {os.path.abspath(r_path)}")
    print(f"  • RF Feature Importance Plot: {os.path.abspath(fi_plot_path)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
