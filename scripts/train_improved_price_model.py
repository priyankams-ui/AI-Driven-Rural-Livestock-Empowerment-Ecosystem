"""
Train and Evaluate Improved Livestock Price Prediction Model
===========================================================
Trains an improved XGBoost Regressor on the verified Karnataka Livestock Market
Transactions dataset using an 80/10/10 split and rigorous cross-validation.
Evaluates both the existing model and the improved model on the independent test set.
Computes calibrated prediction intervals and detailed error breakdowns.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'karnataka_livestock_market_transactions.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OLD_MODEL_PATH = os.path.join(MODELS_DIR, 'price_model.pkl')
NEW_MODEL_PATH = os.path.join(MODELS_DIR, 'price_model.pkl')
FEATURES_JSON = os.path.join(MODELS_DIR, 'price_features.json')
CALIBRATION_JSON = os.path.join(MODELS_DIR, 'price_calibration.json')
METRICS_JSON = os.path.join(MODELS_DIR, 'metrics_report.json')
EVAL_RESULTS_JSON = os.path.join(MODELS_DIR, 'price_model_evaluation.json')

def train_and_evaluate():
    print("================================================================================")
    print("   LIVESTOCKIQ: IMPROVED AI LIVESTOCK PRICE PREDICTION MODEL TRAINING           ")
    print("================================================================================")

    # 1. Load Genuine Market Dataset
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    print(f"[Data] Loaded {len(df)} verified Karnataka market records from {DATA_PATH}")

    # 2. Prepare Features
    # Numeric features
    num_cols = ['age_years', 'milk_production', 'is_pregnant', 'pregnancy_month', 'health_score']
    # Categorical features including Lactation Number and Calf Number
    cat_cols = ['species', 'breed', 'gender', 'district', 'lactation_number', 'calf_number']

    features_raw = pd.get_dummies(df[cat_cols], drop_first=False)
    for col in num_cols:
        features_raw[col] = df[col].values

    # Load existing feature template if present to evaluate old model
    old_template_features = []
    if os.path.exists(FEATURES_JSON):
        try:
            with open(FEATURES_JSON, 'r', encoding='utf-8') as f:
                old_template_features = json.load(f)['feature_names']
        except Exception:
            old_template_features = []

    all_feature_cols = sorted(list(features_raw.columns))
    X = features_raw[all_feature_cols].copy()
    y = df['price'].values

    # 3. Strict 80% / 10% / 10% Split
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED)

    # Keep original metadata for test set analysis
    df_temp = df.iloc[X_temp.index]
    df_val = df_temp.iloc[:len(y_val)].copy()
    df_te = df_temp.iloc[len(y_val):].copy()

    print(f"[Split] Train Set:      {len(y_tr):4d} records (80.0%)")
    print(f"[Split] Validation Set: {len(y_val):4d} records (10.0%)")
    print(f"[Split] Test Set:       {len(y_te):4d} records (10.0%)")

    # 4. Evaluate Existing (Old) Model on Test Set
    old_metrics = {}
    if os.path.exists(OLD_MODEL_PATH):
        try:
            with open(OLD_MODEL_PATH, 'rb') as f:
                old_model = pickle.load(f)
            # Align features for old model
            X_te_old = pd.DataFrame(0, index=X_te.index, columns=old_template_features)
            for col in old_template_features:
                if col in X_te.columns:
                    X_te_old[col] = X_te[col]
            y_te_old_pred = old_model.predict(X_te_old)
            old_mae = float(mean_absolute_error(y_te, y_te_old_pred))
            old_rmse = float(np.sqrt(mean_squared_error(y_te, y_te_old_pred)))
            old_r2 = float(r2_score(y_te, y_te_old_pred))
            old_bias = float(np.mean(y_te_old_pred - y_te))
            old_within_10 = float(np.mean(np.abs(y_te - y_te_old_pred) / y_te <= 0.10))
            old_within_15 = float(np.mean(np.abs(y_te - y_te_old_pred) / y_te <= 0.15))

            old_metrics = {
                'mae': old_mae,
                'rmse': old_rmse,
                'r2_score': old_r2,
                'mean_bias_error': old_bias,
                'within_10_pct': old_within_10,
                'within_15_pct': old_within_15
            }
            print("\n[Old Model Evaluation on Independent Test Set]")
            print(f"   R2 Score:            {old_r2*100:6.2f}%")
            print(f"   MAE:                 INR {old_mae:,.2f}")
            print(f"   RMSE:                INR {old_rmse:,.2f}")
            print(f"   Mean Bias (Overpred):INR {old_bias:,.2f}")
            print(f"   Within +/-10% Acc:   {old_within_10*100:6.2f}%")
            print(f"   Within +/-15% Acc:   {old_within_15*100:6.2f}%")
        except Exception as e:
            print(f"[Warning] Could not evaluate old model: {e}")

    # 5. Hyperparameter Tuning using K-Fold CV on Training Set
    print("\n[Tuning] Optimizing XGBoost hyperparameters using 5-Fold Cross-Validation...")
    feature_weights = np.ones(X.shape[1], dtype=float)
    for i, col in enumerate(all_feature_cols):
        if 'lactation_number' in col:
            feature_weights[i] = 4.5
        elif 'calf_number' in col:
            feature_weights[i] = 0.05
        elif col in ['gender_Male', 'gender_Female']:
            feature_weights[i] = 2.0
        elif col == 'health_score':
            feature_weights[i] = 2.0

    param_grid = [
        {'n_estimators': 350, 'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 0.5, 'reg_lambda': 1.5},
        {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 1.0, 'reg_lambda': 2.0},
        {'n_estimators': 450, 'max_depth': 5, 'learning_rate': 0.04, 'subsample': 0.80, 'colsample_bytree': 0.80, 'reg_alpha': 0.8, 'reg_lambda': 2.5},
        {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.90, 'colsample_bytree': 0.85, 'reg_alpha': 0.3, 'reg_lambda': 1.5},
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 0.5, 'reg_lambda': 2.0},
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    best_params = None
    best_cv_mae = float('inf')

    for idx, params in enumerate(param_grid, 1):
        fold_maes = []
        for tr_idx, val_idx in kf.split(X_tr):
            X_fold_tr, X_fold_val = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
            y_fold_tr, y_fold_val = y_tr[tr_idx], y_tr[val_idx]

            m = xgb.XGBRegressor(
                random_state=RANDOM_SEED,
                feature_weights=feature_weights,
                **params
            )
            m.fit(X_fold_tr, y_fold_tr)
            preds = m.predict(X_fold_val)
            fold_maes.append(mean_absolute_error(y_fold_val, preds))

        avg_mae = np.mean(fold_maes)
        print(f"   Config {idx}: Depth={params['max_depth']}, N_est={params['n_estimators']}, LR={params['learning_rate']} -> CV MAE: INR {avg_mae:,.2f}")
        if avg_mae < best_cv_mae:
            best_cv_mae = avg_mae
            best_params = params

    print(f"[Tuning] Selected Best Config: {best_params} (CV MAE: INR {best_cv_mae:,.2f})")

    # 6. Fit Final Model on 80% Training Set, Validated on 10% Validation Set
    best_model = xgb.XGBRegressor(
        random_state=RANDOM_SEED,
        feature_weights=feature_weights,
        early_stopping_rounds=30,
        eval_metric='mae',
        **best_params
    )
    best_model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False
    )

    # 7. Residual Calibration on Validation Set for Prediction Intervals
    val_preds = best_model.predict(X_val)
    val_residuals = y_val - val_preds
    val_abs_residuals = np.abs(val_residuals)

    val_mae = float(mean_absolute_error(y_val, val_preds))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_r2 = float(r2_score(y_val, val_preds))

    # Conformal residual quantile for ~85% empirical coverage
    q_85 = float(np.percentile(val_abs_residuals, 85))
    residual_std = float(np.std(val_residuals))

    calibration_payload = {
        "calibration_method": "conformal_empirical_quantile",
        "target_coverage_nominal": 0.85,
        "half_width_inr": round(q_85, -1),
        "residual_std_inr": round(residual_std, -1),
        "val_mae_inr": round(val_mae, 1),
        "val_rmse_inr": round(val_rmse, 1),
        "min_market_floor_inr": 15000,
        "diminishing_marginal_yield_threshold": 20.0
    }

    # 8. Unbiased Evaluation on Held-Out Independent Test Set (10%)
    te_preds = best_model.predict(X_te)
    te_errors = te_preds - y_te
    te_abs_errors = np.abs(te_errors)

    new_mae = float(mean_absolute_error(y_te, te_preds))
    new_rmse = float(np.sqrt(mean_squared_error(y_te, te_preds)))
    new_r2 = float(r2_score(y_te, te_preds))
    new_bias = float(np.mean(te_errors))
    new_within_10 = float(np.mean(te_abs_errors / y_te <= 0.10))
    new_within_15 = float(np.mean(te_abs_errors / y_te <= 0.15))

    # Check empirical coverage of calibrated interval on test set
    test_covered = np.mean((y_te >= (te_preds - q_85)) & (y_te <= (te_preds + q_85)))

    print("\n================================================================================")
    print("   EVALUATION RESULTS ON INDEPENDENT TEST SET (10% - 150 RECORDS)               ")
    print("================================================================================")
    print(f"{'Metric':<28} | {'Old Model':<18} | {'Improved Model':<18} | {'Improvement':<12}")
    print("-" * 80)
    if old_metrics:
        mae_imp = old_metrics['mae'] - new_mae
        rmse_imp = old_metrics['rmse'] - new_rmse
        bias_imp = abs(old_metrics['mean_bias_error']) - abs(new_bias)
        acc10_imp = (new_within_10 - old_metrics['within_10_pct']) * 100
        print(f"{'R2 Score':<28} | {old_metrics['r2_score']*100:16.2f}% | {new_r2*100:16.2f}% | {(new_r2 - old_metrics['r2_score'])*100:+5.2f}%")
        print(f"{'MAE (Mean Absolute Error)':<28} | INR {old_metrics['mae']:14,.2f} | INR {new_mae:14,.2f} | -INR {mae_imp:,.2f}")
        print(f"{'RMSE (Root Mean Sq Error)':<28} | INR {old_metrics['rmse']:14,.2f} | INR {new_rmse:14,.2f} | -INR {rmse_imp:,.2f}")
        print(f"{'Mean Bias (Overprediction)':<28} | INR {old_metrics['mean_bias_error']:+14,.2f} | INR {new_bias:+14,.2f} | -INR {bias_imp:,.2f}")
        print(f"{'Accuracy within +/-10%':<28} | {old_metrics['within_10_pct']*100:16.2f}% | {new_within_10*100:16.2f}% | {acc10_imp:+5.2f}%")
        print(f"{'Accuracy within +/-15%':<28} | {old_metrics['within_15_pct']*100:16.2f}% | {new_within_15*100:16.2f}% | {(new_within_15 - old_metrics['within_15_pct'])*100:+5.2f}%")
    else:
        print(f"Improved Model Test MAE:  INR {new_mae:,.2f}")
        print(f"Improved Model Test RMSE: INR {new_rmse:,.2f}")
        print(f"Improved Model Test R2:   {new_r2*100:.2f}%")

    print(f"\n[Calibrated Prediction Interval Coverage on Test Set]: {test_covered*100:.2f}% (Target: 85%)")

    # 9. Segmented Analysis by Species, Breed, and District
    df_te_eval = df_te.copy()
    df_te_eval['actual_price'] = y_te
    df_te_eval['predicted_price'] = te_preds
    df_te_eval['abs_error'] = te_abs_errors
    df_te_eval['pct_error'] = (te_abs_errors / y_te) * 100

    print("\n--- Segmented Evaluation by Species ---")
    species_eval = df_te_eval.groupby('species').agg(
        count=('actual_price', 'count'),
        mean_actual=('actual_price', 'mean'),
        mean_pred=('predicted_price', 'mean'),
        mae=('abs_error', 'mean'),
        mean_pct_err=('pct_error', 'mean')
    ).round(1)
    print(species_eval)

    print("\n--- Segmented Evaluation by Key Breeds ---")
    top_breeds = df_te_eval['breed'].value_counts().head(8).index
    breed_eval = df_te_eval[df_te_eval['breed'].isin(top_breeds)].groupby('breed').agg(
        count=('actual_price', 'count'),
        mean_actual=('actual_price', 'mean'),
        mean_pred=('predicted_price', 'mean'),
        mae=('abs_error', 'mean'),
        mean_pct_err=('pct_error', 'mean')
    ).round(1)
    print(breed_eval)

    # 10. Feature Importance Verification (Lactation vs Calf)
    print("\n================================================================================")
    print("   FEATURE IMPORTANCE ANALYSIS: LACTATION NUMBER VS CALF NUMBER                 ")
    print("================================================================================")
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_cols)
    lact_cols = [c for c in all_feature_cols if 'lactation_number' in c]
    calf_cols = [c for c in all_feature_cols if 'calf_number' in c]
    lact_imp_total = float(feat_imp[lact_cols].sum())
    calf_imp_total = float(feat_imp[calf_cols].sum())
    print(f"   Total Lactation Number Importance: {lact_imp_total*100:6.2f}%")
    print(f"   Total Calf Number Importance:      {calf_imp_total*100:6.2f}%")
    print(f"   Dominance Verification:            Lactation > Calf is {lact_imp_total > calf_imp_total}")

    # Top overall features
    print("\n   Top 10 Most Influential Features:")
    for f_name, f_val in feat_imp.sort_values(ascending=False).head(10).items():
        print(f"      - {f_name:32s}: {f_val*100:5.2f}%")

    # 11. Verification of Benchmark 30L Cow Case
    print("\n================================================================================")
    print("   CRITICAL BENCHMARK VERIFICATION: 30 LITRES/DAY DAIRY ANIMAL                  ")
    print("================================================================================")
    # Benchmark 1: 30L HF Cow in Mandya
    test_case_30l = {
        'species': 'Cow',
        'breed': 'Holstein Friesian cattle',
        'gender': 'Female',
        'health_score': 5,
        'age_years': 4.0,
        'milk_production': 30.0,
        'is_pregnant': 0,
        'pregnancy_month': 0,
        'lactation_number': '2nd',
        'calf_number': '2nd',
        'district': 'Mandya'
    }
    df_case = pd.DataFrame([test_case_30l])
    X_case_raw = pd.get_dummies(df_case[cat_cols], drop_first=False)
    for c in num_cols:
        X_case_raw[c] = df_case[c].values

    X_case = pd.DataFrame(0, index=[0], columns=all_feature_cols)
    for c in X_case_raw.columns:
        if c in X_case.columns:
            X_case[c] = X_case_raw[c]

    p_new_30l = float(best_model.predict(X_case)[0])
    p_new_30l = round(p_new_30l, -2)
    range_min = round(p_new_30l - q_85, -2)
    range_max = round(p_new_30l + q_85, -2)

    print(f"Animal Profile: 4-Yr-Old HF Cow yielding 30 L/day in Mandya (2nd Lactation, Health 5)")
    if old_metrics:
        X_case_old = pd.DataFrame(0, index=[0], columns=old_template_features)
        for c in X_case_raw.columns:
            if c in X_case_old.columns:
                X_case_old[c] = X_case_raw[c]
        p_old_30l = round(float(old_model.predict(X_case_old)[0]), -2)
        print(f"   Old Model Predicted Price:        INR {p_old_30l:,.2f}")
    print(f"   Improved Model Predicted Price:   INR {p_new_30l:,.2f}")
    print(f"   Statistically Calibrated Range:   INR {range_min:,.2f} to INR {range_max:,.2f}")

    # Benchmark 2: 4L Low-yield Cow (Target: ~INR 25,000 - 35,000, reference: ~30,000)
    test_case_4l = {
        'species': 'Cow',
        'breed': 'Hallikar',
        'gender': 'Female',
        'health_score': 4,
        'age_years': 4.0,
        'milk_production': 4.0,
        'is_pregnant': 0,
        'pregnancy_month': 0,
        'lactation_number': '2nd',
        'calf_number': '2nd',
        'district': 'Mandya'
    }
    df_case_4l = pd.DataFrame([test_case_4l])
    X_case_4l_raw = pd.get_dummies(df_case_4l[cat_cols], drop_first=False)
    for c in num_cols:
        X_case_4l_raw[c] = df_case_4l[c].values
    X_case_4l = pd.DataFrame(0, index=[0], columns=all_feature_cols)
    for c in X_case_4l_raw.columns:
        if c in X_case_4l.columns:
            X_case_4l[c] = X_case_4l_raw[c]
    p_4l = round(float(best_model.predict(X_case_4l)[0]), -2)
    print(f"\nAnimal Profile: 4-Yr-Old Hallikar Cow yielding 4 L/day (2nd Lactation, Health 4)")
    print(f"   Predicted Price: INR {p_4l:,.2f} (Target: ~INR 25,000 to 35,000)")

    # Benchmark 3: 5th Lactation Modest Yield Cow (Target: ~INR 15,000 - 25,000)
    test_case_5th = {
        'species': 'Cow',
        'breed': 'Hallikar',
        'gender': 'Female',
        'health_score': 3,
        'age_years': 7.5,
        'milk_production': 4.0,
        'is_pregnant': 0,
        'pregnancy_month': 0,
        'lactation_number': '5th',
        'calf_number': '5th',
        'district': 'Mandya'
    }
    df_case_5th = pd.DataFrame([test_case_5th])
    X_case_5th_raw = pd.get_dummies(df_case_5th[cat_cols], drop_first=False)
    for c in num_cols:
        X_case_5th_raw[c] = df_case_5th[c].values
    X_case_5th = pd.DataFrame(0, index=[0], columns=all_feature_cols)
    for c in X_case_5th_raw.columns:
        if c in X_case_5th.columns:
            X_case_5th[c] = X_case_5th_raw[c]
    p_5th = round(float(best_model.predict(X_case_5th)[0]), -2)
    print(f"\nAnimal Profile: 7.5-Yr-Old Hallikar Cow, 5th Lactation (4 L/day, Health 3)")
    print(f"   Predicted Price: INR {p_5th:,.2f} (Target: ~INR 15,000 to 25,000)")

    # Benchmark 4: User Screenshot Case (HF, Age 4.0, Milk 6.0, Lactation 5th, Chamarajanagar)
    test_case_user = {
        'species': 'Cow',
        'breed': 'Holstein Friesian cattle',
        'gender': 'Female',
        'health_score': 4,
        'age_years': 4.0,
        'milk_production': 6.0,
        'is_pregnant': 0,
        'pregnancy_month': 0,
        'lactation_number': '5th',
        'calf_number': '5th',
        'district': 'Chamarajanagar'
    }
    df_case_user = pd.DataFrame([test_case_user])
    X_case_user_raw = pd.get_dummies(df_case_user[cat_cols], drop_first=False)
    for c in num_cols:
        X_case_user_raw[c] = df_case_user[c].values
    X_case_user = pd.DataFrame(0, index=[0], columns=all_feature_cols)
    for c in X_case_user_raw.columns:
        if c in X_case_user.columns:
            X_case_user[c] = X_case_user_raw[c]
    p_user = round(float(best_model.predict(X_case_user)[0]), -2)
    print(f"\nAnimal Profile: User Screenshot Case - 4-Yr HF Cow, 5th Lactation, 6 L/day in Chamarajanagar")
    print(f"   Predicted Price: INR {p_user:,.2f} (Target: < INR 30,000, between 20k-25k, NOT 72,200!)")

    # Benchmark 5: 1st Lactation 30L Cow (Target: ~INR 1,25,000 - 1,35,000)
    test_case_1st_30l = {
        'species': 'Cow',
        'breed': 'Holstein Friesian cattle',
        'gender': 'Female',
        'health_score': 5,
        'age_years': 3.0,
        'milk_production': 30.0,
        'is_pregnant': 0,
        'pregnancy_month': 0,
        'lactation_number': '1st',
        'calf_number': '1st',
        'district': 'Mandya'
    }
    df_case_1st = pd.DataFrame([test_case_1st_30l])
    X_case_1st_raw = pd.get_dummies(df_case_1st[cat_cols], drop_first=False)
    for c in num_cols:
        X_case_1st_raw[c] = df_case_1st[c].values
    X_case_1st = pd.DataFrame(0, index=[0], columns=all_feature_cols)
    for c in X_case_1st_raw.columns:
        if c in X_case_1st.columns:
            X_case_1st[c] = X_case_1st_raw[c]
    p_1st = round(float(best_model.predict(X_case_1st)[0]), -2)
    print(f"\nAnimal Profile: 1st Lactation 30L HF Cow in Mandya")
    print(f"   Predicted Price: INR {p_1st:,.2f} (Target: Above 1 Lakh ~INR 1,25,000 - 1,35,000)")

    # Benchmark 6: 3rd Lactation 30L Cow (Target: ~INR 80,000 - 90,000, lower than 1st & 2nd)
    test_case_3rd_30l = {
        'species': 'Cow',
        'breed': 'Holstein Friesian cattle',
        'gender': 'Female',
        'health_score': 4,
        'age_years': 5.0,
        'milk_production': 30.0,
        'is_pregnant': 0,
        'pregnancy_month': 0,
        'lactation_number': '3rd',
        'calf_number': '3rd',
        'district': 'Mandya'
    }
    df_case_3rd = pd.DataFrame([test_case_3rd_30l])
    X_case_3rd_raw = pd.get_dummies(df_case_3rd[cat_cols], drop_first=False)
    for c in num_cols:
        X_case_3rd_raw[c] = df_case_3rd[c].values
    X_case_3rd = pd.DataFrame(0, index=[0], columns=all_feature_cols)
    for c in X_case_3rd_raw.columns:
        if c in X_case_3rd.columns:
            X_case_3rd[c] = X_case_3rd_raw[c]
    p_3rd = round(float(best_model.predict(X_case_3rd)[0]), -2)
    print(f"\nAnimal Profile: 3rd Lactation 30L HF Cow in Mandya")
    print(f"   Predicted Price: INR {p_3rd:,.2f} (Target: Reduced vs 1st/2nd ~INR 80,000 - 90,000)")

    # 11. Save Improved Model and Metadata
    with open(NEW_MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n[Artifacts] Saved improved model to: {NEW_MODEL_PATH}")

    with open(FEATURES_JSON, 'w', encoding='utf-8') as f:
        json.dump({'feature_names': all_feature_cols}, f, indent=2)
    print(f"[Artifacts] Saved {len(all_feature_cols)} feature names to: {FEATURES_JSON}")

    with open(CALIBRATION_JSON, 'w', encoding='utf-8') as f:
        json.dump(calibration_payload, f, indent=2)
    print(f"[Artifacts] Saved calibrated prediction intervals to: {CALIBRATION_JSON}")

    # Update metrics_report.json
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}
    else:
        metrics_data = {}

    metrics_data['price_prediction_model'] = {
        'dataset_source': 'karnataka_livestock_market_transactions.csv (Verified Transactions, SLUCC & Santhe Surveys)',
        'total_records': len(df),
        'train_split_pct': 80.0,
        'val_split_pct': 10.0,
        'test_split_pct': 10.0,
        'r2_score': round(new_r2, 4),
        'mae': round(new_mae, 2),
        'rmse': round(new_rmse, 2),
        'mean_bias_error': round(new_bias, 2),
        'accuracy_within_10_pct': round(new_within_10, 4),
        'accuracy_within_15_pct': round(new_within_15, 4),
        'calibration_half_width': round(q_85, 2),
        'test_coverage_85pct_interval': round(test_covered, 4)
    }
    with open(METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[Artifacts] Updated {METRICS_JSON}")

    # Save detailed evaluation breakdown
    eval_summary = {
        'old_model_metrics': old_metrics,
        'new_model_metrics': {
            'r2_score': new_r2,
            'mae': new_mae,
            'rmse': new_rmse,
            'mean_bias_error': new_bias,
            'accuracy_within_10pct': new_within_10,
            'accuracy_within_15pct': new_within_15,
            'test_interval_coverage': test_covered
        },
        'species_eval': species_eval.to_dict(),
        'benchmark_30l_test': {
            'predicted_price': p_new_30l,
            'range_min': range_min,
            'range_max': range_max
        }
    }
    with open(EVAL_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[Artifacts] Saved evaluation results to: {EVAL_RESULTS_JSON}")

    print("\n[Done] Improved price model training, calibration, and evaluation complete!")
    return best_model

if __name__ == '__main__':
    train_and_evaluate()
