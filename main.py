import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,r2_score, mean_squared_error, mean_absolute_error, roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import shap

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7

# function style_panel definition
def style_panel(ax, letter, desc=None, desc_offset=-26):
    ax.text(-0.02, 1.04, f'({letter})', transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')
    if desc:
        ax.annotate(desc, xy=(0.5, 0.0), xycoords='axes fraction',xytext=(0, desc_offset), textcoords='offset points', ha='center', va='top', fontsize=8, annotation_clip=False)

element_cols = ['Al', 'Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu']

atomic_radius = {'Al': 143, 'Ti': 145, 'V': 134, 'Cr': 128, 'Fe': 126, 'Co': 125, 'Ni': 124, 'Cu': 128}
electronegativity = {'Al': 1.61, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90}
VEC_dict = {'Al': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11}
mix_enthalpy = {
    ('Al', 'Ti'): -30, ('Al', 'V'): -16, ('Al', 'Cr'): -10, ('Al', 'Fe'): -11, ('Al', 'Co'): -19, ('Al', 'Ni'): -22, ('Al', 'Cu'): -1,
    ('Ti', 'V'): -2, ('Ti', 'Cr'): -7, ('Ti', 'Fe'): -17, ('Ti', 'Co'): -20, ('Ti', 'Ni'): -35, ('Ti', 'Cu'): -9,
    ('V', 'Cr'): -2, ('V', 'Fe'): -7, ('V', 'Co'): -14, ('V', 'Ni'): -18, ('V', 'Cu'): 5,
    ('Cr', 'Fe'): -1, ('Cr', 'Co'): -4, ('Cr', 'Ni'): -7, ('Cr', 'Cu'): 12,
    ('Fe', 'Co'): -1, ('Fe', 'Ni'): -2, ('Fe', 'Cu'): 13,
    ('Co', 'Ni'): 0, ('Co', 'Cu'): 6,
    ('Ni', 'Cu'): 4,
}
R = 8.314


def compute_physical_descriptors(comp_dict):    
    elements = [e for e, c in comp_dict.items() if c > 0]
    if not elements:
        return pd.Series({'delta': np.nan, 'delta_chi': np.nan, 'VEC': np.nan, 'delta_smix': np.nan, 'delta_hmix': np.nan})
    c = np.array([comp_dict[e] for e in elements])
    c = c / c.sum()
    r = np.array([atomic_radius[e] for e in elements])
    r_bar = np.sum(c * r)
    delta = np.sqrt(np.sum(c * (1 - r / r_bar) ** 2)) * 100
    chi = np.array([electronegativity[e] for e in elements])
    chi_bar = np.sum(c * chi)
    delta_chi = np.sqrt(np.sum(c * (chi - chi_bar) ** 2))
    vec = np.sum(c * np.array([VEC_dict[e] for e in elements]))
    delta_smix = -R * np.sum(c * np.log(c))
    delta_hmix = 0.0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            e1, e2 = elements[i], elements[j]
            key = (e1, e2) if (e1, e2) in mix_enthalpy else (e2, e1)
            if key in mix_enthalpy:
                delta_hmix += 4 * mix_enthalpy[key] * c[i] * c[j]
    return pd.Series({'delta': delta, 'delta_chi': delta_chi, 'VEC': vec, 'delta_smix': delta_smix, 'delta_hmix': delta_hmix})

def clean_dataset(df, name):
    n0 = len(df)
    df = df.dropna(subset=element_cols + ['y']).copy()
    df = df[(df[element_cols] >= 0).all(axis=1)]
    df = df[df[element_cols].sum(axis=1) > 0]
    df = df[np.isfinite(pd.to_numeric(df['y'], errors='coerce'))]
    removed = n0 - len(df)
    print(f"{name}: original records {n0} , {removed} incomplete records removed,  {len(df)} records remaining")
    return df.reset_index(drop=True)

def l1_normalize(df, name):
    row_sums = df[element_cols].sum(axis=1)
    if (row_sums <= 0).any():
        bad = int((row_sums <= 0).sum())
        raise ValueError(f"{name}:  {bad} rows with composition sum ≤ 0 exist. Please run clean_dataset first")
    df = df.copy()
    df[element_cols] = df[element_cols].div(row_sums, axis=0)
    print(f"{name}:L1 normalization completed, range of composition sum "
          f"{df[element_cols].sum(axis=1).min():.4f}-{df[element_cols].sum(axis=1).max():.4f}")
    return df


def compute_shap_thresholds(shap_values, features, feature_names, target_features=('Al', 'VEC', 'delta'), n_bins=20):
    thresholds = {}
    for feat in target_features:
        if feat not in feature_names:
            continue
        idx = feature_names.index(feat)
        x = np.asarray(features[:, idx], dtype=float)
        y = np.asarray(shap_values[:, idx], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 10:
            print(f"  [Threshold] {feat}: Number of valid samples {len(x)} < 10, skipped")
            continue
        bins = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
        if len(bins) < 4:
            print(f" [Threshold] {feat}: insufficient feature values, skipped")
            continue
        bin_id = np.clip(np.digitize(x, bins[1:-1]), 0, len(bins) - 2)
        xb, yb = [], []
        for b in range(len(bins) - 1):
            m = bin_id == b
            if m.sum() >= 1:
                xb.append(float(x[m].mean()))
                yb.append(float(y[m].mean()))
        xb, yb = np.array(xb), np.array(yb)
        yb_s = gaussian_filter1d(yb, sigma=1) if len(xb) >= 5 else yb
        # elbow point
        if len(xb) >= 7:
            with np.errstate(divide='ignore', invalid='ignore'):
                dy = np.gradient(yb_s, xb)
                d2y = np.gradient(dy, xb)
            core = np.abs(d2y[2:-2])
            core_ok = np.isfinite(core)
            if core_ok.any():
                inflection = float(xb[np.argmax(np.where(core_ok, core, -np.inf)) + 2])
            else:
                inflection = float(np.median(xb))
        else:
            inflection = float(np.median(xb))
        # peak value
        peak = float(xb[np.argmax(yb_s)])
        # saturation point
        with np.errstate(divide='ignore', invalid='ignore'):
            dy = np.gradient(yb_s, xb)
        dy = np.where(np.isfinite(dy), dy, 0.0)
        dy_max = np.max(np.abs(dy))
        if dy_max > 0:
            sat_mask = np.abs(dy) < 0.1 * dy_max
            saturation = float(xb[np.where(sat_mask)[0][0]]) if sat_mask.any() else inflection
        else:
            saturation = inflection
        # 80% maximum range
        ymax = float(np.max(yb_s))
        if ymax > 0:
            m80 = yb_s >= 0.8 * ymax
            region80 = (float(xb[m80].min()), float(xb[m80].max()))
        else:
            region80 = (float(xb.min()), float(xb.max()))
        thresholds[feat] = {'inflection': inflection, 'saturation': saturation, 'peak': peak,
                            'region80': region80,
                            'x_range': (float(xb.min()), float(xb.max()))}
    return thresholds


def stratified_reg_split(X, y, test_size, seed):
    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
        counts = pd.Series(bins).value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=bins)
    except Exception:
        pass
    print(" Hardness quantile stratification failed, falling back to simple random splitting")
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def fmt_ms(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


# ============================================================
# Step 1: Load data, clean, normalize composition, compute physical descriptors, overlap check
# ============================================================
print("=" * 60)
print("Step 1: Load data, clean, normalize composition, compute physical descriptors, overlap check")
print("=" * 60)

for csv in ('phase_data.csv', 'hardness.csv'):
    if not os.path.exists(csv):
        raise FileNotFoundError(f" {csv} target file not found")

df_phase = pd.read_csv('data/phase_data.csv')
df_hard = pd.read_csv('data/hardness.csv')
print(f"Original phase classification data: {len(df_phase)} records")
print(f"Original hardness data: {len(df_hard)} records")


df_phase = clean_dataset(df_phase, 'Phase classification data')
df_hard = clean_dataset(df_hard, 'hardness data')


df_phase = l1_normalize(df_phase, 'Phase classification data')
df_hard = l1_normalize(df_hard, 'hardness data')

n_before_dedup = len(df_phase)
comp_counts = df_phase.groupby(element_cols + ['y']).size().reset_index(name='cnt')
conflict_comps = df_phase.groupby(element_cols)['y'].nunique()
conflict_keys = conflict_comps[conflict_comps > 1].index
n_conflict_rows = 0
if len(conflict_keys) > 0:
    is_conflict = df_phase.set_index(element_cols).index.isin(conflict_keys)
    n_conflict_rows = int(is_conflict.sum())
    df_phase = df_phase[~is_conflict].reset_index(drop=True)
df_phase = df_phase.drop_duplicates().reset_index(drop=True)
df_hard = df_hard.drop_duplicates().reset_index(drop=True)
print(f"Deduplication and conflict removal: original {n_before_dedup} records -> {n_conflict_rows} records with conflicting labels removed -> "
      f"{n_before_dedup - n_conflict_rows - len(df_phase)} duplicate records removed -> {len(df_phase)} records remaining")
print(f"Deduplicated phase classification data: {len(df_phase)} records (BCC {int((df_phase['y'] == 1).sum())} / "
      f"non-BCC {int((df_phase['y'] == 0).sum())})")
print(f"Deduplicated hardness data: {len(df_hard)} records, hardness range {df_hard['y'].min():.0f}-{df_hard['y'].max():.0f} HV")


q1, q3 = df_hard['y'].quantile(0.25), df_hard['y'].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df_hard[(df_hard['y'] < lower) | (df_hard['y'] > upper)]
print(f"Hardness IQR outlier range: {lower:.1f} - {upper:.1f}, number of outliers: {len(outliers)}")

# Calculate physical descriptors
print("\n Calculate physical descriptors")
for df in (df_phase, df_hard):
    phys_desc = df.apply(lambda row: compute_physical_descriptors(
        {e: row[e] for e in element_cols}), axis=1)
    for col in phys_desc.columns:
        df[col] = phys_desc[col]

feature_cols = element_cols + ['delta', 'delta_chi', 'VEC', 'delta_smix', 'delta_hmix']
print(f"Feature dimension: {len(feature_cols)} (8 elements + 5 physical descriptors)")

phase_keys = set(df_phase[element_cols].round(4).apply(tuple, axis=1))
hard_keys = df_hard[element_cols].round(4).apply(tuple, axis=1)
n_mapped = int(hard_keys.isin(phase_keys).sum())
overlap = len(set(hard_keys) & phase_keys)
print(f"Number of overlapping composition samples between the two datasets: {len(set(hard_keys) & phase_keys)}")
print(f"Map hardness samples to the phase dataset: {n_mapped}/{len(hard_keys)}")
full_overlap = (n_mapped == len(hard_keys))
if not full_overlap:
    print("'every hardness sample maps compositionally onto the phase dataset' ")

X_p = df_phase[feature_cols].values
y_p = df_phase['y'].values
X_h = df_hard[feature_cols].values
y_h = df_hard['y'].values

# ============================================================
# Step 2: Train the main model
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Train the main model")
print("=" * 60)

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_p, y_p, test_size=0.3, random_state=42, stratify=y_p)
Xh_train, Xh_test, yh_train, yh_test = stratified_reg_split(X_h, y_h, 0.3, 42)
print(f"Train/test split: classification {len(Xp_train)}/{len(Xp_test)}, regression {len(Xh_train)}/{len(Xh_test)}")
print(f"Test set hardness range: {yh_test.min():.0f} - {yh_test.max():.0f} HV")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------- classification model ----------
print("\n--- classification model ---")
baseline_cls = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
}
cls_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
}
cls_param_grids = {
    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 5, 6], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]},
    'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 6]},
}

cls_results = {}
for name, model in baseline_cls.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    cv_scores = cross_val_score(pipe, X_p, y_p, cv=skf, scoring='accuracy', n_jobs=-1)
    pipe.fit(Xp_train, yp_train)
    yp_pred = pipe.predict(Xp_test)
    acc = accuracy_score(yp_test, yp_pred)
    auc = roc_auc_score(yp_test, pipe.predict_proba(Xp_test)[:, 1])
    cls_results[name] = {'accuracy': acc, 'auc': auc, 'is_baseline': True,
                         'cv_accuracy': float(np.mean(cv_scores)), 'cv_std': float(np.std(cv_scores)),
                         'model': pipe}
    print(f"{name:20s} | Acc={acc:.4f} | AUC={auc:.4f} | "
          f"5-fold CV={fmt_ms(np.mean(cv_scores), np.std(cv_scores))}")

for name, model in cls_models.items():
    grid = GridSearchCV(model, cls_param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
    grid.fit(Xp_train, yp_train)
    best_model = grid.best_estimator_
    yp_pred = best_model.predict(Xp_test)
    yp_prob = best_model.predict_proba(Xp_test)[:, 1]
    acc = accuracy_score(yp_test, yp_pred)
    auc = roc_auc_score(yp_test, yp_prob)
    cv_mean = float(grid.best_score_)
    cv_std = float(grid.cv_results_['std_test_score'][grid.best_index_])
    cls_results[name] = {'accuracy': acc, 'auc': auc, 'is_baseline': False,
                         'cv_accuracy': cv_mean, 'cv_std': cv_std,
                         'model': best_model, 'best_params': grid.best_params_, 'y_prob': yp_prob}
    print(f"{name:20s} | Acc={acc:.4f} | AUC={auc:.4f} | CV={fmt_ms(cv_mean, cv_std)} | {grid.best_params_}")

tree_cls_names = [k for k in cls_models]
best_cls_name = max(tree_cls_names, key=lambda k: cls_results[k]['cv_accuracy'])
best_cls = cls_results[best_cls_name]['model']
print(f"\n the best classification model): {best_cls_name} "
      f"(CV={cls_results[best_cls_name]['cv_accuracy']:.4f}, "
      f"Acc={cls_results[best_cls_name]['accuracy']:.4f}, "
      f"AUC={cls_results[best_cls_name]['auc']:.4f})")
if 'SVM' in cls_results and cls_results['SVM']['accuracy'] > cls_results[best_cls_name]['accuracy']:
    print(f"[Info] SVM single split Acc={cls_results['SVM']['accuracy']:.4f} higher than {best_cls_name}; "
          f"Keep the tree model: TreeExplainer SHAP ")
print("\nClassification report:")
print(classification_report(yp_test, best_cls.predict(Xp_test), target_names=['FCC/Other', 'BCC']))
cm = confusion_matrix(yp_test, best_cls.predict(Xp_test))
print(f"Confusion matrix :\n{cm}")

# ---------- regression model ----------
print("\n--- regression model ---")
baseline_reg = {
    'Linear Regression': LinearRegression(),
}
svr_baseline_grid = {
    'model__kernel': ['rbf', 'linear'],
    'model__C': [0.1, 1, 10, 100, 1000],
    'model__gamma': ['scale', 0.001, 0.01, 0.1, 1.0],
    'model__epsilon': [0.1, 1, 5, 10],
}
reg_models = {
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
}
reg_param_grids = {
    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]},
    'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 6]},
}

reg_results = {}
for name, model in baseline_reg.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    cv_scores = cross_val_score(pipe, X_h, y_h, cv=kf, scoring='r2', n_jobs=-1)
    pipe.fit(Xh_train, yh_train)
    yh_pred = pipe.predict(Xh_test)
    r2 = r2_score(yh_test, yh_pred)
    rmse = np.sqrt(mean_squared_error(yh_test, yh_pred))
    mae = mean_absolute_error(yh_test, yh_pred)
    reg_results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'is_baseline': True,
                         'cv_r2': float(np.mean(cv_scores)), 'cv_std': float(np.std(cv_scores)),
                         'model': pipe}
    print(f"{name:20s} | R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f} | "
          f"5-fold CV-R²={fmt_ms(np.mean(cv_scores), np.std(cv_scores))}")

svr_pipe = Pipeline([('scaler', StandardScaler()), ('model', SVR())])
svr_grid = GridSearchCV(svr_pipe, svr_baseline_grid, cv=kf, scoring='r2', n_jobs=-1)
svr_grid.fit(Xh_train, yh_train)
svr_pred = svr_grid.predict(Xh_test)
svr_cv_std = float(svr_grid.cv_results_['std_test_score'][svr_grid.best_index_])
reg_results['SVR'] = {
    'R2': r2_score(yh_test, svr_pred),
    'RMSE': np.sqrt(mean_squared_error(yh_test, svr_pred)),
    'MAE': mean_absolute_error(yh_test, svr_pred),
    'is_baseline': True,
    'cv_r2': float(svr_grid.best_score_), 'cv_std': svr_cv_std,
    'model': svr_grid.best_estimator_, 'best_params': svr_grid.best_params_,
}
print(f"{'SVR (tuned)':20s} | R²={reg_results['SVR']['R2']:.4f} | "
      f"RMSE={reg_results['SVR']['RMSE']:.2f} | MAE={reg_results['SVR']['MAE']:.2f} | "
      f"5-fold CV-R²={fmt_ms(reg_results['SVR']['cv_r2'], svr_cv_std)} | {svr_grid.best_params_}")

for name, model in reg_models.items():
    grid = GridSearchCV(model, reg_param_grids[name], cv=kf, scoring='r2', n_jobs=-1)
    grid.fit(Xh_train, yh_train)
    best_model = grid.best_estimator_
    yh_pred = best_model.predict(Xh_test)
    r2 = r2_score(yh_test, yh_pred)
    rmse = np.sqrt(mean_squared_error(yh_test, yh_pred))
    mae = mean_absolute_error(yh_test, yh_pred)
    cv_r2 = float(grid.best_score_)
    cv_std = float(grid.cv_results_['std_test_score'][grid.best_index_])
    reg_results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'is_baseline': False,
                         'cv_r2': cv_r2, 'cv_std': cv_std,
                         'model': best_model, 'best_params': grid.best_params_, 'y_pred': yh_pred}
    print(f"{name:20s} | R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f} | CV-R²={fmt_ms(cv_r2, cv_std)} | {grid.best_params_}")

best_reg_name = max(reg_models, key=lambda k: reg_results[k]['cv_r2'])
best_reg = reg_results[best_reg_name]['model']
yh_pred_test = reg_results[best_reg_name]['y_pred']
print(f"\n Best regression model (selected by 5-fold CV R²): {best_reg_name} (CV-R²={reg_results[best_reg_name]['cv_r2']:.4f}, "
      f"R²={reg_results[best_reg_name]['R2']:.4f})")

residuals = yh_test - yh_pred_test
print(f" Residual statistics: mean={residuals.mean():.2f}, std={residuals.std():.2f} ")
high_hard_mask = yh_test > 500
if high_hard_mask.sum() >= 2:
    high_rmse = float(np.sqrt(mean_squared_error(yh_test[high_hard_mask], yh_pred_test[high_hard_mask])))
    print(f"Number of samples in the high-hardness region (>500 HV): {int(high_hard_mask.sum())}, RMSE={high_rmse:.2f}")
else:
    high_rmse = None
    print(f"Number of samples in the high-hardness region (>500 HV): {int(high_hard_mask.sum())} (<2), RMSE Not computed")

# ============================================================
# Step 3: Repeated validation
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Repeated validation")
print("=" * 60)

n_repeats = 10
repeat_results = {'cls_acc': [], 'cls_auc': [], 'reg_r2': [], 'reg_rmse': []}
cls_best_params = {k: v for k, v in cls_results['XGBoost']['best_params'].items() if k != 'random_state'}
reg_best_params = {k: v for k, v in reg_results['XGBoost']['best_params'].items() if k != 'random_state'}

for rep in range(n_repeats):
    seed = 42 + rep
    # Classification (XGBoost with tuned hyperparameters)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(X_p, y_p, test_size=0.3, random_state=seed, stratify=y_p)
    cls_model = XGBClassifier(random_state=seed, eval_metric='logloss', **cls_best_params)
    cls_model.fit(Xp_tr, yp_tr)
    yp_pred = cls_model.predict(Xp_te)
    yp_prob = cls_model.predict_proba(Xp_te)[:, 1]
    repeat_results['cls_acc'].append(accuracy_score(yp_te, yp_pred))
    repeat_results['cls_auc'].append(roc_auc_score(yp_te, yp_prob))
    # Regression (XGBoost with tuned hyperparameters)
    Xh_tr, Xh_te, yh_tr, yh_te = stratified_reg_split(X_h, y_h, 0.3, seed)
    reg_model = XGBRegressor(random_state=seed, **reg_best_params)
    reg_model.fit(Xh_tr, yh_tr)
    yh_pred = reg_model.predict(Xh_te)
    repeat_results['reg_r2'].append(r2_score(yh_te, yh_pred))
    repeat_results['reg_rmse'].append(np.sqrt(mean_squared_error(yh_te, yh_pred)))

print(f"\n Repeated validation results (XGBoost, tuned hyperparameters = {reg_best_params}）:")
print(f"  Classification accuracy: {np.mean(repeat_results['cls_acc']):.3f} ± {np.std(repeat_results['cls_acc']):.3f}")
print(f"  ClassificationAUC: {np.mean(repeat_results['cls_auc']):.3f} ± {np.std(repeat_results['cls_auc']):.3f}")
print(f"  Regression R²: {np.mean(repeat_results['reg_r2']):.3f} ± {np.std(repeat_results['reg_r2']):.3f} "
      f"( mean test R² ± std)")
print(f"  Classification RMSE: {np.mean(repeat_results['reg_rmse']):.1f} ± {np.std(repeat_results['reg_rmse']):.1f} HV")

# ============================================================
# Step 4: Dynamic sample counts across dataset splits
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Fig1")
print("=" * 60)
n_phase, n_hard = len(df_phase), len(df_hard)
fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.4))
axes[0, 0].bar(['BCC (y=1)', 'FCC/Other (y=0)'], [int((df_phase['y'] == 1).sum()), int((df_phase['y'] == 0).sum())],
               color=['#4C72B0', '#DD8452'], edgecolor='black')
axes[0, 0].set_ylabel('Count')
style_panel(axes[0, 0], 'a', f'Phase distribution (n={n_phase})')
axes[0, 1].hist(df_hard['y'], bins=20, color='#55A868', edgecolor='black', alpha=0.8)
axes[0, 1].set_xlabel('Vickers Hardness (HV)')
axes[0, 1].set_ylabel('Count')
style_panel(axes[0, 1], 'b', f'Hardness distribution (n={n_hard})')
sc = axes[0, 2].scatter(df_hard['VEC'], df_hard['y'], c=df_hard['delta'], cmap='viridis', s=12, alpha=0.7)
axes[0, 2].set_xlabel('VEC')
axes[0, 2].set_ylabel('Hardness (HV)')
style_panel(axes[0, 2], 'c', 'VEC vs hardness (color: δ)')
plt.colorbar(sc, ax=axes[0, 2], label='δ (%)')
sc2 = axes[1, 0].scatter(df_hard['delta_hmix'], df_hard['y'], c=df_hard['Al'], cmap='RdYlBu_r', s=12, alpha=0.7)
axes[1, 0].set_xlabel('ΔHmix (kJ/mol)')
axes[1, 0].set_ylabel('Hardness (HV)')
style_panel(axes[1, 0], 'd', 'ΔHmix vs hardness (color: Al)')
plt.colorbar(sc2, ax=axes[1, 0], label='Al fraction')
df_melt = df_hard[['delta', 'delta_chi', 'VEC', 'delta_smix', 'delta_hmix']].melt(var_name='Descriptor', value_name='Value')
sns.boxplot(x='Descriptor', y='Value', data=df_melt, ax=axes[1, 1], palette='Set2')
style_panel(axes[1, 1], 'e', 'Physical descriptors', desc_offset=-32)
axes[1, 1].tick_params(axis='x', rotation=45)
bcc_vec = df_phase[df_phase['y'] == 1]['VEC']
fcc_vec = df_phase[df_phase['y'] == 0]['VEC']
axes[1, 2].hist([bcc_vec, fcc_vec], bins=15, label=['BCC', 'FCC/Other'], color=['#4C72B0', '#DD8452'],
                alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('VEC')
axes[1, 2].set_ylabel('Count')
style_panel(axes[1, 2], 'f', 'VEC distribution by phase')
axes[1, 2].legend(loc='upper left', fontsize=7)
plt.tight_layout(h_pad=1.8, w_pad=0.6)
plt.savefig('Fig1_dataset_distribution.png')
plt.savefig('Fig1_dataset_distribution.pdf')
plt.savefig('Fig1_dataset_distribution.eps')
plt.close()
print(f"Fig1 saved (n={n_phase}/{n_hard})")

# ============================================================
# Step 5: Model performance
# ============================================================
print("\nStep 5: Fig2")
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.1))
model_names = list(cls_models.keys())
x = np.arange(len(model_names))
w = 0.35
accs = [cls_results[m]['accuracy'] for m in model_names]
aucs = [cls_results[m]['auc'] for m in model_names]
axes[0, 0].bar(x - w / 2, accs, w, label='Accuracy', color='#4C72B0', edgecolor='black')
axes[0, 0].bar(x + w / 2, aucs, w, label='ROC-AUC', color='#DD8452', edgecolor='black')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names, fontsize=7)
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_ylim(0.6, 1.0)
style_panel(axes[0, 0], 'a', 'Classification: accuracy & ROC-AUC', desc_offset=-24)
axes[0, 0].legend(loc='lower right', fontsize=6.5)
for i, (a, c) in enumerate(zip(accs, aucs)):
    axes[0, 0].text(i - w / 2, a + 0.005, f'{a:.3f}', ha='center', fontsize=6.5)
    axes[0, 0].text(i + w / 2, c + 0.005, f'{c:.3f}', ha='center', fontsize=6.5)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 7}, ax=axes[0, 1],
            xticklabels=['FCC/Other', 'BCC'], yticklabels=['FCC/Other', 'BCC'])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')
style_panel(axes[0, 1], 'b', f'Confusion matrix ({best_cls_name})', desc_offset=-28)

reg_model_names = list(reg_results.keys())
rx = np.arange(len(reg_model_names))
r2s = [reg_results[m]['R2'] for m in reg_model_names]
rmses = [reg_results[m]['RMSE'] for m in reg_model_names]
ax2 = axes[1, 0].twinx()
axes[1, 0].bar(rx - w / 2, r2s, w, label='R²', color='#55A868', edgecolor='black')
ax2.plot(rx + w / 2, rmses, 'o-', color='#C44E52', label='RMSE', markersize=4)
axes[1, 0].set_xticks(rx)
axes[1, 0].set_xticklabels(reg_model_names, fontsize=7, rotation=35, ha='right')
axes[1, 0].set_ylabel('R²', color='#55A868')
ax2.set_ylabel('RMSE (HV)', color='#C44E52')
axes[1, 0].set_ylim(0, 1.32)
ax2.set_ylim(40, 100)
style_panel(axes[1, 0], 'c', 'Regression: R² & RMSE', desc_offset=-46)
for i, r in enumerate(r2s):
    axes[1, 0].text(rx[i] - w / 2, r + 0.02, f'{r:.3f}', ha='center', fontsize=5.5)
lines1, labels1 = axes[1, 0].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1, 0].legend(lines1 + lines2, labels1 + labels2, fontsize=6.5,
                  loc='upper center', ncol=2, framealpha=0.95)

yh_pred_train = best_reg.predict(Xh_train)
axes[1, 1].scatter(yh_pred_train, yh_train - yh_pred_train, alpha=0.5, s=12, c='#4C72B0', label='Training')
axes[1, 1].scatter(yh_pred_test, residuals, alpha=0.7, s=16, c='#DD8452', marker='^', label='Test')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=1)
axes[1, 1].set_xlabel('Predicted Hardness (HV)')
axes[1, 1].set_ylabel('Residual (HV)')
style_panel(axes[1, 1], 'd', f'Residuals ({best_reg_name})')
axes[1, 1].legend()
plt.tight_layout(h_pad=1.8, w_pad=0.6)
plt.savefig('Fig2_model_performance.png')
plt.savefig('Fig2_model_performance.pdf')
plt.savefig('Fig2_model_performance.eps')
plt.close()
print("Fig2 saved")

# ============================================================
# Step 6: SHAP analysis , interaction analysis , threshold extraction
# ============================================================
print("\n" + "=" * 60)
print("Step 6: SHAP analysis , interaction analysis , threshold extraction")
print("=" * 60)
explainer_cls = shap.TreeExplainer(best_cls)
shap_cls_raw = explainer_cls.shap_values(Xp_test)
if isinstance(shap_cls_raw, list):
    shap_cls = shap_cls_raw[1]
elif hasattr(shap_cls_raw, 'ndim') and shap_cls_raw.ndim == 3:
    shap_cls = shap_cls_raw[:, :, 1]
else:
    shap_cls = shap_cls_raw

explainer_reg = shap.TreeExplainer(best_reg)
shap_reg = explainer_reg.shap_values(Xh_test)


SHORT = {'delta': 'δ', 'delta_chi': 'Δχ', 'delta_smix': 'ΔS_mix', 'delta_hmix': 'ΔH_mix'}

def _short_names(ax, axis='y', fontsize=7):
    getter = ax.get_yticklabels if axis == 'y' else ax.get_xticklabels
    setter = ax.set_yticklabels if axis == 'y' else ax.set_xticklabels
    setter([SHORT.get(t.get_text(), t.get_text()) for t in getter()], fontsize=fontsize)

def _polish_shap(ax, xlabel):
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=6.5)
    _short_names(ax, 'y', 7)

fig, axes = plt.subplots(3, 2, figsize=(6.5, 8.0))
fig3, fig3_axes = fig, axes
plt.sca(axes[0, 0])
shap.summary_plot(shap_cls, Xp_test, feature_names=feature_cols, plot_type="dot", show=False, max_display=13, plot_size=None)
_polish_shap(axes[0, 0], 'SHAP value (impact on model output)')
style_panel(axes[0, 0], 'a', 'SHAP summary: phase (BCC)', desc_offset=-24)
plt.sca(axes[0, 1])
shap.summary_plot(shap_cls, Xp_test, feature_names=feature_cols, plot_type="bar", show=False, max_display=13, plot_size=None)
_polish_shap(axes[0, 1], 'Mean |SHAP value|')
style_panel(axes[0, 1], 'b', 'Mean |SHAP|: phase', desc_offset=-24)
plt.sca(axes[1, 0])
shap.summary_plot(shap_reg, Xh_test, feature_names=feature_cols, plot_type="dot", show=False, max_display=13, plot_size=None)
_polish_shap(axes[1, 0], 'SHAP value (impact on model output)')
style_panel(axes[1, 0], 'c', 'SHAP summary: hardness', desc_offset=-24)
plt.sca(axes[1, 1])
shap.summary_plot(shap_reg, Xh_test, feature_names=feature_cols, plot_type="bar", show=False, max_display=13, plot_size=None)
_polish_shap(axes[1, 1], 'Mean |SHAP value|')
style_panel(axes[1, 1], 'd', 'Mean |SHAP|: hardness', desc_offset=-24)
print("Fig3 have been ploted ")

# SHAP dependency graph
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.1))
plt.sca(axes[0, 0])
shap.dependence_plot('Al', shap_reg, Xh_test, feature_names=feature_cols, ax=axes[0, 0], show=False)
style_panel(axes[0, 0], 'a', 'SHAP dependence: Al (hardness)', desc_offset=-28)
plt.sca(axes[0, 1])
shap.dependence_plot('VEC', shap_reg, Xh_test, feature_names=feature_cols, ax=axes[0, 1], show=False)
style_panel(axes[0, 1], 'b', 'SHAP dependence: VEC (hardness)', desc_offset=-28)
plt.sca(axes[1, 0])
shap.dependence_plot('delta', shap_reg, Xh_test, feature_names=feature_cols, ax=axes[1, 0], show=False)
axes[1, 0].set_xlabel('δ', fontsize=8)
style_panel(axes[1, 0], 'c', 'SHAP dependence: δ (hardness)', desc_offset=-28)
plt.sca(axes[1, 1])
shap.dependence_plot('Al', shap_cls, Xp_test, feature_names=feature_cols, ax=axes[1, 1], show=False)
style_panel(axes[1, 1], 'd', 'SHAP dependence: Al (phase)', desc_offset=-28)
plt.tight_layout(h_pad=1.8, w_pad=0.6)
plt.savefig('Fig3b_SHAP_dependence.png')
plt.savefig('Fig3b_SHAP_dependence.pdf')
plt.savefig('Fig3b_SHAP_dependence.eps')
plt.close()
print("SHAP dependency graph saved")

print("\n--- SHAP interaction analysis ---")
interaction_summary = {}
try:
    inter_cls_raw = explainer_cls.shap_interaction_values(Xp_test)
    if isinstance(inter_cls_raw, list):
        inter_cls = inter_cls_raw[1]
    else:
        inter_cls = inter_cls_raw
    inter_reg = explainer_reg.shap_interaction_values(Xh_test)


    for ax, inter, letter, desc in zip([fig3_axes[2, 0], fig3_axes[2, 1]], [inter_cls, inter_reg],
                                       ['e', 'f'],
                                       ['Mean |SHAP interaction|: phase (BCC)',
                                        'Mean |SHAP interaction|: hardness']):

        inter_mean = np.abs(inter).mean(axis=0)
        sns.heatmap(inter_mean, annot=False, cmap='viridis',
                    xticklabels=feature_cols, yticklabels=feature_cols, ax=ax,
                    cbar_kws={'shrink': 0.85})
        ax.tick_params(axis='x', rotation=45)
        _short_names(ax, 'x', 7)
        _short_names(ax, 'y', 7)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        style_panel(ax, letter, desc, desc_offset=-32)

        pairs = [(feature_cols[i], feature_cols[j], inter_mean[i, j])
                 for i in range(len(feature_cols)) for j in range(i + 1, len(feature_cols))]
        pairs.sort(key=lambda t: -t[2])
        task = 'Phase' if letter == 'e' else 'Hardness'
        interaction_summary[task] = {'top_pairs': pairs[:5],
                                     'matrix': inter_mean}
    fig3.tight_layout(h_pad=2.4, w_pad=1.2)
    fig3.savefig('Fig3_SHAP_analysis.png')
    fig3.savefig('Fig3_SHAP_analysis.pdf')
    fig3.savefig('Fig3_SHAP_analysis.eps')
    plt.close(fig3)
    print("Fig3 saved")
    for task, s in interaction_summary.items():
        print(f"{task} interaction Top5:")
        for f1, f2, v in s['top_pairs']:
            print(f"  {f1}–{f2}: {v:.3f}")
    for task, s in interaction_summary.items():
        m = s['matrix']
        al_v = m[feature_cols.index('Al'), feature_cols.index('V')]
        print(f"{task} Al–V mean |interaction| = {al_v:.3f} (quantitative basis for 'Al–V synergy')")
except Exception as e:
    print(f"️ SHAP interaction compute failed: {e}")

    try:
        fig3.tight_layout(h_pad=2.4, w_pad=1.2)
        fig3.savefig('Fig3_SHAP_analysis.png')
        fig3.savefig('Fig3_SHAP_analysis.pdf')
        fig3.savefig('Fig3_SHAP_analysis.eps')
        plt.close(fig3)
        print("Fig3 saved")
    except Exception as e2:
        print(f" Fig3 save failed: {e2}")


print("\n--- SHAP Threshold extraction---")
cls_thresholds = compute_shap_thresholds(shap_cls, Xp_test, feature_cols, ('Al', 'VEC', 'delta'))
reg_thresholds = compute_shap_thresholds(shap_reg, Xh_test, feature_cols, ('Al', 'VEC', 'delta'))
print(" Phase classification SHAP Threshold:")
for feat, vals in cls_thresholds.items():
    print(f"  {feat}: elbow point={vals['inflection']:.3f}, satuation point ={vals['saturation']:.3f}, "
          f"peak value ={vals['peak']:.3f}, 80%range ={vals['region80'][0]:.3f}-{vals['region80'][1]:.3f}, "
          f"range={vals['x_range'][0]:.3f}-{vals['x_range'][1]:.3f}")
print("SHAP threshold for hardness prediction:")
for feat, vals in reg_thresholds.items():
    print(f"  {feat}: elbow point ={vals['inflection']:.3f}, satuation point={vals['saturation']:.3f}, "
          f"peak value={vals['peak']:.3f}, 80%range={vals['region80'][0]:.3f}-{vals['region80'][1]:.3f}, "
          f"range={vals['x_range'][0]:.3f}-{vals['x_range'][1]:.3f}")


cls_imp = pd.DataFrame({'Feature': feature_cols, 'Mean_SHAP': np.abs(shap_cls).mean(axis=0)}).sort_values('Mean_SHAP', ascending=False)
reg_imp = pd.DataFrame({'Feature': feature_cols, 'Mean_SHAP': np.abs(shap_reg).mean(axis=0)}).sort_values('Mean_SHAP', ascending=False)
print("\nSHAP importance for phase classification (13 features):")
print(cls_imp.to_string(index=False))
print("\n SHAP importance for hardness prediction (13 features):")
print(reg_imp.to_string(index=False))

# ============================================================
# Step 7: Virtual screening
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Virtual screening")
print("=" * 60)
np.random.seed(123)
n_cand = 10000
cand_elem = np.zeros((n_cand, len(element_cols)))
for i, e in enumerate(element_cols):
    lo, hi = df_hard[e].min(), df_hard[e].max()
    if hi > lo:
        cand_elem[:, i] = np.random.uniform(lo, hi, n_cand)
    else:
        cand_elem[:, i] = lo
row_sums = cand_elem.sum(axis=1, keepdims=True)
if (row_sums <= 0).any():
    raise ValueError("Candidate composition contains all-zero rows")
cand_elem = cand_elem / row_sums

print("Compute physical descriptors for candidate compositions")
cand_phys_list = []
for j in range(n_cand):
    comp = {e: cand_elem[j, i] for i, e in enumerate(element_cols)}
    cand_phys_list.append(compute_physical_descriptors(comp).values)
cand_phys = np.array(cand_phys_list)
cand_all = np.hstack([cand_elem, cand_phys])


scaler_ad = StandardScaler()
Xh_train_scaled_ad = scaler_ad.fit_transform(Xh_train)
cand_scaled = scaler_ad.transform(cand_all)

tree = cKDTree(Xh_train_scaled_ad) 
d_train, _ = tree.query(Xh_train_scaled_ad, k=2)  
train_dists = d_train[:, 1]
ad_threshold = float(np.percentile(train_dists, 95))
min_distances, _ = tree.query(cand_scaled, k=1)
in_ad = min_distances <= ad_threshold
print(f"AD threshold: {ad_threshold:.3f}, AD internal candidates: {int(in_ad.sum())}/{n_cand}")

cand_in = cand_all[in_ad]
cand_phase_prob = best_cls.predict_proba(cand_in)[:, 1]
cand_hardness = best_reg.predict(cand_in)
cand_df = pd.DataFrame(cand_in, columns=feature_cols)
cand_df['BCC_prob'] = cand_phase_prob
cand_df['Predicted_HV'] = cand_hardness
cand_df['score'] = cand_df['BCC_prob'] * cand_df['Predicted_HV']
top10 = cand_df.nlargest(10, 'score').reset_index(drop=True)
print("\nTop10 candidate:")
print(top10[element_cols + ['BCC_prob', 'Predicted_HV']].to_string(index=False))
top10.to_excel('virtual_screening_top10.xlsx', index=False)

# Check whether the top composition is in the training set
top_comp = tuple(round(top10.iloc[0][e], 4) for e in element_cols)
train_comps = set(tuple(round(x[i], 4) for i in range(len(element_cols))) for x in Xh_train)
in_train = top_comp in train_comps
print(f"\n Check whether the top composition is in the training set: {'Yes' if in_train else 'No'}")

fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.6))
sc = axes[0].scatter(cand_in[:, 0], cand_in[:, feature_cols.index('VEC')], c=cand_hardness,
                     cmap='RdYlGn_r', s=10, alpha=0.5)
axes[0].scatter(top10['Al'], top10['VEC'], c='blue', s=60, marker='*', edgecolors='black', linewidths=0.8, label='Top10')
axes[0].set_xlabel('Al molar fraction')
axes[0].set_ylabel('VEC')
style_panel(axes[0], 'a', 'Screening: Al vs VEC (within AD)')
axes[0].legend()
plt.colorbar(sc, ax=axes[0], label='Predicted HV')
sc3 = axes[1].scatter(cand_phase_prob, cand_hardness, c=cand_in[:, 0], cmap='viridis', s=10, alpha=0.5)
axes[1].scatter(top10['BCC_prob'], top10['Predicted_HV'], c='red', s=50, marker='*',
                edgecolors='black', label='Top10')
axes[1].set_xlabel('Predicted BCC probability')
axes[1].set_ylabel('Predicted Hardness (HV)')
style_panel(axes[1], 'b', 'BCC stability vs hardness (within AD)')
axes[1].legend()
plt.colorbar(sc3, ax=axes[1], label='Al fraction')
axes[2].barh(range(10), top10['Predicted_HV'].values[::-1], color=plt.cm.viridis(np.linspace(0.2, 0.8, 10)))
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([f'Cand-{i + 1}' for i in range(10)][::-1])
axes[2].set_xlabel('Predicted Hardness (HV)')
style_panel(axes[2], 'c', 'Top-10 candidates (within AD)')
axes[2].set_xlim(0, float(top10['Predicted_HV'].max()) * 1.15)
for i, v in enumerate(top10['Predicted_HV'].values[::-1]):
    axes[2].text(v + 2, i, f'{v:.0f}', va='center', fontsize=7)
plt.tight_layout(w_pad=0.8)
plt.savefig('Fig4_virtual_screening.png')
plt.savefig('Fig4_virtual_screening.pdf')
plt.savefig('Fig4_virtual_screening.eps')
plt.close()
print("Fig4 Saved")

# ============================================================
# Save all the results
# ============================================================
print("\n" + "=" * 60)
print("save results")
print("=" * 60)
with open('results_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== final results sunmmary (v5) ===\n\n")
    f.write(f"dataset: Phase classification: {n_phase} records (BCC {int((df_phase['y'] == 1).sum())} / "
            f"non-BCC {int((df_phase['y'] == 0).sum())}), Hardness: {n_hard} records "
            f"({df_hard['y'].min():.0f}-{df_hard['y'].max():.0f} HV)\n")
    f.write(f"Train/test: classification {len(Xp_train)}/{len(Xp_test)}, regression {len(Xh_train)}/{len(Xh_test)}\n")
    f.write(f"Map hardness sample compositions to the phase dataset: {n_mapped}/{n_hard} "
            f"({'Complete mapping' if full_overlap else 'Incomplete'})\n\n")
    f.write("=== Classification models (including baseline, 5-fold CV) ===\n")
    for name, res in cls_results.items():
        f.write(f"{name}: Acc={res['accuracy']:.4f}, AUC={res['auc']:.4f}, "
                f"5-fold CV={res['cv_accuracy']:.4f} ± {res['cv_std']:.4f}")
        if 'best_params' in res:
            f.write(f", best_params={res['best_params']}")
        f.write("\n")
    f.write(f"\n Best (selected by CV): {best_cls_name}\n")
    f.write(f"Confuse Matrix: {cm.tolist()}\n\n")
    f.write("=== Classification model (including baseline 5-fold CV) ===\n")
    for name, res in reg_results.items():
        f.write(f"{name}: R²={res['R2']:.4f}, RMSE={res['RMSE']:.2f}, MAE={res['MAE']:.2f}, "
                f"5-fold CV-R²={res['cv_r2']:.4f} ± {res['cv_std']:.4f}")
        if 'best_params' in res:
            f.write(f", best_params={res['best_params']}")
        f.write("\n")
    f.write(f"\n Best(selected by CV): {best_reg_name}\n")
    if high_rmse is not None:
        f.write(f"high-hardness region (>500HV) RMSE: {high_rmse:.2f} (sample number {int(high_hard_mask.sum())})\n")
    else:
        f.write("high-hardness region (>500HV) sample insufficient, RMSE uncompute\n")
    f.write("\n=== Repeated validation (XGBoost, Tuned hyperparameters) ===\n")
    f.write(f"Classification precision: {np.mean(repeat_results['cls_acc']):.3f} ± {np.std(repeat_results['cls_acc']):.3f}\n")
    f.write(f"Classification AUC: {np.mean(repeat_results['cls_auc']):.3f} ± {np.std(repeat_results['cls_auc']):.3f}\n")
    f.write(f"Regression R²: {np.mean(repeat_results['reg_r2']):.3f} ± {np.std(repeat_results['reg_r2']):.3f}\n")
    f.write(f"Regression RMSE: {np.mean(repeat_results['reg_rmse']):.1f} ± {np.std(repeat_results['reg_rmse']):.1f} HV\n\n")
    f.write("=== SHAP threshold ===\n")
    f.write("phase classification:\n")
    for feat, vals in cls_thresholds.items():
        f.write(f"  {feat}: elbow point ={vals['inflection']:.3f}, saturation point ={vals['saturation']:.3f}, "
                f"peak value={vals['peak']:.3f}, 80%range={vals['region80'][0]:.3f}-{vals['region80'][1]:.3f}\n")
    f.write("hardness prediction:\n")
    for feat, vals in reg_thresholds.items():
        f.write(f"  {feat}: elbow point={vals['inflection']:.3f}, saturation={vals['saturation']:.3f}, "
                f"peak value={vals['peak']:.3f}, 80%range={vals['region80'][0]:.3f}-{vals['region80'][1]:.3f}\n")
    if interaction_summary:
        f.write("\n=== SHAP interaction Top5 ===\n")
        for task, s in interaction_summary.items():
            f.write(f"{task}:\n")
            for f1, f2, v in s['top_pairs']:
                f.write(f"  {f1}–{f2}: {v:.3f}\n")
    f.write("\n=== Virtual screening ===\n")
    f.write(f"AD threshold: {ad_threshold:.3f}\n")
    f.write(f"AD inner candidate: {int(in_ad.sum())}/{n_cand}\n")
    f.write(f"Whether the top composition is in the training set: {'No（new component）' if not in_train else 'Yes'}\n")
    f.write("\nTop10 candidate:\n")
    f.write(top10[element_cols + ['BCC_prob', 'Predicted_HV']].to_string(index=False))

print("Results saved as results_summary.txt")
