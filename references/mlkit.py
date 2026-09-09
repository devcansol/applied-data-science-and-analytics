"""
MLKIT  —  the modelling helpers used on Days 3, 4 and 5.
Written so that the SAME code works on every team's dataset.
Paste this whole cell once, near the top of your notebook.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

MAX_ONEHOT = 25          # a category with more levels than this is dropped
MAX_NULL_FRAC = 0.5      # a column emptier than this is dropped


# ---------------------------------------------------------------- target
def make_target(df, num_col=None, target=None, task="auto", verbose=True):
    """
    Returns (y, task, derived_from, label).
    target=None  -> derive a yes/no target by splitting num_col at its median.
                    This works on ANY dataset that has one number in it.
    """
    derived_from = None
    if target is None:
        if num_col is None:
            raise ValueError("Give either target= or num_col= so a target can be derived.")
        s = pd.to_numeric(df[num_col], errors="coerce")
        cut = s.median()
        y = (s >= cut).astype("float")
        y[s.isna()] = np.nan
        task, derived_from = "classification", num_col
        label = f"is {num_col} at or above its median ({cut:,.2f})?"
    else:
        s = df[target]
        if pd.api.types.is_numeric_dtype(s):
            nunique = s.nunique(dropna=True)
            if task == "auto":
                task = "classification" if nunique <= 10 else "regression"
            y = pd.to_numeric(s, errors="coerce")
        else:
            task = "classification" if task == "auto" else task
            levels = sorted(s.dropna().astype(str).unique())
            if len(levels) == 2:
                mapping = {levels[0]: 0, levels[1]: 1}
                y = s.astype(str).map(mapping)
                y[s.isna()] = np.nan
                print(f"ENCODED  : {levels[0]} -> 0    {levels[1]} -> 1  "
                      f"(so \"positive\" below means {levels[1]})")
            else:
                y = s.astype(str)
        label = f"{target}"

    if verbose:
        print(f"TARGET   : {label}")
        print(f"TASK     : {task}")
        if derived_from:
            print(f"           (derived from {derived_from} - so {derived_from} MUST be excluded from the features)")
        if task == "classification":
            vc = pd.Series(y).value_counts(dropna=True)
            share = (vc / vc.sum() * 100).round(1)
            print(f"BALANCE  : " + "  ".join(f"{k}={v}%" for k, v in share.items()))
            if share.max() > 80:
                print("           WARNING - one class dominates. Accuracy will look great and mean nothing.")
    return y, task, derived_from, label


# ---------------------------------------------------------------- features
def make_features(df, y, derived_from=None, target=None, drop=None, verbose=True):
    """One-hot encodes, imputes, and removes anything that would leak or be useless."""
    drop = set(drop or [])
    if derived_from: drop.add(derived_from)
    if target:       drop.add(target)

    X = df.copy()
    n = len(X)
    removed = []

    for c in list(X.columns):
        if c in drop:
            removed.append((c, "excluded on purpose - it would leak the answer" if c == derived_from
                               else "this is the target")); X = X.drop(columns=c); continue
        s = X[c]
        if s.isnull().mean() > MAX_NULL_FRAC:
            removed.append((c, f"{s.isnull().mean()*100:.0f}% empty")); X = X.drop(columns=c); continue
        if s.nunique(dropna=True) <= 1:
            removed.append((c, "same value in every row")); X = X.drop(columns=c); continue
        if pd.api.types.is_datetime64_any_dtype(s):
            removed.append((c, "a date - engineer month/weekday from it first")); X = X.drop(columns=c); continue
        if not pd.api.types.is_numeric_dtype(s):
            u = s.nunique(dropna=True)
            if u == n:
                looks_dated = pd.to_datetime(s, errors="coerce", format="mixed").notna().mean() > 0.8
                removed.append((c, "a date - engineer month/weekday from it first" if looks_dated
                                   else "one value per row - an ID, not a feature"))
                X = X.drop(columns=c); continue
            if u > MAX_ONEHOT:
                removed.append((c, f"{u} categories - too many to one-hot")); X = X.drop(columns=c); continue
        else:
            if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == n and n > 10:
                removed.append((c, "one value per row - an ID, not a feature")); X = X.drop(columns=c); continue

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in num_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).str.strip().fillna("missing")

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

    keep = pd.Series(y).notna().values
    X, y = X.loc[keep], pd.Series(y).loc[keep]

    if verbose:
        print(f"\nFEATURES : {X.shape[1]} columns built from {len(num_cols)} numeric + {len(cat_cols)} categorical")
        print(f"ROWS     : {len(X)} usable")
        if removed:
            print("REMOVED  :")
            for c, why in removed[:12]:
                print(f"           {c:<24} {why}")
    return X, y


# ---------------------------------------------------------------- split
def split(X, y, task, test_size=0.25, seed=42):
    strat = y if (task == "classification" and pd.Series(y).value_counts().min() >= 2) else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    print(f"TRAIN    : {len(Xtr)} rows      TEST: {len(Xte)} rows  (the test rows are never used to fit)")
    return Xtr, Xte, ytr, yte


# ---------------------------------------------------------------- metrics
def score(y_true, y_pred, task, y_prob=None):
    if task == "regression":
        return {"MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "R2": r2_score(y_true, y_pred)}
    avg = "binary" if pd.Series(y_true).nunique() == 2 else "macro"
    out = {"Accuracy": accuracy_score(y_true, y_pred),
           "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
           "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
           "F1": f1_score(y_true, y_pred, average=avg, zero_division=0)}
    if y_prob is not None and pd.Series(y_true).nunique() == 2:
        try: out["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        except Exception: pass
    return out


RESULTS = {}

def evaluate(name, model, Xtr, Xte, ytr, yte, task, store=True, verbose=True):
    """Fit, predict on the held-out test set, score, remember the result."""
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    prob = None
    if task == "classification" and hasattr(model, "predict_proba"):
        try: prob = model.predict_proba(Xte)[:, 1]
        except Exception: pass
    s = score(yte, pred, task, prob)
    if store: RESULTS[name] = s
    if verbose:
        print(f"{name:<22} " + "   ".join(f"{k} {v:.3f}" for k, v in s.items()))
    return model, s


def baseline(Xtr, Xte, ytr, yte, task):
    """The score to beat. A model that cannot beat this has learnt nothing."""
    m = DummyRegressor(strategy="mean") if task == "regression" else DummyClassifier(strategy="most_frequent")
    _, s = evaluate("BASELINE (no model)", m, Xtr, Xte, ytr, yte, task)
    return s


def leaderboard():
    if not RESULTS:
        print("Nothing evaluated yet."); return None
    t = pd.DataFrame(RESULTS).T.round(3)
    key = "R2" if "R2" in t.columns else "F1"
    return t.sort_values(key, ascending=False)


def cv(model, X, y, task, folds=5):
    scoring = "r2" if task == "regression" else "f1_macro"
    kf = KFold(folds, shuffle=True, random_state=42) if task == "regression" \
         else StratifiedKFold(folds, shuffle=True, random_state=42)
    s = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    print(f"{folds}-fold {scoring}: " + " ".join(f"{v:.3f}" for v in s))
    print(f"  mean {s.mean():.3f}   spread +/- {s.std():.3f}")
    return s


def leak_check(X, y, task, thresh=0.95):
    """
    Flags any single feature that predicts the target almost perfectly on its own.
    The automatic guard only removes the column the target was DERIVED from.
    It cannot know that m1..m5 add up to total - only you can. This finds those.
    """
    yy = pd.Series(y).astype(float) if task == "regression" else pd.Series(y)
    suspects = []
    for c in X.columns:
        col = X[c]
        if col.nunique() <= 1:
            continue
        if task == "regression":
            r = abs(np.corrcoef(col, yy)[0, 1])
        else:
            try:    r = abs(np.corrcoef(col, pd.factorize(yy)[0])[0, 1])
            except Exception: continue
        if np.isfinite(r) and r >= thresh:
            suspects.append((c, round(float(r), 3)))
    if suspects:
        print("POSSIBLE LEAK - these features almost ARE the target:")
        for c, r in sorted(suspects, key=lambda t: -t[1]):
            print(f"   {c:<28} correlation {r}")
        print("   If a feature is computed FROM the target, drop it. A model that")
        print("   sees the answer is not predicting - it is reading.")
    else:
        print("No single feature gives the answer away. (Combinations still might.)")
    return suspects


def importances(model, X, top=12):
    if hasattr(model, "feature_importances_"):
        v = pd.Series(model.feature_importances_, index=X.columns)
    elif hasattr(model, "coef_"):
        c = np.ravel(model.coef_)
        v = pd.Series(np.abs(c), index=X.columns)
    else:
        print("This model does not expose importances."); return None
    return v.sort_values(ascending=False).head(top).round(4)
