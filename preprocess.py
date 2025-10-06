# preprocess.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

COLUMNS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]
DATA_DIR   = Path(__file__).resolve().parent / "census+income"
TRAIN_PATH = DATA_DIR / "adult.data"
TEST_PATH  = DATA_DIR / "adult.test"

def load_data():
    train = pd.read_csv(TRAIN_PATH, names=COLUMNS, na_values="?", skipinitialspace=True)
    test  = pd.read_csv(TEST_PATH,  names=COLUMNS, na_values="?", skipinitialspace=True, skiprows=1)
    train["income"] = train["income"].str.strip()
    test["income"]  = test["income"].str.strip().str.replace(".", "", regex=False)
    df = pd.concat([train, test], ignore_index=True)

    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income"])

    drop_cols = ["education", "fnlwgt"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    return X, y

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric = Pipeline([
        ("imp", SimpleImputer(strategy="median")),

        ("sc",  StandardScaler(with_mean=False))
    ])
    categorical = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ], remainder="drop", sparse_threshold=1.0)

def preprocess(test_size=0.2, random_state=42):
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pre = build_preprocessor(X)
    X_tr_t = pre.fit_transform(X_tr)
    X_te_t = pre.transform(X_te)
    return pre, X_tr_t, X_te_t, y_tr, y_te
