from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold


SEED = 42
TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"
SURVIVAL_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "TicketPrefix",
    "AgeBin",
    "FareBin",
]
IMPUTATION_CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck", "TicketPrefix"]


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^.]*)\.", str(name))
    if not match:
        return "Unknown"
    title = match.group(1).strip()
    normalized = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Sir": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Jonkheer": "Royalty",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
    }
    return normalized.get(title, title)


def extract_ticket_prefix(ticket: str) -> str:
    ticket = str(ticket).strip().replace(".", "").replace("/", "")
    parts = ticket.split()
    if len(parts) == 1 and parts[0].isdigit():
        return "NUM"
    return parts[0] if parts else "UNKNOWN"


def build_ticket_group_sizes(df: pd.DataFrame) -> dict[str, int]:
    return df["Ticket"].astype(str).value_counts().to_dict()


def make_bin_edges(series: pd.Series, fallback: list[float]) -> np.ndarray:
    if series.nunique() < 4:
        return np.array(fallback, dtype=float)

    _, edges = pd.qcut(series, q=5, duplicates="drop", retbins=True)
    edges = np.unique(edges.astype(float))
    if len(edges) < 3:
        return np.array(fallback, dtype=float)

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def cat_features_for(columns: pd.Index | list[str]) -> list[str]:
    return [column for column in IMPUTATION_CATEGORICAL_FEATURES if column in set(columns)]


def build_imputation_frame(df: pd.DataFrame, ticket_group_sizes: dict[str, int]) -> pd.DataFrame:
    family_size = df["SibSp"] + df["Parch"] + 1
    fare = pd.to_numeric(df["Fare"], errors="coerce")

    features = pd.DataFrame(index=df.index)
    features["Pclass"] = df["Pclass"].astype(int)
    features["Sex"] = df["Sex"].fillna("Unknown").astype(str)
    features["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    features["Fare"] = fare.astype(float)
    features["Embarked"] = df["Embarked"].fillna("Missing").astype(str)
    features["SibSp"] = df["SibSp"].astype(int)
    features["Parch"] = df["Parch"].astype(int)
    features["FamilySize"] = family_size.astype(int)
    features["IsAlone"] = (family_size == 1).astype(int)
    features["FarePerPerson"] = (fare / family_size.replace(0, 1)).astype(float)
    features["Title"] = df["Name"].map(extract_title).astype(str)
    features["Deck"] = df["Cabin"].fillna("Missing").astype(str).str[0]
    features["CabinKnown"] = df["Cabin"].notna().astype(int)
    features["TicketPrefix"] = df["Ticket"].map(extract_ticket_prefix).astype(str)
    features["TicketGroupSize"] = df["Ticket"].astype(str).map(ticket_group_sizes).fillna(1).astype(int)
    return features


@dataclass
class TreeBasedImputer:
    ticket_group_sizes: dict[str, int]
    age_fallback: float
    fare_fallback: float
    embarked_fallback: str
    age_bounds: tuple[float, float]
    fare_bounds: tuple[float, float]
    age_model: CatBoostRegressor | None
    fare_model: CatBoostRegressor | None
    embarked_model: CatBoostClassifier | None

    @classmethod
    def fit(cls, df: pd.DataFrame, seed: int) -> "TreeBasedImputer":
        ticket_group_sizes = build_ticket_group_sizes(df)

        age_non_null = df["Age"].dropna()
        fare_non_null = df["Fare"].dropna()
        embarked_non_null = df["Embarked"].dropna()

        age_fallback = float(age_non_null.median()) if not age_non_null.empty else 29.7
        fare_fallback = float(fare_non_null.median()) if not fare_non_null.empty else 14.4542
        embarked_fallback = str(embarked_non_null.mode().iloc[0]) if not embarked_non_null.empty else "S"
        age_bounds = (
            float(age_non_null.min()) if not age_non_null.empty else 0.0,
            float(age_non_null.max()) if not age_non_null.empty else 80.0,
        )
        fare_bounds = (
            float(fare_non_null.min()) if not fare_non_null.empty else 0.0,
            float(fare_non_null.max()) if not fare_non_null.empty else 512.3292,
        )

        frame = build_imputation_frame(df, ticket_group_sizes)

        embarked_model: CatBoostClassifier | None = None
        embarked_mask = df["Embarked"].notna()
        if embarked_mask.sum() >= 50 and df.loc[embarked_mask, "Embarked"].nunique() > 1:
            embarked_features = frame.loc[embarked_mask].drop(columns=["Embarked"])
            embarked_target = df.loc[embarked_mask, "Embarked"].astype(str)
            embarked_model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                iterations=300,
                learning_rate=0.05,
                depth=5,
                l2_leaf_reg=4.0,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
            embarked_model.fit(
                Pool(
                    embarked_features,
                    label=embarked_target,
                    cat_features=cat_features_for(embarked_features.columns),
                )
            )

        fare_model: CatBoostRegressor | None = None
        fare_mask = df["Fare"].notna()
        if fare_mask.sum() >= 50:
            fare_features = frame.loc[fare_mask].drop(columns=["Fare"])
            fare_target = df.loc[fare_mask, "Fare"].astype(float)
            fare_model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=400,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=5.0,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
            fare_model.fit(
                Pool(
                    fare_features,
                    label=fare_target,
                    cat_features=cat_features_for(fare_features.columns),
                )
            )

        age_model: CatBoostRegressor | None = None
        age_mask = df["Age"].notna()
        if age_mask.sum() >= 50:
            age_features = frame.loc[age_mask].drop(columns=["Age"])
            age_target = df.loc[age_mask, "Age"].astype(float)
            age_model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=5.0,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
            age_model.fit(
                Pool(
                    age_features,
                    label=age_target,
                    cat_features=cat_features_for(age_features.columns),
                )
            )

        return cls(
            ticket_group_sizes=ticket_group_sizes,
            age_fallback=age_fallback,
            fare_fallback=fare_fallback,
            embarked_fallback=embarked_fallback,
            age_bounds=age_bounds,
            fare_bounds=fare_bounds,
            age_model=age_model,
            fare_model=fare_model,
            embarked_model=embarked_model,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        embarked_missing = data["Embarked"].isna()
        if self.embarked_model is not None and embarked_missing.any():
            embarked_features = build_imputation_frame(data, self.ticket_group_sizes).drop(columns=["Embarked"])
            embarked_pred = self.embarked_model.predict(
                Pool(
                    embarked_features.loc[embarked_missing],
                    cat_features=cat_features_for(embarked_features.columns),
                )
            )
            data.loc[embarked_missing, "Embarked"] = np.asarray(embarked_pred).reshape(-1).astype(str)

        fare_missing = data["Fare"].isna()
        if self.fare_model is not None and fare_missing.any():
            fare_features = build_imputation_frame(data, self.ticket_group_sizes).drop(columns=["Fare"])
            fare_pred = np.asarray(
                self.fare_model.predict(
                    Pool(
                        fare_features.loc[fare_missing],
                        cat_features=cat_features_for(fare_features.columns),
                    )
                )
            ).reshape(-1)
            data.loc[fare_missing, "Fare"] = np.clip(fare_pred, self.fare_bounds[0], self.fare_bounds[1])

        age_missing = data["Age"].isna()
        if self.age_model is not None and age_missing.any():
            age_features = build_imputation_frame(data, self.ticket_group_sizes).drop(columns=["Age"])
            age_pred = np.asarray(
                self.age_model.predict(
                    Pool(
                        age_features.loc[age_missing],
                        cat_features=cat_features_for(age_features.columns),
                    )
                )
            ).reshape(-1)
            data.loc[age_missing, "Age"] = np.clip(age_pred, self.age_bounds[0], self.age_bounds[1])

        data["Embarked"] = data["Embarked"].fillna(self.embarked_fallback)
        data["Fare"] = data["Fare"].fillna(self.fare_fallback)
        data["Age"] = data["Age"].fillna(self.age_fallback)
        return data


@dataclass
class FeatureBuilder:
    age_median: float
    fare_median: float
    embarked_mode: str
    age_bin_edges: np.ndarray
    fare_bin_edges: np.ndarray
    ticket_group_sizes: dict[str, int]
    tree_imputer: TreeBasedImputer | None = None

    @classmethod
    def fit(cls, df: pd.DataFrame, use_tree_imputer: bool, seed: int) -> "FeatureBuilder":
        tree_imputer = TreeBasedImputer.fit(df, seed=seed) if use_tree_imputer else None
        stats_source = tree_imputer.transform(df) if tree_imputer is not None else df.copy()

        age_median = float(stats_source["Age"].median())
        fare_median = float(stats_source["Fare"].median())
        embarked_mode = str(stats_source["Embarked"].mode(dropna=True).iloc[0])

        age_bin_edges = make_bin_edges(stats_source["Age"].dropna(), fallback=[0.0, 16.0, 32.0, 48.0, 64.0, 80.0])
        fare_bin_edges = make_bin_edges(stats_source["Fare"].dropna(), fallback=[0.0, 8.0, 15.0, 32.0, 80.0, 600.0])

        return cls(
            age_median=age_median,
            fare_median=fare_median,
            embarked_mode=embarked_mode,
            age_bin_edges=age_bin_edges,
            fare_bin_edges=fare_bin_edges,
            ticket_group_sizes=build_ticket_group_sizes(df),
            tree_imputer=tree_imputer,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        original = df.copy()
        data = self.tree_imputer.transform(df) if self.tree_imputer is not None else df.copy()

        age_filled = data["Age"].fillna(self.age_median)
        fare_filled = data["Fare"].fillna(self.fare_median)
        embarked_filled = data["Embarked"].fillna(self.embarked_mode)

        family_size = data["SibSp"] + data["Parch"] + 1
        ticket_count = data["Ticket"].astype(str).map(self.ticket_group_sizes).fillna(1).astype(int)

        features = pd.DataFrame(index=data.index)
        features["Pclass"] = data["Pclass"].astype(int)
        features["Sex"] = data["Sex"].fillna("Unknown").astype(str)
        features["Embarked"] = embarked_filled.astype(str)
        features["Age"] = age_filled.astype(float)
        features["Fare"] = fare_filled.astype(float)
        features["SibSp"] = data["SibSp"].astype(int)
        features["Parch"] = data["Parch"].astype(int)
        features["FamilySize"] = family_size.astype(int)
        features["IsAlone"] = (family_size == 1).astype(int)
        features["FarePerPerson"] = (fare_filled / family_size.replace(0, 1)).astype(float)
        features["Title"] = data["Name"].map(extract_title).astype(str)
        features["Deck"] = data["Cabin"].fillna("Missing").astype(str).str[0]
        features["CabinKnown"] = data["Cabin"].notna().astype(int)
        features["TicketPrefix"] = data["Ticket"].map(extract_ticket_prefix).astype(str)
        features["TicketGroupSize"] = ticket_count
        features["AgeBin"] = pd.cut(age_filled, bins=self.age_bin_edges, include_lowest=True).astype(str)
        features["FareBin"] = pd.cut(fare_filled, bins=self.fare_bin_edges, include_lowest=True).astype(str)
        features["AgeWasMissing"] = original["Age"].isna().astype(int)
        features["FareWasMissing"] = original["Fare"].isna().astype(int)
        features["EmbarkedWasMissing"] = original["Embarked"].isna().astype(int)
        return features


@dataclass(frozen=True)
class CatBoostConfig:
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    iterations: int
    subsample: float

    def to_model_kwargs(self, seed: int) -> dict[str, float | int | str | bool]:
        return {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "bootstrap_type": "Bernoulli",
            "subsample": self.subsample,
            "random_strength": 1.5,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }

    def dedupe_key(self) -> tuple[int, float, float, int, float]:
        return (
            self.depth,
            round(self.learning_rate, 10),
            round(self.l2_leaf_reg, 10),
            self.iterations,
            round(self.subsample, 10),
        )

    def summary(self) -> str:
        return (
            f"depth={self.depth} learning_rate={self.learning_rate:.4f} "
            f"l2_leaf_reg={self.l2_leaf_reg:.4f} iterations={self.iterations} subsample={self.subsample:.2f}"
        )


@dataclass
class CVResult:
    name: str
    config: CatBoostConfig
    fold_accuracies: list[float]
    fold_aucs: list[float]
    best_iterations: list[int]
    oof_pred: np.ndarray
    test_pred: np.ndarray | None

    @property
    def cv_accuracy_mean(self) -> float:
        return float(np.mean(self.fold_accuracies))

    @property
    def cv_accuracy_std(self) -> float:
        return float(np.std(self.fold_accuracies))

    @property
    def cv_auc_mean(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def cv_auc_std(self) -> float:
        return float(np.std(self.fold_aucs))

    def overall_accuracy(self, target: pd.Series) -> float:
        return float(accuracy_score(target, (self.oof_pred >= 0.5).astype(int)))

    def overall_auc(self, target: pd.Series) -> float:
        return float(roc_auc_score(target, self.oof_pred))


def build_pools(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
    categorical_features: list[str],
) -> tuple[Pool, Pool]:
    train_pool = Pool(train_features, label=train_target, cat_features=categorical_features)
    valid_pool = Pool(valid_features, label=valid_target, cat_features=categorical_features)
    return train_pool, valid_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Titanic survival model with tree-based imputation and ensemble CatBoost CV."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path(__file__).with_name("train.csv"),
        help="Path to the Titanic train.csv file.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path(__file__).with_name("test.csv"),
        help="Path to the Titanic test.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("submission_catboost_cv.csv"),
        help="Path to save the submission CSV.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of StratifiedKFold splits.")
    parser.add_argument("--iterations", type=int, default=800, help="Maximum boosting iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="CatBoost learning rate.")
    parser.add_argument("--depth", type=int, default=7, help="CatBoost tree depth.")
    parser.add_argument("--l2-leaf-reg", type=float, default=4.0, help="CatBoost L2 leaf regularization.")
    parser.add_argument("--subsample", type=float, default=0.9, help="CatBoost Bernoulli subsample ratio.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--tune", action="store_true", help="Run random search before training.")
    parser.add_argument("--tune-trials", type=int, default=16, help="Number of sampled tuning trials.")
    parser.add_argument(
        "--tune-builder-mode",
        choices=["simple", "tree"],
        default="tree",
        help="Feature builder used during hyperparameter tuning.",
    )
    parser.add_argument("--depth-grid", type=int, nargs="+", default=[4, 5, 6, 7, 8], help="Depth candidates.")
    parser.add_argument(
        "--learning-rate-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.05, 0.08],
        help="Learning-rate candidates.",
    )
    parser.add_argument(
        "--l2-leaf-reg-grid",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0, 10.0],
        help="L2 regularization candidates.",
    )
    parser.add_argument(
        "--iterations-grid",
        type=int,
        nargs="+",
        default=[400, 800, 1200, 1600],
        help="Iteration candidates.",
    )
    parser.add_argument(
        "--subsample-grid",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9, 1.0],
        help="Subsample candidates.",
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        default=-1.0,
        help="Tree-imputed model weight. Use a negative value to search the best OOF weight automatically.",
    )
    return parser.parse_args()


def train_with_cv(
    train_df: pd.DataFrame,
    target: pd.Series,
    test_df: pd.DataFrame | None,
    folds: int,
    seed: int,
    config: CatBoostConfig,
    builder_mode: str,
    log_prefix: str = "",
) -> CVResult:
    oof_pred = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float) if test_df is not None else None
    fold_accuracies: list[float] = []
    fold_aucs: list[float] = []
    best_iterations: list[int] = []

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_df, target), start=1):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()

        builder = FeatureBuilder.fit(
            fold_train,
            use_tree_imputer=(builder_mode == "tree"),
            seed=seed + fold,
        )
        train_features = builder.transform(fold_train)
        valid_features = builder.transform(fold_valid)

        train_pool, valid_pool = build_pools(
            train_features=train_features,
            train_target=target.iloc[train_idx],
            valid_features=valid_features,
            valid_target=target.iloc[valid_idx],
            categorical_features=SURVIVAL_CATEGORICAL_FEATURES,
        )

        model = CatBoostClassifier(**config.to_model_kwargs(seed + fold))
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=100)

        valid_pred = model.predict_proba(valid_pool)[:, 1]
        oof_pred[valid_idx] = valid_pred

        if test_df is not None and test_pred is not None:
            test_features = builder.transform(test_df)
            test_pool = Pool(test_features, cat_features=SURVIVAL_CATEGORICAL_FEATURES)
            test_pred += model.predict_proba(test_pool)[:, 1] / folds

        valid_label = (valid_pred >= 0.5).astype(int)
        fold_accuracy = accuracy_score(target.iloc[valid_idx], valid_label)
        fold_auc = roc_auc_score(target.iloc[valid_idx], valid_pred)
        fold_accuracies.append(float(fold_accuracy))
        fold_aucs.append(float(fold_auc))

        best_iteration = model.get_best_iteration()
        if best_iteration is None or best_iteration < 0:
            best_iteration = config.iterations
        best_iterations.append(int(best_iteration))

        print(
            f"{log_prefix}fold={fold} accuracy={fold_accuracy:.4f} auc={fold_auc:.4f} "
            f"best_iteration={best_iteration}"
        )

    return CVResult(
        name=builder_mode,
        config=config,
        fold_accuracies=fold_accuracies,
        fold_aucs=fold_aucs,
        best_iterations=best_iterations,
        oof_pred=oof_pred,
        test_pred=test_pred,
    )


def build_default_config(args: argparse.Namespace) -> CatBoostConfig:
    return CatBoostConfig(
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        iterations=args.iterations,
        subsample=args.subsample,
    )


def tune_hyperparameters(
    train_df: pd.DataFrame,
    target: pd.Series,
    args: argparse.Namespace,
) -> CatBoostConfig:
    search_space = {
        "depth": args.depth_grid,
        "learning_rate": args.learning_rate_grid,
        "l2_leaf_reg": args.l2_leaf_reg_grid,
        "iterations": args.iterations_grid,
        "subsample": args.subsample_grid,
    }

    total_candidates = math.prod(len(values) for values in search_space.values())
    n_trials = min(args.tune_trials, total_candidates)
    sampled_params = list(ParameterSampler(search_space, n_iter=n_trials, random_state=args.seed))

    baseline_config = build_default_config(args)
    candidates = [baseline_config]
    seen = {baseline_config.dedupe_key()}

    for params in sampled_params:
        config = CatBoostConfig(
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            iterations=int(params["iterations"]),
            subsample=float(params["subsample"]),
        )
        if config.dedupe_key() in seen:
            continue
        seen.add(config.dedupe_key())
        candidates.append(config)

    best_result: CVResult | None = None
    best_score = float("-inf")

    print(f"tuning_builder_mode={args.tune_builder_mode} tuning_trials={len(candidates)}")
    for trial, config in enumerate(candidates, start=1):
        print(f"trial={trial} config={config.summary()}")
        result = train_with_cv(
            train_df=train_df,
            target=target,
            test_df=None,
            folds=args.folds,
            seed=args.seed,
            config=config,
            builder_mode=args.tune_builder_mode,
            log_prefix=f"trial={trial} {args.tune_builder_mode} ",
        )
        score = result.overall_auc(target)
        print(
            f"trial={trial} cv_accuracy_mean={result.cv_accuracy_mean:.4f} "
            f"cv_auc_mean={result.cv_auc_mean:.4f} oof_auc={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_result = result

    assert best_result is not None
    print(f"best_config={best_result.config.summary()}")
    print(
        f"best_cv_accuracy_mean={best_result.cv_accuracy_mean:.4f} "
        f"best_cv_auc_mean={best_result.cv_auc_mean:.4f} best_oof_auc={best_result.overall_auc(target):.4f}"
    )
    return best_result.config


def summarize_result(result: CVResult, target: pd.Series) -> None:
    print(
        f"{result.name}_cv_accuracy_mean={result.cv_accuracy_mean:.4f} "
        f"{result.name}_cv_accuracy_std={result.cv_accuracy_std:.4f}"
    )
    print(f"{result.name}_cv_auc_mean={result.cv_auc_mean:.4f} {result.name}_cv_auc_std={result.cv_auc_std:.4f}")
    print(
        f"{result.name}_oof_accuracy={result.overall_accuracy(target):.4f} "
        f"{result.name}_oof_auc={result.overall_auc(target):.4f}"
    )
    print(f"{result.name}_best_iterations={result.best_iterations}")


def find_best_ensemble_weight(
    target: pd.Series,
    simple_pred: np.ndarray,
    tree_pred: np.ndarray,
) -> tuple[float, float]:
    best_weight = 0.5
    best_auc = float("-inf")

    for weight in np.linspace(0.0, 1.0, 41):
        ensemble_pred = (1.0 - weight) * simple_pred + weight * tree_pred
        auc = float(roc_auc_score(target, ensemble_pred))
        if auc > best_auc:
            best_auc = auc
            best_weight = float(weight)

    return best_weight, best_auc


def save_submission(output: Path, test_ids: pd.Series, test_pred: np.ndarray) -> None:
    submission = pd.DataFrame(
        {
            ID_COLUMN: test_ids.astype(int),
            TARGET_COLUMN: (test_pred >= 0.5).astype(int),
        }
    )
    submission.to_csv(output, index=False)
    print(f"saved={output}")
    print(submission.head(10).to_string(index=False))



def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    target = train_df[TARGET_COLUMN].copy()
    test_ids = test_df[ID_COLUMN].copy()

    config = tune_hyperparameters(train_df=train_df, target=target, args=args) if args.tune else build_default_config(args)
    print(f"final_config={config.summary()}")

    simple_result = train_with_cv(
        train_df=train_df,
        target=target,
        test_df=test_df,
        folds=args.folds,
        seed=args.seed,
        config=config,
        builder_mode="simple",
        log_prefix="simple ",
    )
    summarize_result(simple_result, target)

    tree_result = train_with_cv(
        train_df=train_df,
        target=target,
        test_df=test_df,
        folds=args.folds,
        seed=args.seed,
        config=config,
        builder_mode="tree",
        log_prefix="tree ",
    )
    summarize_result(tree_result, target)

    if simple_result.test_pred is None or tree_result.test_pred is None:
        raise RuntimeError("Expected test predictions to be available for submission generation.")

    if args.ensemble_weight < 0.0:
        ensemble_weight, searched_auc = find_best_ensemble_weight(target, simple_result.oof_pred, tree_result.oof_pred)
        print(f"ensemble_weight_search_best={ensemble_weight:.2f} ensemble_search_oof_auc={searched_auc:.4f}")
    else:
        ensemble_weight = args.ensemble_weight
        print(f"ensemble_weight_fixed={ensemble_weight:.2f}")

    ensemble_oof = (1.0 - ensemble_weight) * simple_result.oof_pred + ensemble_weight * tree_result.oof_pred
    ensemble_test = (1.0 - ensemble_weight) * simple_result.test_pred + ensemble_weight * tree_result.test_pred

    ensemble_accuracy = float(accuracy_score(target, (ensemble_oof >= 0.5).astype(int)))
    ensemble_auc = float(roc_auc_score(target, ensemble_oof))
    print(f"ensemble_oof_accuracy={ensemble_accuracy:.4f} ensemble_oof_auc={ensemble_auc:.4f}")

    save_submission(args.output, test_ids, ensemble_test)


if __name__ == "__main__":
    main()
