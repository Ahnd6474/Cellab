from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"


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


@dataclass
class FeatureBuilder:
    age_median: float
    fare_median: float
    embarked_mode: str
    age_bin_edges: np.ndarray
    fare_bin_edges: np.ndarray
    ticket_group_sizes: dict[str, int]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "FeatureBuilder":
        age_series = df["Age"].dropna()
        fare_series = df["Fare"].dropna()

        age_median = float(df["Age"].median())
        fare_median = float(df["Fare"].median())
        embarked_mode = str(df["Embarked"].mode(dropna=True).iloc[0])

        age_bin_edges = cls._make_bin_edges(age_series, fallback=[0.0, 16.0, 32.0, 48.0, 64.0, 80.0])
        fare_bin_edges = cls._make_bin_edges(fare_series, fallback=[0.0, 8.0, 15.0, 32.0, 80.0, 600.0])

        ticket_group_sizes = df["Ticket"].astype(str).value_counts().to_dict()
        return cls(
            age_median=age_median,
            fare_median=fare_median,
            embarked_mode=embarked_mode,
            age_bin_edges=age_bin_edges,
            fare_bin_edges=fare_bin_edges,
            ticket_group_sizes=ticket_group_sizes,
        )

    @staticmethod
    def _make_bin_edges(series: pd.Series, fallback: list[float]) -> np.ndarray:
        if series.nunique() < 4:
            return np.array(fallback, dtype=float)

        _, edges = pd.qcut(series, q=5, duplicates="drop", retbins=True)
        edges = np.unique(edges.astype(float))
        if len(edges) < 3:
            return np.array(fallback, dtype=float)

        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

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
        return features


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
        description="Train Titanic survival model with CatBoost, feature engineering, and stratified CV."
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
    parser.add_argument("--iterations", type=int, default=1500, help="Maximum boosting iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="CatBoost learning rate.")
    parser.add_argument("--depth", type=int, default=6, help="CatBoost tree depth.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    target = train_df[TARGET_COLUMN].copy()
    test_ids = test_df[ID_COLUMN].copy()

    categorical_features = [
        "Sex",
        "Embarked",
        "Title",
        "Deck",
        "TicketPrefix",
        "AgeBin",
        "FareBin",
    ]

    oof_pred = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)
    fold_accuracies: list[float] = []
    fold_aucs: list[float] = []

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_df, target), start=1):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()

        builder = FeatureBuilder.fit(fold_train)
        train_features = builder.transform(fold_train)
        valid_features = builder.transform(fold_valid)
        test_features = builder.transform(test_df)

        train_pool, valid_pool = build_pools(
            train_features=train_features,
            train_target=target.iloc[train_idx],
            valid_features=valid_features,
            valid_target=target.iloc[valid_idx],
            categorical_features=categorical_features,
        )
        test_pool = Pool(test_features, cat_features=categorical_features)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=5.0,
            random_strength=1.5,
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=100)

        valid_pred = model.predict_proba(valid_pool)[:, 1]
        oof_pred[valid_idx] = valid_pred
        test_pred += model.predict_proba(test_pool)[:, 1] / args.folds

        valid_label = (valid_pred >= 0.5).astype(int)
        fold_accuracy = accuracy_score(target.iloc[valid_idx], valid_label)
        fold_auc = roc_auc_score(target.iloc[valid_idx], valid_pred)
        fold_accuracies.append(fold_accuracy)
        fold_aucs.append(fold_auc)

        print(
            f"fold={fold} accuracy={fold_accuracy:.4f} auc={fold_auc:.4f} "
            f"best_iteration={model.get_best_iteration()}"
        )

    overall_accuracy = accuracy_score(target, (oof_pred >= 0.5).astype(int))
    overall_auc = roc_auc_score(target, oof_pred)
    print(f"cv_accuracy_mean={np.mean(fold_accuracies):.4f} cv_accuracy_std={np.std(fold_accuracies):.4f}")
    print(f"cv_auc_mean={np.mean(fold_aucs):.4f} cv_auc_std={np.std(fold_aucs):.4f}")
    print(f"oof_accuracy={overall_accuracy:.4f} oof_auc={overall_auc:.4f}")

    submission = pd.DataFrame(
        {
            ID_COLUMN: test_ids.astype(int),
            TARGET_COLUMN: (test_pred >= 0.5).astype(int),
        }
    )
    submission.to_csv(args.output, index=False)
    print(f"saved={args.output}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
