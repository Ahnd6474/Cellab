import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train_hierarchical_transformer import (
    BasicTokenizer,
    TransformerCollator,
    TweetDataset,
    build_classifier,
    build_model_text,
    build_vocab,
    clean_text,
    evaluate,
    generate_submission,
    set_seed,
)


POSITIVE_LABEL_ALIASES = {
    "1",
    "true",
    "yes",
    "positive",
    "relevant",
    "related",
    "informative",
    "on topic",
    "ontopic",
    "disaster",
}

NEGATIVE_LABEL_ALIASES = {
    "0",
    "false",
    "no",
    "negative",
    "irrelevant",
    "unrelated",
    "not informative",
    "non informative",
    "off topic",
    "offtopic",
    "not disaster",
}

DEFAULT_POSITIVE_KEYWORDS = [
    "earthquake",
    "wildfire",
    "flood",
    "hurricane",
    "evacuation",
    "landslide",
    "explosion",
    "tornado",
]

DEFAULT_LOCATIONS = [
    "Seoul",
    "Busan",
    "Tokyo",
    "Jakarta",
    "Los Angeles",
    "Miami",
    "Houston",
    "Manila",
]

POSITIVE_TEMPLATES = [
    "{source}: {keyword} reported near {location}. Emergency crews are on scene and residents are being evacuated. {hashtag}",
    "Authorities issued an alert for {location} after a {keyword} damaged roads and power lines overnight. {hashtag}",
    "Update from {location}: responders are searching homes after the {keyword}. Avoid the area and follow official instructions. {hashtag}",
    "Breaking: heavy smoke and debris seen in {location} after the {keyword}. Rescue teams are moving in now. {hashtag}",
]

NEGATIVE_TEMPLATES = [
    "My {context} is {phrase} after the product launch, but everything is completely fine. {ending} {hashtag}",
    "This {context} turned into {phrase} when the concert tickets dropped. {ending} {hashtag}",
    "{context} is {phrase}, and I still have not started the essay. {ending} {hashtag}",
    "That {context} is {phrase} in the best possible way. {ending} {hashtag}",
]

NEGATIVE_PHRASES = [
    ("fire", "on fire"),
    ("flood", "a flood of messages"),
    ("explosion", "an explosion of memes"),
    ("crash", "a full crash on the couch"),
    ("disaster", "an absolute disaster"),
    ("meltdown", "a tiny meltdown"),
    ("storm", "a storm of ideas"),
    ("wreck", "a train wreck"),
]

NEGATIVE_CONTEXTS = [
    "inbox",
    "group chat",
    "playlist",
    "calendar",
    "sleep schedule",
    "to-do list",
    "weekend plan",
    "browser tabs",
    "desk setup",
    "gaming session",
]

NEGATIVE_ENDINGS = [
    "nothing serious though",
    "still just a normal day",
    "absolutely no real emergency",
    "all jokes aside, everyone is okay",
    "just being dramatic as usual",
    "it is only a metaphor",
]

SYNTHETIC_HASHTAGS = ["#breaking", "#update", "#news", "#alert", "#concert", "#finals"]


def normalized_text_key(text: str) -> str:
    return clean_text(text).lower()


def build_model_text_from_parts(keyword: str = "", location: str = "", text: str = "") -> str:
    pieces: List[str] = []
    keyword = clean_text(keyword)
    location = clean_text(location)
    text = clean_text(text)
    if keyword:
        pieces.append(f"[KEYWORD] {keyword}")
    if location:
        pieces.append(f"[LOCATION] {location}")
    if text:
        pieces.append(text)
    return " ".join(pieces)


def parse_aliases(raw: str) -> set[str]:
    if not raw:
        return set()
    return {
        clean_text(alias).lower().replace("_", " ").replace("-", " ")
        for alias in raw.split(",")
        if clean_text(alias)
    }


def normalize_binary_label(value, positive_aliases: set[str], negative_aliases: set[str]) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        value = int(value)
        return value if value in (0, 1) else None
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            value = int(value)
            return value if value in (0, 1) else None
        return None
    normalized = clean_text(str(value)).lower().replace("_", " ").replace("-", " ")
    if normalized in positive_aliases:
        return 1
    if normalized in negative_aliases:
        return 0
    return None


def resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def normalize_supervised_frame(
    df: pd.DataFrame,
    source_name: str,
    positive_aliases: set[str],
    negative_aliases: set[str],
) -> pd.DataFrame:
    text_column = resolve_column(df, ["model_text", "text", "tweet_text", "tweet", "content", "message"])
    label_column = resolve_column(df, ["target", "label", "class", "binary_label"])
    keyword_column = resolve_column(df, ["keyword", "keywords"])
    location_column = resolve_column(df, ["location", "place", "region"])
    if text_column is None or label_column is None:
        raise ValueError("expected text and label columns")

    normalized_labels = df[label_column].apply(
        lambda value: normalize_binary_label(value, positive_aliases, negative_aliases)
    )
    usable = df.loc[normalized_labels.notna()].copy()
    if usable.empty:
        raise ValueError("no binary labels found")

    usable["target"] = normalized_labels.loc[usable.index].astype(int)
    usable["keyword"] = usable[keyword_column].fillna("").astype(str) if keyword_column else ""
    usable["location"] = usable[location_column].fillna("").astype(str) if location_column else ""
    usable["text"] = usable[text_column].fillna("").astype(str)
    usable["model_text"] = usable.apply(build_model_text, axis=1).map(clean_text)
    usable = usable[usable["model_text"].str.len() >= 5].copy()
    usable["source"] = source_name
    return usable[["model_text", "target", "source"]].reset_index(drop=True)


def deduplicate_by_text(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    deduped = df.copy()
    deduped["_text_key"] = deduped["model_text"].map(normalized_text_key)
    deduped = deduped.drop_duplicates(subset="_text_key", keep="first").drop(columns="_text_key")
    return deduped.reset_index(drop=True)


def drop_text_overlap(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    if train_df.empty or holdout_df.empty:
        return train_df.reset_index(drop=True)
    holdout_keys = set(holdout_df["model_text"].map(normalized_text_key))
    filtered = train_df.loc[~train_df["model_text"].map(normalized_text_key).isin(holdout_keys)].copy()
    return filtered.reset_index(drop=True)


def load_extra_supervised_data(
    extra_data_dir: Path,
    positive_aliases: set[str],
    negative_aliases: set[str],
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if not extra_data_dir.exists():
        print(f"No extra data directory found at: {extra_data_dir}")
        return pd.DataFrame(columns=["model_text", "target", "source"])

    candidate_paths = sorted(extra_data_dir.glob("*.csv")) + sorted(extra_data_dir.glob("*.tsv"))
    if not candidate_paths:
        print(f"No extra CSV/TSV files found in: {extra_data_dir}")
        return pd.DataFrame(columns=["model_text", "target", "source"])

    frames: List[pd.DataFrame] = []
    for path in candidate_paths:
        try:
            frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_csv(path, sep="\t")
            normalized = normalize_supervised_frame(frame, path.name, positive_aliases, negative_aliases)
            frames.append(normalized)
            print(f"Loaded extra data from {path.name}: {len(normalized)} rows")
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["model_text", "target", "source"])

    extra_df = deduplicate_by_text(pd.concat(frames, ignore_index=True))
    if max_rows > 0 and len(extra_df) > max_rows:
        extra_df = extra_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return extra_df


def keyword_pool_from_frame(df: pd.DataFrame, target: int) -> List[str]:
    if "keyword" not in df.columns:
        return []
    subset = df.loc[df["target"].astype(int) == target]
    keywords = [clean_text(value).lower() for value in subset["keyword"].tolist() if clean_text(value)]
    return sorted(set(keywords))


def location_pool_from_frame(df: pd.DataFrame) -> List[str]:
    if "location" not in df.columns:
        return []
    locations = [clean_text(value) for value in df["location"].tolist() if clean_text(value)]
    return sorted(set(locations))


def generate_synthetic_examples(train_df: pd.DataFrame, num_positive: int, num_negative: int, seed: int) -> pd.DataFrame:
    if num_positive <= 0 and num_negative <= 0:
        return pd.DataFrame(columns=["model_text", "target", "source"])

    rng = random.Random(seed + 17)
    positive_keywords = keyword_pool_from_frame(train_df, target=1) or DEFAULT_POSITIVE_KEYWORDS
    locations = location_pool_from_frame(train_df) or DEFAULT_LOCATIONS
    seen = set(train_df["model_text"].map(normalized_text_key).tolist())
    rows: List[dict] = []
    positive_attempts = 0
    negative_attempts = 0

    while sum(row["target"] == 1 for row in rows) < num_positive:
        positive_attempts += 1
        if positive_attempts > num_positive * 20 + 1000:
            raise ValueError("too many synthetic positives requested for the available template combinations")
        keyword = rng.choice(positive_keywords)
        location = rng.choice(locations)
        text = rng.choice(POSITIVE_TEMPLATES).format(
            source=rng.choice(["ALERT", "Dispatch", "Local update", "Witness report"]),
            keyword=keyword,
            location=location,
            hashtag=rng.choice(SYNTHETIC_HASHTAGS),
        )
        model_text = build_model_text_from_parts(
            keyword=keyword,
            location=location,
            text=text,
        )
        text_key = normalized_text_key(model_text)
        if text_key in seen:
            continue
        seen.add(text_key)
        rows.append({"model_text": model_text, "target": 1, "source": "synthetic_positive"})

    while sum(row["target"] == 0 for row in rows) < num_negative:
        negative_attempts += 1
        if negative_attempts > num_negative * 20 + 1000:
            raise ValueError("too many synthetic negatives requested for the available template combinations")
        keyword, phrase = rng.choice(NEGATIVE_PHRASES)
        text = rng.choice(NEGATIVE_TEMPLATES).format(
            context=rng.choice(NEGATIVE_CONTEXTS),
            phrase=phrase,
            ending=rng.choice(NEGATIVE_ENDINGS),
            hashtag=rng.choice(SYNTHETIC_HASHTAGS),
        )
        model_text = build_model_text_from_parts(keyword=keyword, location="", text=text)
        text_key = normalized_text_key(model_text)
        if text_key in seen:
            continue
        seen.add(text_key)
        rows.append({"model_text": model_text, "target": 0, "source": "synthetic_negative"})

    return deduplicate_by_text(pd.DataFrame(rows))


def build_labeled_dataloader(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    collator: TransformerCollator,
) -> DataLoader:
    return DataLoader(
        TweetDataset(df["model_text"].tolist(), df["target"].astype(int).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


def save_training_summary(output_dir: Path, summary: dict) -> None:
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    base_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(base_dir / "train.csv")
    train_df["target"] = train_df["target"].astype(int)
    train_df["keyword"] = train_df.get("keyword", "").fillna("")
    train_df["location"] = train_df.get("location", "").fillna("")
    train_df["text"] = train_df.get("text", "").fillna("")
    train_df["model_text"] = train_df.apply(build_model_text, axis=1)
    train_df["source"] = "kaggle_train"

    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["target"].tolist(),
    )
    base_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    positive_aliases = POSITIVE_LABEL_ALIASES | parse_aliases(args.extra_positive_labels)
    negative_aliases = NEGATIVE_LABEL_ALIASES | parse_aliases(args.extra_negative_labels)

    extra_df = load_extra_supervised_data(
        extra_data_dir=Path(args.extra_data_dir),
        positive_aliases=positive_aliases,
        negative_aliases=negative_aliases,
        max_rows=args.max_extra_rows,
        seed=args.seed,
    )
    extra_df = drop_text_overlap(extra_df, val_df)

    synthetic_df = generate_synthetic_examples(
        train_df=base_train_df,
        num_positive=args.synthetic_positives,
        num_negative=args.synthetic_negatives,
        seed=args.seed,
    )
    synthetic_df = drop_text_overlap(synthetic_df, val_df)

    base_train_supervised = deduplicate_by_text(base_train_df[["model_text", "target", "source"]].copy())
    main_train_df = pd.concat([base_train_supervised, synthetic_df], ignore_index=True)
    if args.extra_training_mode == "mix" and not extra_df.empty:
        main_train_df = pd.concat([main_train_df, extra_df], ignore_index=True)
    main_train_df = deduplicate_by_text(main_train_df)

    vocab_corpus = main_train_df["model_text"].tolist()
    if args.extra_training_mode == "pretrain" and not extra_df.empty:
        vocab_corpus += extra_df["model_text"].tolist()
    vocab = build_vocab(vocab_corpus, min_freq=args.min_freq)
    tokenizer = BasicTokenizer(vocab=vocab)
    collator = TransformerCollator(tokenizer=tokenizer, max_tokens=args.max_tokens)

    main_train_loader = build_labeled_dataloader(main_train_df, args.batch_size, True, collator)
    val_loader = build_labeled_dataloader(
        val_df[["model_text", "target"]].assign(source="kaggle_val"),
        args.eval_batch_size,
        False,
        collator,
    )

    extra_train_loader = None
    if args.extra_training_mode == "pretrain" and args.extra_pretrain_epochs > 0 and not extra_df.empty:
        extra_train_loader = build_labeled_dataloader(extra_df, args.batch_size, True, collator)

    summary = {
        "base_train_rows": int(len(base_train_supervised)),
        "validation_rows": int(len(val_df)),
        "synthetic_rows": int(len(synthetic_df)),
        "extra_rows": int(len(extra_df)),
        "final_main_rows": int(len(main_train_df)),
        "extra_training_mode": args.extra_training_mode,
        "extra_pretrain_epochs": int(args.extra_pretrain_epochs),
        "synthetic_positives": int(args.synthetic_positives),
        "synthetic_negatives": int(args.synthetic_negatives),
    }
    save_training_summary(output_dir, summary)
    print(f"Data summary: {json.dumps(summary, ensure_ascii=False)}")

    model = build_classifier(args, tokenizer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    phases = []
    if extra_train_loader is not None:
        phases.append(("extra_pretrain", extra_train_loader, args.extra_pretrain_epochs))
    phases.append(("main_train", main_train_loader, args.epochs))

    best_f1 = -1.0
    model_path = output_dir / "best_transformer.pt"
    total_epochs = sum(epochs for _, _, epochs in phases if epochs > 0)
    global_epoch = 0

    for phase_name, train_loader, phase_epochs in phases:
        if phase_epochs <= 0:
            continue
        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            model.train()
            running_loss: List[float] = []
            progress = tqdm(
                train_loader,
                desc=f"{phase_name} {phase_epoch}/{phase_epochs} | overall {global_epoch}/{total_epochs}",
                leave=False,
            )
            for batch in progress:
                logits = model(
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                )
                labels = batch.labels.to(device)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                running_loss.append(loss.item())
                progress.set_postfix(train_loss=f"{np.mean(running_loss):.4f}")

            val_metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"[{phase_name}] epoch={phase_epoch}/{phase_epochs} "
                f"overall={global_epoch}/{total_epochs} "
                f"train_loss={np.mean(running_loss):.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": vars(args),
                        "vocab": vocab,
                        "data_summary": summary,
                    },
                    model_path,
                )

    submission_path = generate_submission(
        data_dir=base_dir,
        output_dir=output_dir,
        model_path=model_path,
        device=device,
    )

    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Saved checkpoint to: {model_path}")
    print(f"Saved submission to: {submission_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a disaster tweets transformer with optional extra and synthetic data.",
    )
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs_extra"))
    parser.add_argument("--extra-data-dir", type=str, default=str(Path(__file__).resolve().parent / "extra_data"))
    parser.add_argument("--extra-training-mode", choices=["pretrain", "mix"], default="pretrain")
    parser.add_argument("--extra-pretrain-epochs", type=int, default=2)
    parser.add_argument("--max-extra-rows", type=int, default=0)
    parser.add_argument("--extra-positive-labels", type=str, default="")
    parser.add_argument("--extra-negative-labels", type=str, default="")
    parser.add_argument("--synthetic-positives", type=int, default=1000)
    parser.add_argument("--synthetic-negatives", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
