import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_model_text(row: pd.Series) -> str:
    pieces: List[str] = []
    keyword = clean_text(row.get("keyword", ""))
    location = clean_text(row.get("location", ""))
    text = clean_text(row.get("text", ""))

    if keyword:
        pieces.append(f"[KEYWORD] {keyword}")
    if location:
        pieces.append(f"[LOCATION] {location}")
    pieces.append(text)
    return " ".join(piece for piece in pieces if piece)


class TweetDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Optional[Sequence[int]] = None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


class BasicTokenizer:
    def __init__(self, vocab: dict[str, int], lower: bool = True):
        self.vocab = vocab
        self.lower = lower
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token_id = vocab[self.pad_token]
        self.unk_token_id = vocab[self.unk_token]
        self.cls_token_id = vocab[self.cls_token]
        self.sep_token_id = vocab[self.sep_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        normalized = text.lower() if self.lower else text
        return re.findall(r"#\w+|@\w+|\w+|[^\w\s]", normalized)

    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in self.tokenize(text)]
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def __call__(
        self,
        texts: Sequence[str],
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> dict[str, List[List[int]]]:
        if padding:
            raise ValueError("BasicTokenizer does not support padding=True in this pipeline.")
        encoded = [
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length if truncation else None,
            )
            for text in texts
        ]
        return {"input_ids": encoded}


def build_vocab(texts: Sequence[str], min_freq: int) -> dict[str, int]:
    counter: dict[str, int] = {}
    pattern = re.compile(r"#\w+|@\w+|\w+|[^\w\s]")
    for text in texts:
        for token in pattern.findall(text.lower()):
            counter[token] = counter.get(token, 0) + 1

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
    }
    for token, freq in sorted(counter.items()):
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


@dataclass
class HierarchicalBatch:
    input_ids: torch.Tensor
    token_mask: torch.Tensor
    chunk_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None


class HierarchicalCollator:
    def __init__(self, tokenizer, max_tokens: int, chunk_size: int, max_chunks: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks

    def __call__(self, batch: Sequence[dict]) -> HierarchicalBatch:
        texts = [sample["text"] for sample in batch]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
        )

        batch_size = len(batch)
        input_ids = torch.zeros((batch_size, self.max_chunks, self.chunk_size), dtype=torch.long)
        token_mask = torch.zeros((batch_size, self.max_chunks, self.chunk_size), dtype=torch.bool)
        chunk_mask = torch.zeros((batch_size, self.max_chunks), dtype=torch.bool)

        for row_idx, token_ids in enumerate(encoded["input_ids"]):
            chunks = [
                token_ids[start : start + self.chunk_size]
                for start in range(0, min(len(token_ids), self.chunk_size * self.max_chunks), self.chunk_size)
            ]
            for chunk_idx, chunk in enumerate(chunks[: self.max_chunks]):
                length = len(chunk)
                input_ids[row_idx, chunk_idx, :length] = torch.tensor(chunk, dtype=torch.long)
                token_mask[row_idx, chunk_idx, :length] = True
                chunk_mask[row_idx, chunk_idx] = True

        labels = None
        if "label" in batch[0]:
            labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)

        return HierarchicalBatch(
            input_ids=input_ids,
            token_mask=token_mask,
            chunk_mask=chunk_mask,
            labels=labels,
        )


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, hidden_size: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HierarchicalTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        chunk_size: int,
        max_chunks: int,
        hidden_size: int,
        num_heads: int,
        num_token_layers: int,
        num_chunk_layers: int,
        ff_dim: int,
        dropout: float,
        num_labels: int = 2,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.token_position = PositionalEncoding(chunk_size, hidden_size)
        self.chunk_position = PositionalEncoding(max_chunks, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        token_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=num_token_layers)
        self.token_pooler = AttentionPooling(hidden_size)

        chunk_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.chunk_encoder = nn.TransformerEncoder(chunk_layer, num_layers=num_chunk_layers)
        self.chunk_pooler = AttentionPooling(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_chunks, chunk_size = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * max_chunks, chunk_size)
        flat_token_mask = token_mask.view(batch_size * max_chunks, chunk_size)
        safe_flat_token_mask = flat_token_mask.clone()
        empty_token_rows = ~safe_flat_token_mask.any(dim=1)
        safe_flat_token_mask[empty_token_rows, 0] = True

        token_embeddings = self.token_embedding(flat_input_ids) * math.sqrt(self.hidden_size)
        token_embeddings = self.token_position(token_embeddings)
        token_embeddings = self.embedding_dropout(token_embeddings)

        encoded_tokens = self.token_encoder(
            token_embeddings,
            src_key_padding_mask=~safe_flat_token_mask,
        )
        chunk_vectors = self.token_pooler(encoded_tokens, flat_token_mask)
        chunk_vectors = chunk_vectors.view(batch_size, max_chunks, self.hidden_size)

        safe_chunk_mask = chunk_mask.clone()
        empty_chunk_rows = ~safe_chunk_mask.any(dim=1)
        safe_chunk_mask[empty_chunk_rows, 0] = True
        chunk_vectors = self.chunk_position(chunk_vectors)
        chunk_vectors = self.embedding_dropout(chunk_vectors)
        encoded_chunks = self.chunk_encoder(
            chunk_vectors,
            src_key_padding_mask=~safe_chunk_mask,
        )
        tweet_vector = self.chunk_pooler(encoded_chunks, chunk_mask)
        tweet_vector = self.norm(tweet_vector)
        return self.classifier(tweet_vector)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch.input_ids.to(device),
                token_mask=batch.token_mask.to(device),
                chunk_mask=batch.chunk_mask.to(device),
            )
            labels = batch.labels.to(device)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
    }


def predict(model, dataloader, device):
    model.eval()
    predictions: List[int] = []
    probabilities: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch.input_ids.to(device),
                token_mask=batch.token_mask.to(device),
                chunk_mask=batch.chunk_mask.to(device),
            )
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            probabilities.extend(probs)
            predictions.extend(preds)

    return predictions, probabilities


def build_classifier(args, tokenizer: BasicTokenizer) -> HierarchicalTransformerClassifier:
    return HierarchicalTransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_token_layers=args.num_token_layers,
        num_chunk_layers=args.num_chunk_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )


def load_model_from_checkpoint(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_config = argparse.Namespace(**checkpoint["config"])
    tokenizer = BasicTokenizer(vocab=checkpoint["vocab"])
    model = build_classifier(checkpoint_config, tokenizer).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer, checkpoint_config


def generate_submission(
    data_dir: Path,
    output_dir: Path,
    model_path: Path,
    device: torch.device,
):
    model, tokenizer, checkpoint_config = load_model_from_checkpoint(model_path, device)

    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    test_df["model_text"] = test_df.apply(build_model_text, axis=1)

    collator = HierarchicalCollator(
        tokenizer=tokenizer,
        max_tokens=checkpoint_config.max_tokens,
        chunk_size=checkpoint_config.chunk_size,
        max_chunks=checkpoint_config.max_chunks,
    )
    test_loader = DataLoader(
        TweetDataset(test_df["model_text"].tolist()),
        batch_size=checkpoint_config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    test_predictions, test_probabilities = predict(model, test_loader, device)

    submission = sample_submission.copy()
    if len(submission) != len(test_predictions):
        raise ValueError(
            f"sample submission row count ({len(submission)}) does not match predictions ({len(test_predictions)})."
        )
    target_column = submission.columns[-1]
    submission[target_column] = test_predictions
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    pd.DataFrame(
        {
            "id": test_df["id"],
            "target": test_predictions,
            "prob_disaster": test_probabilities,
        }
    ).to_csv(output_dir / "test_probabilities_hierarchical_transformer.csv", index=False)

    return submission_path


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    base_dir = Path(args.data_dir)

    train_df = pd.read_csv(base_dir / "train.csv")
    train_df["model_text"] = train_df.apply(build_model_text, axis=1)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["model_text"].tolist(),
        train_df["target"].astype(int).tolist(),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["target"].astype(int).tolist(),
    )

    vocab = build_vocab(train_texts, min_freq=args.min_freq)
    tokenizer = BasicTokenizer(vocab=vocab)
    collator = HierarchicalCollator(
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )

    train_loader = DataLoader(
        TweetDataset(train_texts, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        TweetDataset(val_texts, val_labels),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = build_classifier(args, tokenizer).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_hierarchical_transformer.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in progress:
            logits = model(
                input_ids=batch.input_ids.to(device),
                token_mask=batch.token_mask.to(device),
                chunk_mask=batch.chunk_mask.to(device),
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
            f"Epoch {epoch}: "
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
        description="Train a hierarchical transformer for Kaggle disaster tweets.",
    )
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-token-layers", type=int, default=2)
    parser.add_argument("--num-chunk-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
