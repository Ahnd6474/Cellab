from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
MODEL_COLUMNS = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
BINARY_COLUMNS = ["Survived", "Sex"]
PCLASS_CATEGORIES = [1, 2, 3]
EMBARKED_CATEGORIES = ["C", "Q", "S"]
CONTINUOUS_COLUMNS = ["Age", "SibSp", "Parch", "Fare"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TitanicPreprocessor:
    scaler: StandardScaler
    age_fill: float
    fare_fill: float
    embarked_fill: str
    continuous_mins: dict[str, float]
    continuous_maxs: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "TitanicPreprocessor":
        age_fill = float(df["Age"].median())
        fare_fill = float(df["Fare"].median())
        embarked_fill = str(df["Embarked"].mode(dropna=True).iloc[0])

        filled = df.copy()
        filled["Age"] = filled["Age"].fillna(age_fill)
        filled["Fare"] = filled["Fare"].fillna(fare_fill)
        filled["Embarked"] = filled["Embarked"].fillna(embarked_fill)

        scaler = StandardScaler()
        scaler.fit(filled[CONTINUOUS_COLUMNS])

        continuous_mins = {col: float(filled[col].min()) for col in CONTINUOUS_COLUMNS}
        continuous_maxs = {col: float(filled[col].max()) for col in CONTINUOUS_COLUMNS}
        return cls(
            scaler=scaler,
            age_fill=age_fill,
            fare_fill=fare_fill,
            embarked_fill=embarked_fill,
            continuous_mins=continuous_mins,
            continuous_maxs=continuous_maxs,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        filled = df.copy()
        filled["Age"] = filled["Age"].fillna(self.age_fill)
        filled["Fare"] = filled["Fare"].fillna(self.fare_fill)
        filled["Embarked"] = filled["Embarked"].fillna(self.embarked_fill)

        binary = pd.DataFrame(
            {
                "Survived": filled["Survived"].astype(np.float32),
                "Sex": filled["Sex"].map({"male": 0.0, "female": 1.0}).astype(np.float32),
            }
        )

        pclass = pd.DataFrame(
            {
                f"Pclass_{category}": (filled["Pclass"] == category).astype(np.float32)
                for category in PCLASS_CATEGORIES
            }
        )
        embarked = pd.DataFrame(
            {
                f"Embarked_{category}": (filled["Embarked"] == category).astype(np.float32)
                for category in EMBARKED_CATEGORIES
            }
        )
        continuous = self.scaler.transform(filled[CONTINUOUS_COLUMNS]).astype(np.float32)

        encoded = np.concatenate(
            [binary.values, pclass.values, embarked.values, continuous],
            axis=1,
        ).astype(np.float32)
        return encoded

    def decode(self, decoded_logits: torch.Tensor) -> pd.DataFrame:
        decoded = decoded_logits.detach().cpu()

        binary_probs = torch.sigmoid(decoded[:, 0:2]).numpy()
        pclass_probs = torch.softmax(decoded[:, 2:5], dim=1).numpy()
        embarked_probs = torch.softmax(decoded[:, 5:8], dim=1).numpy()
        continuous_scaled = decoded[:, 8:12].numpy()

        continuous = self.scaler.inverse_transform(continuous_scaled)
        synthetic = pd.DataFrame(continuous, columns=CONTINUOUS_COLUMNS)

        for column in CONTINUOUS_COLUMNS:
            synthetic[column] = synthetic[column].clip(
                lower=self.continuous_mins[column],
                upper=self.continuous_maxs[column],
            )

        synthetic["Survived"] = (binary_probs[:, 0] >= 0.5).astype(int)
        synthetic["Sex"] = np.where(binary_probs[:, 1] >= 0.5, "female", "male")
        synthetic["Pclass"] = np.array(PCLASS_CATEGORIES)[pclass_probs.argmax(axis=1)]
        synthetic["Embarked"] = np.array(EMBARKED_CATEGORIES)[embarked_probs.argmax(axis=1)]

        synthetic["Age"] = synthetic["Age"].astype(np.float64).round(1)
        synthetic["Fare"] = synthetic["Fare"].astype(np.float64).round(4)
        synthetic["SibSp"] = synthetic["SibSp"].round().astype(int)
        synthetic["Parch"] = synthetic["Parch"].round().astype(int)

        synthetic["PassengerId"] = np.arange(1, len(synthetic) + 1, dtype=int)
        synthetic = synthetic[
            ["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        ]
        return synthetic


class TabularVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    continuous_weight: float,
) -> torch.Tensor:
    binary_loss = F.binary_cross_entropy_with_logits(
        reconstructed[:, 0:2], target[:, 0:2], reduction="sum"
    )
    pclass_loss = F.binary_cross_entropy_with_logits(
        reconstructed[:, 2:5], target[:, 2:5], reduction="sum"
    )
    embarked_loss = F.binary_cross_entropy_with_logits(
        reconstructed[:, 5:8], target[:, 5:8], reduction="sum"
    )
    continuous_loss = F.mse_loss(reconstructed[:, 8:12], target[:, 8:12], reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return binary_loss + pclass_loss + embarked_loss + continuous_weight * continuous_loss + beta * kl_divergence


def train_vae(
    features: np.ndarray,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    learning_rate: float,
    beta: float,
    device: torch.device,
) -> TabularVAE:
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TabularVAE(input_dim=features.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = vae_loss(
                reconstructed=reconstructed,
                target=batch,
                mu=mu,
                logvar=logvar,
                beta=beta,
                continuous_weight=1.0,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            average_loss = running_loss / len(dataset)
            print(f"epoch={epoch:4d} loss_per_row={average_loss:.4f}")

    return model


def sample_synthetic_rows(
    model: TabularVAE,
    preprocessor: TitanicPreprocessor,
    source_features: np.ndarray,
    num_samples: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        source_tensor = torch.tensor(source_features, dtype=torch.float32, device=device)
        mu, logvar = model.encode(source_tensor)
        sample_indices = torch.randint(0, mu.size(0), (num_samples,), device=device)
        sampled_mu = mu[sample_indices]
        sampled_std = torch.exp(0.5 * logvar[sample_indices])
        latent = sampled_mu + torch.randn_like(sampled_std) * sampled_std
        decoded = model.decode(latent)
    return preprocessor.decode(decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on Titanic data and sample synthetic rows.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("train.csv"),
        help="Path to the original Titanic train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("synthetic_titanic_vae_20000.csv"),
        help="Path to save the generated CSV",
    )
    parser.add_argument("--num-samples", type=int, default=20_000, help="Number of synthetic rows to generate")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--latent-dim", type=int, default=4, help="Latent dimension")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--beta", type=float, default=0.02, help="KL divergence weight")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw = pd.read_csv(args.input)
    train_df = raw[MODEL_COLUMNS].copy()
    preprocessor = TitanicPreprocessor.fit(train_df)
    features = preprocessor.transform(train_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training_device={device}")
    print(f"rows={len(train_df)} feature_dim={features.shape[1]}")

    model = train_vae(
        features=features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        beta=args.beta,
        device=device,
    )

    synthetic = sample_synthetic_rows(
        model=model,
        preprocessor=preprocessor,
        source_features=features,
        num_samples=args.num_samples,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(args.output, index=False)

    print(f"saved={args.output}")
    print(f"shape={synthetic.shape}")
    print(synthetic.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
