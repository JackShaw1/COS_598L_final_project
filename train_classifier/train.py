import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from dataset import PocketDataset
from model import PocketClassifier
from transformations_and_data_collect import (
    extract_pocket_features,
    extract_surface_pocket_features,
    random_rotate_sample
)

"""
This script trains an Imatinib binding pocket classifier using model.py,
dataset.py, and transformations.py.

Training logic:
1. Load original positive and negative pocket features.
2. Repeat positive samples to balance positives and negatives.
3. Randomly rotate every balanced sample N_TRANSFORMS times.
4. Train on all transformed samples.
"""

PROJECT_DIR = "train_classifier"
BATCH_SIZE = 15
N_EPOCHS = 30
LEARNING_RATE = 0.0005
N_FOLDS = 5
POSITIVE_REPEAT = 10
N_TRANSFORMS = 5
NEGATIVE_RANKING_SAMPLES = 100
RANDOM_SEED = 598

# Helpers
def read_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if row and row[0] != "file_path":
                rows.append(row)
    return rows

def full_path(relative_path):
    return os.path.join(PROJECT_DIR, relative_path)

def load_training_data():
    samples = []
    labels = []

    positives = read_csv(full_path("training_files/positives.csv"))
    negatives = read_csv(full_path("training_files/negatives.csv"))
    surface_negatives = read_csv(full_path("training_files/surface_negatives.csv"))

    # True Imatinib binding pockets
    for file_path, chain_id, residue_number in positives:
        features = extract_pocket_features(file_path, chain_id, residue_number)
        if features is not None:
            samples.append(features)
            labels.append(1)

    # Binding pockets of other small molecules (controls)
    for file_path, chain_id, residue_number in negatives:
        features = extract_pocket_features(file_path, chain_id, residue_number)
        if features is not None:
            samples.append(features)
            labels.append(0)

    # Random surface-exposed residues from proteins in the AlphaFold Database (controls)
    for file_path, chain_id, residue_number in surface_negatives:
        features = extract_surface_pocket_features(f"train_classifier/{file_path}", chain_id, residue_number)
        if features is not None:
            samples.append(features)
            labels.append(0)

    labels = np.array(labels, dtype=np.int64)
    # Print total number of positives and negatives from original files
    print(f"Loaded {labels.sum()} positives and {(labels == 0).sum()} negatives")
    return samples, labels

# Repeat positive samples to balance positive and negative classes
def repeat_positive_samples(samples, labels):
    repeated_samples = list(samples)
    repeated_labels = list(labels)
    for sample, label in zip(samples, labels):
        if label == 1:
            for _ in range(POSITIVE_REPEAT - 1):
                repeated_samples.append(sample)
                repeated_labels.append(1)
    repeated_labels = np.array(repeated_labels, dtype=np.int64)
    return repeated_samples, repeated_labels.tolist()

# Creates 5 augmented samples for each original training sample
def augment_samples(samples, labels, rng):
    augmented_samples = []
    augmented_labels = []
    for sample, label in zip(samples, labels):
        for _ in range(N_TRANSFORMS):
            augmented_samples.append(random_rotate_sample(sample, rng))
            augmented_labels.append(label)
    augmented_labels = np.array(augmented_labels, dtype=np.int64)
    return augmented_samples, augmented_labels.tolist()

# Creates complete dataloader
def make_loader(samples, labels, shuffle):
    dataset = PocketDataset(samples, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Loss --> backpropagation --> optimization
        optimizer.zero_grad()
        probabilities = model(x)
        loss = loss_function(probabilities, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

# Predict and collect scores for CV analysis
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probabilities = []
    all_labels = []

    for x, y in loader:
        all_probabilities.append(model(x.to(device)).cpu())
        all_labels.append(y)

    return torch.cat(all_labels).numpy(), torch.cat(all_probabilities).numpy()

# Score samples for ranking experiment
@torch.no_grad()
def score_samples(model, samples, device):
    dummy_labels = [0] * len(samples)
    loader = make_loader(samples, dummy_labels, shuffle=False)
    _, probabilities = predict(model, loader, device)
    return np.ravel(probabilities)

# Calculate performance metrics based on classifier probabilities
def evaluate(model, loader, device):
    labels, probabilities = predict(model, loader, device)
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    return accuracy, auc

# Function for training model and loading necessary components
def train_model(train_samples, train_labels, device, seed):
    rng = np.random.default_rng(seed)

    train_samples, train_labels = repeat_positive_samples(train_samples, train_labels)
    train_samples, train_labels = augment_samples(train_samples, train_labels, rng)

    train_loader = make_loader(train_samples, train_labels, shuffle=True)

    model = PocketClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCELoss()

    for epoch in range(1, N_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        if epoch % 2 == 0:
            # Print loss after every other epoch
            print(f"epoch {epoch}/{N_EPOCHS} loss={loss:.4f}")
    return model

# Logic for 80/20 train/test cross-validation analysis
def cross_validate(samples, labels, device):
    # Retain class balance
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = []
    roc_data = []
    ranking_data = []

    for fold, (train_indices, val_indices) in enumerate(splitter.split(samples, labels), start=1):
        print(f"\nFold {fold}/{N_FOLDS}")

        train_samples = [samples[i] for i in train_indices]
        train_labels = labels[train_indices].tolist()

        val_samples = [samples[i] for i in val_indices]
        val_labels = labels[val_indices].tolist()

        model = train_model(
            train_samples,
            train_labels,
            device,
            seed=RANDOM_SEED + fold,
        )

        val_loader = make_loader(val_samples, val_labels, shuffle=False)

        accuracy, auc = evaluate(model, val_loader, device)
        print(f"validation acc={accuracy:.3f} AUC={auc:.3f}")

        results.append((accuracy, auc))
        roc_data.append(predict(model, val_loader, device))

        val_positive_samples = [samples[i] for i in val_indices if labels[i] == 1]

        negative_indices = np.where(labels == 0)[0]
        rng = np.random.default_rng(RANDOM_SEED + 1000 + fold)
        if len(negative_indices) > NEGATIVE_RANKING_SAMPLES:
            ranking_negative_indices = rng.choice(
                negative_indices,
                size=NEGATIVE_RANKING_SAMPLES,
                replace=False
            )
        else:
            ranking_negative_indices = negative_indices

        ranking_negative_samples = [samples[i] for i in ranking_negative_indices]

        positive_scores = score_samples(model, val_positive_samples, device)
        negative_scores = score_samples(model, ranking_negative_samples, device)

        print(f"ranking positives={len(positive_scores)} negatives={len(negative_scores)}")
        ranking_data.append((positive_scores, negative_scores))

    return results, roc_data, ranking_data

# Print function for testing results
def print_summary(results):
    results = np.array(results)
    print("\nCross-validation summary")
    print(f"accuracy: {results[:, 0].mean():.3f} +/- {results[:, 0].std():.3f}")
    print(f"AUC: {results[:, 1].mean():.3f} +/- {results[:, 1].std():.3f}")

# Create cross-validation results output
def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(6, 6))

    for fold, (labels, probabilities) in enumerate(roc_data, start=1):
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        ax.plot(fpr, tpr, label=f"Fold {fold} AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)

# Create ranking experiment results output
def plot_ranking_experiments(ranking_data):
    for fold, (positive_scores, negative_scores) in enumerate(ranking_data, start=1):
        sorted_negative_scores = np.sort(negative_scores)[::-1]
        negative_ranks = np.arange(1, len(sorted_negative_scores) + 1)

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            negative_ranks,
            sorted_negative_scores,
            marker="o",
            markersize=3,
            linewidth=1,
            label="Negative controls"
        )

        positive_ranks = [1 + np.sum(negative_scores > positive_score) for positive_score in positive_scores]
        rank_to_indices = {}
        for i, positive_rank in enumerate(positive_ranks):
            if positive_rank not in rank_to_indices:
                rank_to_indices[positive_rank] = []
            rank_to_indices[positive_rank].append(i)

        x_offsets = np.zeros(len(positive_scores))
        for positive_rank, indices in rank_to_indices.items():
            if len(indices) == 1:
                x_offsets[indices[0]] = 0.0
            else:
                offsets = np.linspace(-1.0, 1.0, len(indices))
                for idx, offset in zip(indices, offsets):
                    x_offsets[idx] = offset

        for i, positive_score in enumerate(positive_scores, start=1):
            positive_rank = positive_ranks[i - 1]
            x_pos = positive_rank + x_offsets[i - 1]
            ax.scatter(
                x_pos,
                positive_score,
                s=180,
                marker="*",
                edgecolors="black",
                linewidths=1.0,
                zorder=5
            )
            ax.text(
                x_pos + 0.6,
                positive_score,
                f"P{i}",
                fontsize=8,
                va="center",
                ha="left"
            )

        ax.set_xlabel("Rank among negative controls")
        ax.set_ylabel("Classifier probability")
        ax.set_title(f"Fold {fold} ranking experiment")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PROJECT_DIR, f"ranking_experiment_fold_{fold}.png"), dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Change to cuda to train with CPUs if they're available. 
    # For our case, we just used CPUs since we had so few training samples.
    device = torch.device("cpu")

    samples, labels = load_training_data()

    results, roc_data, ranking_data = cross_validate(samples, labels, device)
    print_summary(results)
    plot_roc_curves(roc_data)
    plot_ranking_experiments(ranking_data)

    # Train final model on entirety of training data.
    # We used this model for subsequent searching throughout the human proteome (AlphaFold Database) for potential Imatinib binding pockets.
    print("\nTraining final model")
    final_model = train_model(
        samples,
        labels.tolist(),
        device,
        seed=RANDOM_SEED,
    )

    # Save weights from final model
    torch.save(final_model.state_dict(), os.path.join(PROJECT_DIR, "pocket_classifier.pt"))