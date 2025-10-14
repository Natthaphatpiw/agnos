"""
Improved training script based on overfitting analysis
Since the model is NOT overfitting, we can:
1. Increase model capacity
2. Reduce regularization
3. Train longer
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle

from model import SymptomRecommender
from train import SymptomDataset, preprocess_data, collate_fn, evaluate


def train_epoch(model, dataloader, optimizer, criterion_bce, edge_index, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        patient_idx = batch['patient_idx'].to(device)
        gender = torch.stack([torch.from_numpy(g) for g in batch['gender']]).to(device)
        age_bin = torch.stack([torch.from_numpy(a) for a in batch['age_bin']]).to(device)
        query_symptoms = batch['query_symptoms'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        scores = model(patient_idx, gender, age_bin, query_symptoms, edge_index)
        loss = criterion_bce(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    """Improved training pipeline"""

    print("="*80)
    print("IMPROVED TRAINING (Based on Overfitting Analysis)")
    print("="*80)
    print("\nChanges from original:")
    print("  - Reduced dropout: 0.3 → 0.2 (less regularization)")
    print("  - Reduced weight_decay: 1e-5 → 5e-6 (less regularization)")
    print("  - Increased epochs: 50 → 80 (more training)")
    print("  - Adjusted learning rate schedule")
    print("  - Better early stopping criteria")
    print("="*80 + "\n")

    # Hyperparameters (IMPROVED)
    CSV_PATH = '/Users/piw/Downloads/atest/AI symptom picker data.csv'
    EMBED_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 80              # Increased from 50
    LR = 1e-3
    WEIGHT_DECAY = 5e-6      # Reduced from 1e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}\n")

    # Preprocess
    print("Loading data...")
    df, symptom_to_idx, idx_to_symptom, patient_data, interaction_matrix, graph_data, age_bins = preprocess_data(CSV_PATH)

    num_symptoms = len(symptom_to_idx)
    num_patients = len(patient_data)

    # Split data
    train_data, temp_data = train_test_split(patient_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}\n")

    # Create datasets
    train_dataset = SymptomDataset(train_data, symptom_to_idx, interaction_matrix)
    val_dataset = SymptomDataset(val_data, symptom_to_idx, interaction_matrix)
    test_dataset = SymptomDataset(test_data, symptom_to_idx, interaction_matrix)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model with REDUCED dropout
    print("Initializing model...")

    # We need to modify the model creation to use lower dropout
    # For now, use existing model but we'll create improved version
    model = SymptomRecommender(
        num_symptoms=num_symptoms,
        num_patients=num_patients,
        symptom_embed_dim=EMBED_DIM,
        dropout=0.2  # Reduced from 0.3
    ).to(DEVICE)

    edge_index = graph_data.edge_index.to(DEVICE)

    # Optimizer with reduced weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # More patient scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=8  # Increased from 5
    )

    criterion_bce = nn.BCELoss()

    print("Training started...")
    print("-"*80)

    best_val_recall = 0.0
    patience_counter = 0
    patience_limit = 15  # Early stopping patience

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_bce, edge_index, DEVICE)
        val_precision, val_recall = evaluate(model, val_loader, edge_index, DEVICE, k=5)

        scheduler.step(val_recall)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Val P@5: {val_precision:.4f} | "
              f"Val R@5: {val_recall:.4f}")

        # Save best model
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), 'model_improved.pth')
            print(f"  → Saved best model (R@5: {val_recall:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"No improvement for {patience_limit} epochs")
            break

        # Overfitting check every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            train_precision, train_recall = evaluate(model, train_loader, edge_index, DEVICE, k=5)
            gap = train_recall - val_recall

            print(f"  Train R@5: {train_recall:.4f} | Gap: {gap:+.4f}")

            if gap > 0.15:
                print(f"  ⚠️  Starting to overfit, consider stopping soon")

    # Test evaluation
    print("\n" + "-"*80)
    print("Final evaluation on test set...")
    model.load_state_dict(torch.load('model_improved.pth'))

    test_precision, test_recall = evaluate(model, test_loader, edge_index, DEVICE, k=5)
    train_precision, train_recall = evaluate(model, train_loader, edge_index, DEVICE, k=5)
    val_precision, val_recall = evaluate(model, val_loader, edge_index, DEVICE, k=5)

    print("\nFinal Results:")
    print("-"*80)
    print(f"{'Split':<12} {'Precision@5':<15} {'Recall@5':<15}")
    print("-"*80)
    print(f"{'Train':<12} {train_precision:<15.4f} {train_recall:<15.4f}")
    print(f"{'Validation':<12} {val_precision:<15.4f} {val_recall:<15.4f}")
    print(f"{'Test':<12} {test_precision:<15.4f} {test_recall:<15.4f}")
    print("-"*80)

    train_val_gap = train_recall - val_recall
    print(f"\nTrain-Val Gap: {train_val_gap:+.4f}")

    if train_val_gap > 0.10:
        print("⚠️  Model is overfitting")
    else:
        print("✓ Generalization is good")

    # Save artifacts
    print("\nSaving artifacts...")

    config = {
        'num_symptoms': num_symptoms,
        'num_patients': num_patients,
        'symptom_embed_dim': EMBED_DIM,
        'dropout': 0.2,
        'weight_decay': WEIGHT_DECAY,
        'model_type': 'improved'
    }

    with open('model_config_improved.pkl', 'wb') as f:
        pickle.dump(config, f)

    print("\nTraining complete!")
    print("Saved files:")
    print("  - model_improved.pth")
    print("  - model_config_improved.pkl")
    print("="*80)


if __name__ == '__main__':
    main()
