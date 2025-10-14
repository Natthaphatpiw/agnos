"""
Training script for lightweight model
Use this if overfitting is detected
"""

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict

from model_lightweight import LightweightSymptomRecommender, count_parameters
from train import SymptomDataset, preprocess_data, collate_fn


def train_epoch(model, dataloader, optimizer, criterion_bce, device):
    """Train for one epoch - no edge_index needed"""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        patient_idx = batch['patient_idx'].to(device)
        gender = torch.stack([torch.from_numpy(g) for g in batch['gender']]).to(device)
        age_bin = torch.stack([torch.from_numpy(a) for a in batch['age_bin']]).to(device)
        query_symptoms = batch['query_symptoms'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward (no edge_index needed)
        scores = model(patient_idx, gender, age_bin, query_symptoms)

        # BCE loss
        loss = criterion_bce(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, k=5):
    """Evaluate model - no edge_index needed"""
    model.eval()
    all_precisions = []
    all_recalls = []

    with torch.no_grad():
        for batch in dataloader:
            patient_idx = batch['patient_idx'].to(device)
            gender = torch.stack([torch.from_numpy(g) for g in batch['gender']]).to(device)
            age_bin = torch.stack([torch.from_numpy(a) for a in batch['age_bin']]).to(device)
            query_symptoms = batch['query_symptoms'].to(device)
            labels = batch['labels'].to(device)

            scores = model(patient_idx, gender, age_bin, query_symptoms)

            for i in range(len(scores)):
                score = scores[i]
                label = labels[i]

                # Mask query symptoms
                query_mask = torch.zeros_like(score, dtype=torch.bool)
                query_mask[query_symptoms[i]] = True
                score = score.masked_fill(query_mask, -float('inf'))

                top_k_indices = torch.topk(score, k=k).indices.cpu().numpy()
                true_indices = torch.where(label > 0)[0].cpu().numpy()

                if len(true_indices) > 0:
                    pred_set = set(top_k_indices)
                    true_set = set(true_indices)

                    tp = len(pred_set & true_set)
                    precision = tp / k if k > 0 else 0
                    recall = tp / len(true_set) if len(true_set) > 0 else 0

                    all_precisions.append(precision)
                    all_recalls.append(recall)

    avg_precision = np.mean(all_precisions) if all_precisions else 0
    avg_recall = np.mean(all_recalls) if all_recalls else 0

    return avg_precision, avg_recall


def main():
    """Main training pipeline for lightweight model"""

    # Hyperparameters (adjusted for small dataset)
    CSV_PATH = '/Users/piw/Downloads/atest/AI symptom picker data.csv'
    EMBED_DIM = 32        # Reduced from 128
    BATCH_SIZE = 32
    EPOCHS = 30           # Reduced from 50
    LR = 5e-4             # Lower learning rate
    WEIGHT_DECAY = 1e-4   # Higher weight decay
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("LIGHTWEIGHT MODEL TRAINING")
    print("="*80)
    print(f"Using device: {DEVICE}")
    print(f"Embedding dimension: {EMBED_DIM} (vs 128 in original)")
    print(f"Epochs: {EPOCHS} (vs 50 in original)")
    print(f"Weight decay: {WEIGHT_DECAY} (vs 1e-5 in original)")

    # Preprocess
    print("\nPreprocessing data...")
    df, symptom_to_idx, idx_to_symptom, patient_data, interaction_matrix, graph_data, age_bins = preprocess_data(CSV_PATH)

    num_symptoms = len(symptom_to_idx)
    num_patients = len(patient_data)

    print(f"  Vocabulary size: {num_symptoms}")
    print(f"  Number of patients: {num_patients}")

    # Split data
    train_data, temp_data = train_test_split(patient_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets
    train_dataset = SymptomDataset(train_data, symptom_to_idx, interaction_matrix)
    val_dataset = SymptomDataset(val_data, symptom_to_idx, interaction_matrix)
    test_dataset = SymptomDataset(test_data, symptom_to_idx, interaction_matrix)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize lightweight model
    print("\nInitializing lightweight model...")
    model = LightweightSymptomRecommender(
        num_symptoms=num_symptoms,
        num_patients=num_patients,
        symptom_embed_dim=EMBED_DIM
    ).to(DEVICE)

    total_params = count_parameters(model)
    params_per_sample = total_params / len(train_data)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameters per sample: {params_per_sample:.1f}")

    if params_per_sample > 100:
        print(f"  ⚠️  Warning: {params_per_sample:.1f} params/sample is high")
        print(f"     Recommended: < 100 for datasets with {len(train_data)} samples")
    else:
        print(f"  ✓ Good ratio for small dataset")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    criterion_bce = nn.BCELoss()

    print("\nStarting training...")
    print("-"*80)
    best_val_recall = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_bce, DEVICE)
        val_precision, val_recall = evaluate(model, val_loader, DEVICE, k=5)

        scheduler.step(val_recall)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Val P@5: {val_precision:.4f} | "
              f"Val R@5: {val_recall:.4f}")

        # Save best model
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), 'model_lightweight.pth')
            print(f"  → Saved best model (R@5: {val_recall:.4f})")

        # Check for overfitting every 5 epochs
        if epoch % 5 == 0 and epoch > 0:
            train_precision, train_recall = evaluate(model, train_loader, DEVICE, k=5)
            gap = train_recall - val_recall

            if gap > 0.15:
                print(f"  ⚠️  Overfitting detected (gap: {gap:.4f})")
            elif gap > 0.10:
                print(f"  ⚠️  Moderate overfitting (gap: {gap:.4f})")

    # Test evaluation
    print("\n" + "-"*80)
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('model_lightweight.pth'))
    test_precision, test_recall = evaluate(model, test_loader, DEVICE, k=5)
    print(f"Test P@5: {test_precision:.4f} | Test R@5: {test_recall:.4f}")

    # Save artifacts
    print("\nSaving artifacts...")

    with open('symptom_to_idx.pkl', 'wb') as f:
        pickle.dump(symptom_to_idx, f)

    with open('idx_to_symptom.pkl', 'wb') as f:
        pickle.dump(idx_to_symptom, f)

    with open('age_bins.pkl', 'wb') as f:
        pickle.dump(age_bins, f)

    # Save model config for lightweight model
    config = {
        'num_symptoms': num_symptoms,
        'num_patients': num_patients,
        'symptom_embed_dim': EMBED_DIM,
        'model_type': 'lightweight'
    }

    with open('model_config_lightweight.pkl', 'wb') as f:
        pickle.dump(config, f)

    print("Training complete! Artifacts saved:")
    print("  - model_lightweight.pth")
    print("  - model_config_lightweight.pkl")
    print("  - symptom_to_idx.pkl")
    print("  - idx_to_symptom.pkl")
    print("  - age_bins.pkl")
    print("="*80)


if __name__ == '__main__':
    main()
