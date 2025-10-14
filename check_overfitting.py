"""
Script to check if the trained model is overfitting
Run this after training to evaluate model performance
"""

import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import SymptomRecommender
from train import SymptomDataset, preprocess_data, evaluate, collate_fn


def detailed_evaluation(model, dataloader, edge_index, device, split_name="Test"):
    """Detailed evaluation with multiple metrics"""
    model.eval()

    all_precisions_k5 = []
    all_recalls_k5 = []
    all_precisions_k10 = []
    all_recalls_k10 = []

    with torch.no_grad():
        for batch in dataloader:
            patient_idx = batch['patient_idx'].to(device)
            gender = torch.stack([torch.from_numpy(g) for g in batch['gender']]).to(device)
            age_bin = torch.stack([torch.from_numpy(a) for a in batch['age_bin']]).to(device)
            query_symptoms = batch['query_symptoms'].to(device)
            labels = batch['labels'].to(device)

            scores = model(patient_idx, gender, age_bin, query_symptoms, edge_index)

            for i in range(len(scores)):
                score = scores[i]
                label = labels[i]

                # Mask query symptoms
                query_mask = torch.zeros_like(score, dtype=torch.bool)
                query_mask[query_symptoms[i]] = True
                score = score.masked_fill(query_mask, -float('inf'))

                true_indices = torch.where(label > 0)[0].cpu().numpy()

                if len(true_indices) > 0:
                    # Precision@5 and Recall@5
                    top_k5_indices = torch.topk(score, k=5).indices.cpu().numpy()
                    pred_set_k5 = set(top_k5_indices)
                    true_set = set(true_indices)

                    tp_k5 = len(pred_set_k5 & true_set)
                    precision_k5 = tp_k5 / 5
                    recall_k5 = tp_k5 / len(true_set)

                    all_precisions_k5.append(precision_k5)
                    all_recalls_k5.append(recall_k5)

                    # Precision@10 and Recall@10
                    top_k10_indices = torch.topk(score, k=10).indices.cpu().numpy()
                    pred_set_k10 = set(top_k10_indices)

                    tp_k10 = len(pred_set_k10 & true_set)
                    precision_k10 = tp_k10 / 10
                    recall_k10 = tp_k10 / len(true_set)

                    all_precisions_k10.append(precision_k10)
                    all_recalls_k10.append(recall_k10)

    results = {
        'precision@5': np.mean(all_precisions_k5) if all_precisions_k5 else 0,
        'recall@5': np.mean(all_recalls_k5) if all_recalls_k5 else 0,
        'precision@10': np.mean(all_precisions_k10) if all_precisions_k10 else 0,
        'recall@10': np.mean(all_recalls_k10) if all_recalls_k10 else 0,
    }

    return results


def calculate_overfitting_metrics(train_results, val_results, test_results):
    """Calculate overfitting indicators"""

    # Gap between train and validation
    train_val_gap_p5 = train_results['precision@5'] - val_results['precision@5']
    train_val_gap_r5 = train_results['recall@5'] - val_results['recall@5']

    # Gap between train and test
    train_test_gap_p5 = train_results['precision@5'] - test_results['precision@5']
    train_test_gap_r5 = train_results['recall@5'] - test_results['recall@5']

    # Generalization gap (val vs test)
    val_test_gap_p5 = val_results['precision@5'] - test_results['precision@5']
    val_test_gap_r5 = val_results['recall@5'] - test_results['recall@5']

    return {
        'train_val_gap_p5': train_val_gap_p5,
        'train_val_gap_r5': train_val_gap_r5,
        'train_test_gap_p5': train_test_gap_p5,
        'train_test_gap_r5': train_test_gap_r5,
        'val_test_gap_p5': val_test_gap_p5,
        'val_test_gap_r5': val_test_gap_r5,
    }


def analyze_overfitting(gaps, train_results, val_results, test_results):
    """Analyze if model is overfitting and provide recommendations"""

    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS")
    print("="*80)

    print("\n1. Performance Metrics:")
    print("-" * 80)
    print(f"{'Split':<15} {'Precision@5':<15} {'Recall@5':<15} {'Precision@10':<15} {'Recall@10':<15}")
    print("-" * 80)
    print(f"{'Train':<15} {train_results['precision@5']:<15.4f} {train_results['recall@5']:<15.4f} "
          f"{train_results['precision@10']:<15.4f} {train_results['recall@10']:<15.4f}")
    print(f"{'Validation':<15} {val_results['precision@5']:<15.4f} {val_results['recall@5']:<15.4f} "
          f"{val_results['precision@10']:<15.4f} {val_results['recall@10']:<15.4f}")
    print(f"{'Test':<15} {test_results['precision@5']:<15.4f} {test_results['recall@5']:<15.4f} "
          f"{test_results['precision@10']:<15.4f} {test_results['recall@10']:<15.4f}")
    print("-" * 80)

    print("\n2. Performance Gaps (Overfitting Indicators):")
    print("-" * 80)
    print(f"Train vs Validation:")
    print(f"  Precision@5 Gap: {gaps['train_val_gap_p5']:+.4f}")
    print(f"  Recall@5 Gap:    {gaps['train_val_gap_r5']:+.4f}")
    print(f"\nTrain vs Test:")
    print(f"  Precision@5 Gap: {gaps['train_test_gap_p5']:+.4f}")
    print(f"  Recall@5 Gap:    {gaps['train_test_gap_r5']:+.4f}")
    print(f"\nValidation vs Test:")
    print(f"  Precision@5 Gap: {gaps['val_test_gap_p5']:+.4f}")
    print(f"  Recall@5 Gap:    {gaps['val_test_gap_r5']:+.4f}")
    print("-" * 80)

    # Overfitting criteria
    SEVERE_OVERFIT_THRESHOLD = 0.20  # Gap > 20%
    MODERATE_OVERFIT_THRESHOLD = 0.10  # Gap > 10%

    max_gap = max(gaps['train_val_gap_p5'], gaps['train_val_gap_r5'])

    print("\n3. Diagnosis:")
    print("-" * 80)

    is_overfitting = False
    severity = "None"

    if max_gap > SEVERE_OVERFIT_THRESHOLD:
        is_overfitting = True
        severity = "SEVERE"
        print(f"⚠️  SEVERE OVERFITTING DETECTED!")
        print(f"   Maximum gap: {max_gap:.1%} (threshold: {SEVERE_OVERFIT_THRESHOLD:.1%})")
        print(f"\n   The model is memorizing training data instead of learning patterns.")
        print(f"   Performance on unseen data is significantly worse.")

    elif max_gap > MODERATE_OVERFIT_THRESHOLD:
        is_overfitting = True
        severity = "MODERATE"
        print(f"⚠️  Moderate Overfitting Detected")
        print(f"   Maximum gap: {max_gap:.1%} (threshold: {MODERATE_OVERFIT_THRESHOLD:.1%})")
        print(f"\n   The model shows some signs of overfitting but still generalizes reasonably.")

    else:
        print(f"✓ Model Generalization is Good")
        print(f"   Maximum gap: {max_gap:.1%} (threshold: {MODERATE_OVERFIT_THRESHOLD:.1%})")
        print(f"\n   The model generalizes well to unseen data.")

    print("-" * 80)

    # Additional checks
    print("\n4. Additional Checks:")
    print("-" * 80)

    # Check if validation is much worse than test
    if val_results['recall@5'] < test_results['recall@5'] - 0.05:
        print("⚠️  Validation performance is worse than test performance")
        print("   This might indicate issues with data split or small validation set")

    # Check if all metrics are very low
    if test_results['recall@5'] < 0.30:
        print("⚠️  Overall performance is low (Recall@5 < 30%)")
        print("   Model might not be learning useful patterns")

    # Check if train performance is suspiciously high
    if train_results['recall@5'] > 0.90:
        print("⚠️  Training performance is very high (Recall@5 > 90%)")
        print("   Strong indication of overfitting")

    print("-" * 80)

    return is_overfitting, severity, max_gap


def provide_recommendations(is_overfitting, severity, max_gap):
    """Provide actionable recommendations"""

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if not is_overfitting:
        print("\n✓ Current architecture is working well for this dataset.")
        print("  You can continue using the current model.")
        print("\nOptional improvements:")
        print("  1. Collect more data to improve performance further")
        print("  2. Fine-tune hyperparameters (learning rate, weight decay)")
        print("  3. Experiment with different fusion weights")

    elif severity == "MODERATE":
        print("\n⚠️  Model shows moderate overfitting. Consider these improvements:")
        print("\n1. Increase Regularization:")
        print("   - Increase dropout from 0.3 to 0.5")
        print("   - Increase weight_decay from 1e-5 to 1e-4")
        print("   - Add dropout to embedding layers")

        print("\n2. Reduce Model Complexity:")
        print("   - Reduce embedding dimension from 128 to 64")
        print("   - Use 2 MLP layers instead of 4")
        print("   - Reduce GAT hidden dimensions")

        print("\n3. Training Adjustments:")
        print("   - Reduce number of epochs from 50 to 30")
        print("   - Use early stopping with patience=5")
        print("   - Increase learning rate decay")

    else:  # SEVERE
        print("\n⚠️  SEVERE overfitting detected. Recommend switching to simpler architecture:")
        print("\nOption 1: Lightweight Neural Network (RECOMMENDED)")
        print("   - Use simple Matrix Factorization instead of NCF")
        print("   - Remove GAT, use only embeddings + cosine similarity")
        print("   - Embedding dim: 32 instead of 128")
        print("   - Single linear layer instead of MLP")

        print("\nOption 2: Traditional Machine Learning")
        print("   - SVD (Truncated Singular Value Decomposition)")
        print("   - K-Nearest Neighbors on symptom co-occurrence")
        print("   - Item-based Collaborative Filtering")

        print("\nOption 3: Hybrid Lightweight Approach")
        print("   - Content-Based: Cosine Similarity (no training)")
        print("   - Collaborative: Simple Matrix Factorization")
        print("   - Graph: PageRank or Random Walk (no training)")

    print("\n" + "="*80)


def main():
    """Main execution"""
    import sys
    import os

    # Configuration
    CSV_PATH = '/Users/piw/Downloads/atest/AI symptom picker data.csv'
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check which model to use
    use_improved = False
    if os.path.exists('model_improved.pth') and os.path.exists('model_config_improved.pkl'):
        use_improved = True
        model_path = 'model_improved.pth'
        config_path = 'model_config_improved.pkl'
        model_name = "IMPROVED MODEL"
    else:
        model_path = 'model.pth'
        config_path = 'model_config.pkl'
        model_name = "ORIGINAL MODEL"

    print("="*80)
    print("OVERFITTING DETECTION AND ANALYSIS")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Device: {DEVICE}")
    print(f"Data: {CSV_PATH}")

    # Load preprocessed data
    print("\nLoading and preprocessing data...")
    df, symptom_to_idx, idx_to_symptom, patient_data, interaction_matrix, graph_data, age_bins = preprocess_data(CSV_PATH)

    num_symptoms = len(symptom_to_idx)
    num_patients = len(patient_data)

    print(f"  Vocabulary size: {num_symptoms}")
    print(f"  Number of patients: {num_patients}")

    # Split data (same as training)
    train_data, temp_data = train_test_split(patient_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets
    train_dataset = SymptomDataset(train_data, symptom_to_idx, interaction_matrix)
    val_dataset = SymptomDataset(val_data, symptom_to_idx, interaction_matrix)
    test_dataset = SymptomDataset(test_data, symptom_to_idx, interaction_matrix)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Load model
    print(f"\nLoading trained model from {model_path}...")

    try:
        with open(config_path, 'rb') as f:
            model_config = pickle.load(f)

        model = SymptomRecommender(
            num_symptoms=model_config['num_symptoms'],
            num_patients=model_config['num_patients'],
            symptom_embed_dim=model_config['symptom_embed_dim']
        ).to(DEVICE)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        print("  ✓ Model loaded successfully")

    except FileNotFoundError:
        print("\n  ✗ Error: Model files not found!")
        if use_improved:
            print("  Improved model files not found. Please run 'python train_improved.py' first.")
        else:
            print("  Please run 'python train.py' first to train the model.")
        return

    edge_index = graph_data.edge_index.to(DEVICE)

    # Evaluate on all splits
    print("\nEvaluating model performance...")
    print("  (This may take a few minutes...)")

    train_results = detailed_evaluation(model, train_loader, edge_index, DEVICE, "Train")
    val_results = detailed_evaluation(model, val_loader, edge_index, DEVICE, "Validation")
    test_results = detailed_evaluation(model, test_loader, edge_index, DEVICE, "Test")

    # Calculate gaps
    gaps = calculate_overfitting_metrics(train_results, val_results, test_results)

    # Analyze overfitting
    is_overfitting, severity, max_gap = analyze_overfitting(gaps, train_results, val_results, test_results)

    # Provide recommendations
    provide_recommendations(is_overfitting, severity, max_gap)

    # Save results
    results_summary = {
        'model_type': model_name,
        'train': train_results,
        'validation': val_results,
        'test': test_results,
        'gaps': gaps,
        'is_overfitting': is_overfitting,
        'severity': severity,
        'max_gap': max_gap
    }

    output_file = 'overfitting_analysis_improved.pkl' if use_improved else 'overfitting_analysis.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results_summary, f)

    print(f"\n✓ Analysis complete. Results saved to '{output_file}'")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
