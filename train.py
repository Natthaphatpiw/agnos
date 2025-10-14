import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pickle
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from model import SymptomRecommender


class SymptomDataset(Dataset):
    """Dataset for symptom recommendation training"""

    def __init__(
        self,
        patient_data: List[Dict],
        symptom_to_idx: Dict[str, int],
        interaction_matrix: np.ndarray,
        max_query_len: int = 10
    ):
        self.data = patient_data
        self.symptom_to_idx = symptom_to_idx
        self.interaction_matrix = interaction_matrix
        self.max_query_len = max_query_len
        self.num_symptoms = len(symptom_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get positive and negative symptoms
        pos_symptoms = item['yes_symptoms']
        neg_symptoms = item['no_symptoms']

        # Random split: some pos symptoms as query, others as targets
        if len(pos_symptoms) > 1:
            split_idx = max(1, len(pos_symptoms) // 2)
            query_symptoms = pos_symptoms[:split_idx]
            target_symptoms = pos_symptoms[split_idx:]
        else:
            query_symptoms = pos_symptoms
            target_symptoms = pos_symptoms

        # Encode
        query_indices = [self.symptom_to_idx[s] for s in query_symptoms if s in self.symptom_to_idx][:self.max_query_len]
        target_indices = [self.symptom_to_idx[s] for s in target_symptoms if s in self.symptom_to_idx]
        neg_indices = [self.symptom_to_idx[s] for s in neg_symptoms if s in self.symptom_to_idx][:len(target_indices) * 2]

        # Pad query
        query_indices += [0] * (self.max_query_len - len(query_indices))

        # Create labels: 1 for positive, 0 for negative
        labels = torch.zeros(self.num_symptoms)
        for idx in target_indices:
            labels[idx] = 1.0

        return {
            'patient_idx': idx,
            'gender': item['gender_vec'],
            'age_bin': item['age_bin_vec'],
            'query_symptoms': torch.tensor(query_indices, dtype=torch.long),
            'target_symptoms': torch.tensor(target_indices, dtype=torch.long),
            'neg_symptoms': torch.tensor(neg_indices, dtype=torch.long) if neg_indices else torch.tensor([0], dtype=torch.long),
            'labels': labels
        }


def parse_severity(answers: List[str]) -> float:
    """Parse severity from answer strings"""
    severity_map = {
        'ปวดจนไม่สามารถทำงานได้': 3.0,
        'ปวดจนไม่สามารถทำกิจกรรมใดใดได้เลย': 3.0,
        'ปวดจนไม่สามารถ': 3.0,
        'ปวดมาก': 2.5,
        'ปวดปานกลาง': 2.0,
        'ส่งผลต่อการดำเนินกิจวัตรประจำวันบ้าง': 1.5,
        'ปวดเล็กน้อย': 1.0,
        'เล็กน้อย': 1.0,
    }

    text = ' '.join(answers)
    for pattern, score in severity_map.items():
        if pattern in text:
            return score

    return 1.0  # Default weight


def preprocess_data(csv_path: str) -> Tuple:
    """
    Load and preprocess data
    Returns:
        df, symptom_to_idx, idx_to_symptom, patient_data, interaction_matrix, graph_data, age_bins
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")

    # Parse JSON summaries
    print("Parsing summaries...")
    all_symptoms = set()
    patient_data = []

    for idx, row in df.iterrows():
        try:
            summary = json.loads(row['summary'])
        except:
            continue

        # Extract symptoms
        yes_symptoms = []
        no_symptoms = []
        symptom_details = {}

        for sym_obj in summary.get('yes_symptoms', []):
            symptom = sym_obj.get('text', '').strip()
            if symptom and symptom != 'การรักษาก่อนหน้า' and symptom != 'Previous treatment':
                yes_symptoms.append(symptom)
                all_symptoms.add(symptom)
                answers = sym_obj.get('answers', [])
                symptom_details[symptom] = {
                    'answers': answers,
                    'severity': parse_severity(answers)
                }

        for sym_obj in summary.get('no_symptoms', []):
            symptom = sym_obj.get('text', '').strip()
            if symptom:
                no_symptoms.append(symptom)
                all_symptoms.add(symptom)

        # Add from search_term
        if pd.notna(row['search_term']):
            for sym in str(row['search_term']).split(','):
                sym = sym.strip()
                if sym:
                    all_symptoms.add(sym)
                    if sym not in yes_symptoms:
                        yes_symptoms.append(sym)
                        if sym not in symptom_details:
                            symptom_details[sym] = {'answers': [], 'severity': 1.0}

        patient_data.append({
            'patient_idx': idx,
            'gender': row['gender'].lower(),
            'age': row['age'],
            'yes_symptoms': yes_symptoms,
            'no_symptoms': no_symptoms,
            'symptom_details': symptom_details
        })

    # Build vocabulary
    print(f"Building vocabulary from {len(all_symptoms)} unique symptoms...")
    symptom_to_idx = {sym: idx for idx, sym in enumerate(sorted(all_symptoms))}
    idx_to_symptom = {idx: sym for sym, idx in symptom_to_idx.items()}
    num_symptoms = len(symptom_to_idx)
    num_patients = len(patient_data)

    print(f"Vocabulary size: {num_symptoms}")
    print(f"Number of patients: {num_patients}")

    # Build interaction matrix
    print("Building interaction matrix...")
    interaction_matrix = np.zeros((num_patients, num_symptoms), dtype=np.float32)

    for patient in patient_data:
        p_idx = patient['patient_idx']
        for sym in patient['yes_symptoms']:
            if sym in symptom_to_idx:
                s_idx = symptom_to_idx[sym]
                severity = patient['symptom_details'].get(sym, {}).get('severity', 1.0)
                interaction_matrix[p_idx, s_idx] = severity

        for sym in patient['no_symptoms']:
            if sym in symptom_to_idx:
                s_idx = symptom_to_idx[sym]
                interaction_matrix[p_idx, s_idx] = -1.0

    # Build co-occurrence graph
    print("Building co-occurrence graph...")
    G = nx.Graph()
    G.add_nodes_from(range(num_symptoms))

    edge_weights = defaultdict(float)

    for patient in patient_data:
        symptoms = patient['yes_symptoms']
        symptom_indices = [symptom_to_idx[s] for s in symptoms if s in symptom_to_idx]

        # Add edges between co-occurring symptoms
        for i in range(len(symptom_indices)):
            for j in range(i + 1, len(symptom_indices)):
                edge = tuple(sorted([symptom_indices[i], symptom_indices[j]]))
                edge_weights[edge] += 1.0

    # Add edges to graph
    for (u, v), weight in edge_weights.items():
        G.add_edge(u, v, weight=weight)

    # Convert to PyTorch Geometric format
    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])  # Undirected
        weight = data.get('weight', 1.0)
        edge_attr.append([weight])
        edge_attr.append([weight])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)

    graph_data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_symptoms)

    # Encode demographics
    age_bins = [0, 20, 30, 40, 100]

    for patient in patient_data:
        # Gender one-hot
        gender_vec = np.zeros(2, dtype=np.float32)
        gender_vec[0 if patient['gender'] == 'male' else 1] = 1.0
        patient['gender_vec'] = gender_vec

        # Age bin one-hot
        age_bin_vec = np.zeros(4, dtype=np.float32)
        bin_idx = np.digitize(patient['age'], age_bins) - 1
        bin_idx = max(0, min(3, bin_idx))
        age_bin_vec[bin_idx] = 1.0
        patient['age_bin_vec'] = age_bin_vec

    return df, symptom_to_idx, idx_to_symptom, patient_data, interaction_matrix, graph_data, age_bins


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

        # Forward
        scores = model(patient_idx, gender, age_bin, query_symptoms, edge_index)

        # BCE loss
        loss = criterion_bce(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, edge_index, device, k=5):
    """Evaluate model"""
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

            scores = model(patient_idx, gender, age_bin, query_symptoms, edge_index)

            # Get top-k predictions
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


def collate_fn(batch):
    """Custom collate function to handle variable-length data"""
    patient_idx = torch.tensor([item['patient_idx'] for item in batch], dtype=torch.long)
    gender = [item['gender'] for item in batch]
    age_bin = [item['age_bin'] for item in batch]
    query_symptoms = torch.stack([item['query_symptoms'] for item in batch])
    target_symptoms = [item['target_symptoms'] for item in batch]
    neg_symptoms = [item['neg_symptoms'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'patient_idx': patient_idx,
        'gender': gender,
        'age_bin': age_bin,
        'query_symptoms': query_symptoms,
        'target_symptoms': target_symptoms,
        'neg_symptoms': neg_symptoms,
        'labels': labels
    }


def main():
    """Main training pipeline"""

    # Hyperparameters
    CSV_PATH = '/Users/piw/Downloads/atest/AI symptom picker data.csv'
    EMBED_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # Preprocess
    df, symptom_to_idx, idx_to_symptom, patient_data, interaction_matrix, graph_data, age_bins = preprocess_data(CSV_PATH)

    num_symptoms = len(symptom_to_idx)
    num_patients = len(patient_data)

    # Split data
    train_data, temp_data = train_test_split(patient_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets
    train_dataset = SymptomDataset(train_data, symptom_to_idx, interaction_matrix)
    val_dataset = SymptomDataset(val_data, symptom_to_idx, interaction_matrix)
    test_dataset = SymptomDataset(test_data, symptom_to_idx, interaction_matrix)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = SymptomRecommender(
        num_symptoms=num_symptoms,
        num_patients=num_patients,
        symptom_embed_dim=EMBED_DIM
    ).to(DEVICE)

    edge_index = graph_data.edge_index.to(DEVICE)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion_bce = nn.BCELoss()

    print("\nStarting training...")
    best_val_recall = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_bce, edge_index, DEVICE)
        val_precision, val_recall = evaluate(model, val_loader, edge_index, DEVICE, k=5)

        scheduler.step(val_recall)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val P@5: {val_precision:.4f} | Val R@5: {val_recall:.4f}")

        # Save best model
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), 'model.pth')
            print(f"  → Saved best model (R@5: {val_recall:.4f})")

    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('model.pth'))
    test_precision, test_recall = evaluate(model, test_loader, edge_index, DEVICE, k=5)
    print(f"Test P@5: {test_precision:.4f} | Test R@5: {test_recall:.4f}")

    # Save artifacts
    print("\nSaving artifacts...")

    with open('symptom_to_idx.pkl', 'wb') as f:
        pickle.dump(symptom_to_idx, f)

    with open('idx_to_symptom.pkl', 'wb') as f:
        pickle.dump(idx_to_symptom, f)

    with open('age_bins.pkl', 'wb') as f:
        pickle.dump(age_bins, f)

    torch.save(graph_data, 'graph.pt')

    # Save model config
    config = {
        'num_symptoms': num_symptoms,
        'num_patients': num_patients,
        'symptom_embed_dim': EMBED_DIM
    }

    with open('model_config.pkl', 'wb') as f:
        pickle.dump(config, f)

    print("Training complete! Artifacts saved:")
    print("  - model.pth")
    print("  - symptom_to_idx.pkl")
    print("  - idx_to_symptom.pkl")
    print("  - age_bins.pkl")
    print("  - graph.pt")
    print("  - model_config.pkl")


if __name__ == '__main__':
    main()
