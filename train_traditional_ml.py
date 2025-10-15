"""
Traditional ML Hybrid Approach for Symptom Recommendation
- Content-Based: Cosine Similarity (no training)
- Collaborative: SVD + KNN
- Graph: NetworkX + PageRank (no training)
"""

import pandas as pd
import numpy as np
import pickle
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRADITIONAL ML HYBRID MODEL")
print("Content-Based (Cosine) + Collaborative (SVD+KNN) + Graph (PageRank)")
print("="*80)

# Load data
CSV_PATH = 'AI symptom picker data.csv'
print(f"\nLoading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records")

# Parse symptoms from JSON
print("\nParsing symptoms from JSON...")
import json

def parse_symptoms(summary):
    if pd.isna(summary):
        return []
    try:
        data = json.loads(summary)
        yes_symptoms = data.get('yes_symptoms', [])
        return [s['text'] for s in yes_symptoms if isinstance(s, dict) and 'text' in s]
    except:
        return []

df['symptoms'] = df['summary'].apply(parse_symptoms)
df = df[df['symptoms'].apply(len) > 0]

# Build vocabulary
all_symptoms = set()
for symptoms in df['symptoms']:
    all_symptoms.update(symptoms)
all_symptoms = sorted(list(all_symptoms))
symptom_to_idx = {s: i for i, s in enumerate(all_symptoms)}
idx_to_symptom = {i: s for s, i in symptom_to_idx.items()}

num_symptoms = len(all_symptoms)
num_patients = len(df)

print(f"Vocabulary size: {num_symptoms}")
print(f"Number of patients: {num_patients}")

# Build interaction matrix (patient x symptom)
print("\nBuilding interaction matrix...")
interaction_matrix = np.zeros((num_patients, num_symptoms))
for i, row in df.iterrows():
    for symptom in row['symptoms']:
        if symptom in symptom_to_idx:
            interaction_matrix[i, symptom_to_idx[symptom]] = 1

print(f"Interaction matrix shape: {interaction_matrix.shape}")
print(f"Sparsity: {(interaction_matrix == 0).sum() / interaction_matrix.size * 100:.2f}%")

# Demographics
df['gender_encoded'] = df['gender'].map({'ชาย': 0, 'หญิง': 1}).fillna(0).astype(int)
age_bins = [0, 18, 35, 50, 120]
df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=[0, 1, 2, 3]).astype(int)

print("\n" + "="*80)
print("COMPONENT 1: CONTENT-BASED (COSINE SIMILARITY)")
print("="*80)
print("Computing symptom similarity matrix...")

# Normalize interaction matrix for cosine similarity
symptom_vectors = interaction_matrix.T  # (num_symptoms, num_patients)
symptom_vectors_norm = normalize(symptom_vectors, norm='l2', axis=1)
symptom_similarity = symptom_vectors_norm @ symptom_vectors_norm.T

print(f"✓ Symptom similarity matrix computed: {symptom_similarity.shape}")

print("\n" + "="*80)
print("COMPONENT 2: COLLABORATIVE FILTERING (SVD + KNN)")
print("="*80)

# SVD for dimensionality reduction
print("Training SVD...")
n_components = min(50, min(interaction_matrix.shape) - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
patient_embeddings = svd.fit_transform(interaction_matrix)
symptom_embeddings = svd.components_.T

print(f"✓ SVD trained with {n_components} components")
print(f"  Patient embeddings: {patient_embeddings.shape}")
print(f"  Symptom embeddings: {symptom_embeddings.shape}")
print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

# KNN for finding similar symptoms
print("\nTraining KNN...")
knn = NearestNeighbors(n_neighbors=min(20, num_symptoms), metric='cosine')
knn.fit(symptom_embeddings)

print(f"✓ KNN trained with {min(20, num_symptoms)} neighbors")

print("\n" + "="*80)
print("COMPONENT 3: GRAPH-BASED (PAGERANK)")
print("="*80)

# Build symptom co-occurrence graph
print("Building co-occurrence graph...")
G = nx.Graph()
G.add_nodes_from(range(num_symptoms))

# Add edges based on co-occurrence
edge_weights = {}
for _, row in df.iterrows():
    symptoms = [symptom_to_idx[s] for s in row['symptoms'] if s in symptom_to_idx]
    for i in range(len(symptoms)):
        for j in range(i+1, len(symptoms)):
            edge = tuple(sorted([symptoms[i], symptoms[j]]))
            edge_weights[edge] = edge_weights.get(edge, 0) + 1

# Add weighted edges
for (s1, s2), weight in edge_weights.items():
    G.add_edge(s1, s2, weight=weight)

print(f"✓ Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Compute PageRank for each symptom
print("Computing PageRank scores...")
pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100)

print(f"✓ PageRank computed for all symptoms")

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

# Evaluation function
def predict_symptoms(patient_idx, query_symptoms, top_k=5, weights=(0.4, 0.4, 0.2)):
    """
    Predict symptoms using hybrid approach

    Args:
        patient_idx: patient index
        query_symptoms: list of query symptom indices
        top_k: number of recommendations
        weights: (content, collaborative, graph) weights

    Returns:
        top_k symptom indices
    """
    w_content, w_collab, w_graph = weights

    # Initialize scores
    scores = np.zeros(num_symptoms)

    # 1. Content-Based: average similarity to query symptoms
    if len(query_symptoms) > 0:
        content_scores = symptom_similarity[query_symptoms].mean(axis=0)
        scores += w_content * content_scores

    # 2. Collaborative: KNN + SVD
    patient_embed = patient_embeddings[patient_idx]
    collab_scores = symptom_embeddings @ patient_embed
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-8)
    scores += w_collab * collab_scores

    # 3. Graph: PageRank boosted by query symptoms
    graph_scores = np.array([pagerank_scores.get(i, 0) for i in range(num_symptoms)])
    # Boost neighbors of query symptoms
    for q_symptom in query_symptoms:
        if q_symptom in G:
            for neighbor in G.neighbors(q_symptom):
                graph_scores[neighbor] *= 2
    graph_scores = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min() + 1e-8)
    scores += w_graph * graph_scores

    # Mask query symptoms
    for q_symptom in query_symptoms:
        scores[q_symptom] = -np.inf

    # Get top-k
    top_k_idx = np.argsort(scores)[-top_k:][::-1]
    return top_k_idx

# Simple evaluation on a sample
print("Evaluating on sample data...")
from sklearn.model_selection import train_test_split

train_idx, temp_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

def evaluate_split(indices, split_name, max_samples=200):
    precisions_5 = []
    recalls_5 = []

    for idx in indices[:max_samples]:
        row = df.iloc[idx]
        true_symptoms = [symptom_to_idx[s] for s in row['symptoms'] if s in symptom_to_idx]

        if len(true_symptoms) < 2:
            continue

        # Use one symptom as query, others as targets
        query_symptom = [true_symptoms[0]]
        target_symptoms = set(true_symptoms[1:])

        # Predict
        pred_symptoms = predict_symptoms(idx, query_symptom, top_k=5)
        pred_set = set(pred_symptoms)

        # Metrics
        correct = len(pred_set & target_symptoms)
        precision = correct / 5
        recall = correct / len(target_symptoms) if len(target_symptoms) > 0 else 0

        precisions_5.append(precision)
        recalls_5.append(recall)

    return np.mean(precisions_5), np.mean(recalls_5)

train_p5, train_r5 = evaluate_split(train_idx, "Train", max_samples=200)
val_p5, val_r5 = evaluate_split(val_idx, "Validation", max_samples=100)
test_p5, test_r5 = evaluate_split(test_idx, "Test", max_samples=100)

print(f"\n{'Split':<15} {'Precision@5':<15} {'Recall@5':<15}")
print("-"*50)
print(f"{'Train':<15} {train_p5:<15.4f} {train_r5:<15.4f}")
print(f"{'Validation':<15} {val_p5:<15.4f} {val_r5:<15.4f}")
print(f"{'Test':<15} {test_p5:<15.4f} {test_r5:<15.4f}")

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save model components
model_data = {
    'symptom_similarity': symptom_similarity,
    'svd': svd,
    'knn': knn,
    'patient_embeddings': patient_embeddings,
    'symptom_embeddings': symptom_embeddings,
    'pagerank_scores': pagerank_scores,
    'graph': G,
    'symptom_to_idx': symptom_to_idx,
    'idx_to_symptom': idx_to_symptom,
    'num_symptoms': num_symptoms,
    'num_patients': num_patients,
    'age_bins': age_bins,
    'interaction_matrix': interaction_matrix
}

with open('model_traditional_ml.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved to 'model_traditional_ml.pkl'")
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
