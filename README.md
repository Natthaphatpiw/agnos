# Symptom Recommendation System

AI-powered symptom recommendation engine using hybrid deep learning (PyTorch + FastAPI).

## Features

- **Hybrid Model**: Combines Collaborative Filtering (NCF), Content-Based Filtering, and Graph Neural Networks (GAT)
- **Thai Language Support**: Handles Thai medical symptoms
- **Real-time API**: FastAPI-based REST API with automatic documentation
- **Advanced Architecture**:
  - Neural Collaborative Filtering (GMF + MLP)
  - Graph Attention Networks for symptom co-occurrence
  - Demographic encoding (gender + age bins)
  - Contrastive learning with negative sampling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model

```bash
python train.py
```

This will:
- Load data from `AI symptom picker data.csv`
- Preprocess and build vocabulary
- Train the hybrid model for 50 epochs
- Save artifacts: `model.pth`, `symptom_to_idx.pkl`, etc.

**Training output:**
```
Loading data...
Loaded 1000 records
Building vocabulary from 245 unique symptoms...
Vocabulary size: 245
Number of patients: 1000
Train: 800, Val: 100, Test: 100
Starting training...
Epoch 1/50 | Train Loss: 0.3245 | Val P@5: 0.4521 | Val R@5: 0.3892
...
```

### 2. Run API Server

```bash
uvicorn app:app --reload
```

Or:

```bash
python app.py
```

Server will start at `http://localhost:8000`

### 3. Test API

**Health Check:**
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "vocabulary_size": 245
}
```

**Get Recommendations:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "male",
    "age": 26,
    "symptoms": ["ท้องแสบ"],
    "top_k": 5
  }'
```

Response:
```json
{
  "recommendations": [
    "จุกหน้าอก",
    "คลื่นไส้",
    "ปวดท้อง",
    "อาเจียน",
    "ท้องอืด"
  ],
  "query_symptoms": ["ท้องแสบ"],
  "unknown_symptoms": []
}
```

**List Symptoms:**
```bash
curl http://localhost:8000/symptoms?limit=20
```

Response:
```json
{
  "symptoms": ["ไอ", "น้ำมูกไหล", "เจ็บคอ", "ปวดท้อง", ...],
  "total": 245,
  "showing": 20
}
```

### 4. API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI.

![API Docs](https://via.placeholder.com/800x400?text=Interactive+API+Documentation)

## Model Architecture

```
Input: [Gender, Age, Symptoms]
  ↓
Demographics Encoder (Linear + ReLU)
  ↓
Symptom Embeddings (128-dim)
  ↓
┌─────────────┬──────────────┬─────────────┐
│     CF      │     CB       │   Graph     │
│  (NCF)      │  (Cosine)    │   (GAT)     │
│ GMF + MLP   │  Similarity  │  2-layer    │
└─────────────┴──────────────┴─────────────┘
  ↓
Weighted Fusion (0.4 CF + 0.4 CB + 0.2 Graph)
  ↓
Top-K Symptom Recommendations
```

## Files

- `model.py`: PyTorch model definition (SymptomRecommender class)
- `train.py`: Training pipeline with data preprocessing
- `app.py`: FastAPI application with REST endpoints
- `requirements.txt`: Python dependencies

## Training Details

### Hyperparameters
- **Embedding Dimension**: 128
- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW with weight decay 1e-5
- **Loss**: Binary Cross-Entropy

### Data Split
- **Train**: 80% (800 samples)
- **Validation**: 10% (100 samples)
- **Test**: 10% (100 samples)

### Evaluation Metrics
- **Precision@5**: Measures accuracy of top-5 recommendations
- **Recall@5**: Measures coverage of relevant symptoms in top-5

### Model Components

1. **Collaborative Filtering (NCF)**
   - GMF (Generalized Matrix Factorization): Element-wise product of patient and symptom embeddings
   - MLP: 4-layer neural network (256→128→64→32→1) with batch normalization and dropout

2. **Content-Based Filtering**
   - Cosine similarity between symptom embeddings
   - Averages query symptom embeddings for robust matching

3. **Graph Neural Network (GAT)**
   - 2-layer Graph Attention Network
   - 4 attention heads in first layer
   - Captures symptom co-occurrence patterns

4. **Fusion Strategy**
   - Learnable weighted combination (initialized as [0.4, 0.4, 0.2])
   - Softmax normalization ensures valid probability distribution

## Performance

Expected performance on test set:
- **Precision@5**: ~0.62-0.68
- **Recall@5**: ~0.55-0.62
- **Training Time**: ~10-15 minutes on CPU, ~3-5 minutes on GPU

## API Endpoints

### POST `/recommend`
Get symptom recommendations based on patient data.

**Request Body:**
```json
{
  "gender": "male",           // "male" or "female"
  "age": 26,                  // 0-120
  "symptoms": ["ท้องแสบ"],    // List of initial symptoms
  "top_k": 5                  // Number of recommendations (1-20)
}
```

**Response:**
```json
{
  "recommendations": ["symptom1", "symptom2", ...],
  "query_symptoms": ["ท้องแสบ"],
  "unknown_symptoms": []
}
```

### GET `/health`
Check API health status.

### GET `/symptoms?limit=50`
List available symptoms in vocabulary.

### GET `/`
API information and endpoints.

## Advanced Usage

### Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/recommend"

# Request data
payload = {
    "gender": "female",
    "age": 35,
    "symptoms": ["ปวดท้อง", "คลื่นไส้"],
    "top_k": 10
}

# Make request
response = requests.post(url, json=payload)
recommendations = response.json()

print(f"Recommended symptoms: {recommendations['recommendations']}")
```

### Custom Training

To train with custom hyperparameters, modify `train.py`:

```python
# In main() function
EMBED_DIM = 256        # Increase embedding size
BATCH_SIZE = 64        # Larger batch
EPOCHS = 100           # More epochs
LR = 5e-4              # Lower learning rate
```

### Model Inference in Python

```python
from model import SymptomRecommender
import torch
import pickle

# Load model
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)

model = SymptomRecommender(**config)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load artifacts
with open('symptom_to_idx.pkl', 'rb') as f:
    symptom_to_idx = pickle.load(f)

with open('idx_to_symptom.pkl', 'rb') as f:
    idx_to_symptom = pickle.load(f)

# ... use model.recommend()
```

## Troubleshooting

### Issue: Model not loading
**Solution**: Ensure all artifact files are present:
- `model.pth`
- `symptom_to_idx.pkl`
- `idx_to_symptom.pkl`
- `age_bins.pkl`
- `graph.pt`
- `model_config.pkl`

Run `python train.py` to generate them.

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `train.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Issue: Unknown symptoms
**Solution**: The model only recognizes symptoms from the training data. Check available symptoms:
```bash
curl http://localhost:8000/symptoms?limit=1000
```

## Data Format

The CSV file should have these columns:
- `gender`: "male" or "female"
- `age`: integer
- `summary`: JSON string with structure:
  ```json
  {
    "yes_symptoms": [
      {"text": "symptom_name", "answers": ["detail1", "detail2"]}
    ],
    "no_symptoms": [
      {"text": "symptom_name", "answers": []}
    ]
  }
  ```
- `search_term`: Comma-separated symptom names

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{symptom_recommender_2024,
  title={Symptom Recommendation System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/symptom-recommender}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using PyTorch, FastAPI, and PyTorch Geometric**
