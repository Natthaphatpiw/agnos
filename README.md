# Symptom Recommendation System

AI-powered symptom recommendation engine using hybrid deep learning with LLM enhancement layer (PyTorch + FastAPI + OpenAI/Google).

## Features

- **Hybrid Deep Learning Model**: Combines Collaborative Filtering (NCF), Content-Based Filtering, and Graph Neural Networks (GAT)
- **LLM Enhancement Layer**: Optional post-processing using OpenAI GPT-4.1-mini or Google Gemini 2.5-flash for recommendation refinement
- **Bilingual Support**: Handles both Thai and English medical symptoms
- **Real-time API**: FastAPI-based REST API with automatic documentation
- **Docker Ready**: Containerized deployment with docker-compose
- **Advanced Architecture**:
  - Neural Collaborative Filtering (GMF + MLP)
  - Graph Attention Networks for symptom co-occurrence
  - Demographic encoding (gender + age bins)
  - Structured LLM output with quality scoring

## Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Clone repository
git clone https://github.com/Natthaphatpiw/agnos.git
cd agnos

# Configure environment
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Train model (first time only)
python train.py

# Build and run
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Start server
uvicorn app:app --reload
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
    "symptoms": ["ไอ", "เสมหะ"],
    "top_k": 5
  }'
```

Response (with LLM enhancement):
```json
{
  "recommendations": [
    "เจ็บคอ",
    "น้ำมูกไหล",
    "ไข้",
    "ปวดศีรษะ",
    "คัดจมูก"
  ],
  "query_symptoms": ["ไอ", "เสมหะ"],
  "unknown_symptoms": [],
  "score": 0.92,
  "reason": "อาการสอดคล้องกับโรคระบบทางเดินหายใจ เรียงตามความเกี่ยวข้องทางคลินิก"
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

## System Architecture

The system uses a two-stage architecture:

### Stage 1: Deep Learning Hybrid Model
1. **Input Processing**: Validates and encodes patient demographics (gender, age) and symptoms
2. **Feature Engineering**: One-hot encoding for demographics, symptom embeddings (128-dim)
3. **Three Parallel Pathways**:
   - **Collaborative Filtering (NCF)**: GMF + 4-layer MLP for patient-symptom interactions
   - **Content-Based Filtering**: Cosine similarity between symptom embeddings
   - **Graph Attention Network (GAT)**: 2-layer GAT with 4 attention heads for co-occurrence patterns
4. **Weighted Fusion**: Learned combination (0.4 CF + 0.4 CB + 0.2 GAT)
5. **Top-K Selection**: Returns highest-scoring symptoms

### Stage 2: LLM Enhancement (Optional)
1. **API Key Priority**: OpenAI > Google > Skip (DL only)
2. **Refinement Tasks**:
   - Filter irrelevant symptoms and non-symptom items
   - Translate English terms to Thai medical terminology
   - Remove duplicates and semantic overlaps
   - Reorder by clinical importance
3. **Structured Output**: Returns refined recommendations with quality score and reasoning

For detailed architecture documentation, see [solution_overview.html](solution_overview.html).

## Environment Variables

The system requires LLM API keys for optimal performance:

- **OPENAI_API_KEY** (Recommended): For GPT-4.1-mini access - provides best quality
- **GOOGLE_API_KEY** (Optional): For Gemini 2.5-flash - fallback option

Create a `.env` file:
```bash
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-key-here  # Optional
```

**Note**: If no API keys are configured, the system returns raw DL predictions without LLM refinement. For production use, OPENAI_API_KEY is strongly recommended.

## Project Structure

```
.
├── app.py                          # FastAPI application with LLM integration
├── model.py                        # Hybrid DL model (NCF + CB + GAT)
├── model_lightweight.py            # Lightweight alternative model
├── train.py                        # Training pipeline (DL Hybrid)
├── train_traditional_ml.py         # Traditional ML baseline (SVD + KNN + PageRank)
├── train_lightweight.py            # Lightweight NN training
├── predict_traditional.py          # Interactive CLI for Traditional ML
├── compare_final.py                # Model comparison script
├── Dockerfile                      # Docker container definition
├── docker-compose.yml              # Docker orchestration
├── requirements.txt                # Python dependencies
├── solution_overview.html          # Architecture documentation
├── api_documentation.html          # API reference documentation
└── tests/                          # Test suite
    ├── test_api.py                 # API endpoint tests
    └── test_model.py               # Model unit tests
```

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

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_api.py
pytest tests/test_model.py
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

## Documentation

- **[solution_overview.html](solution_overview.html)**: Comprehensive architecture and design documentation
- **[api_documentation.html](api_documentation.html)**: Complete API reference with examples
- **[/docs](http://localhost:8000/docs)**: Interactive Swagger UI (when server is running)

## Model Comparison

The repository includes three model implementations:

| Model | Recall@5 | Parameters | Quality | Use Case |
|-------|----------|------------|---------|----------|
| **DL Hybrid** (main) | 25% | ~2M | High diversity, LLM-enhanced | Production |
| Traditional ML | 99% | 0 | Needs LLM filtering | Baseline |
| Lightweight NN | 60% | ~42K | Moderate | Resource-constrained |

Run comparison:
```bash
python compare_final.py
```

## Repository

```
GitHub: https://github.com/Natthaphatpiw/agnos.git
```

## License

MIT License

---

**Built with PyTorch, FastAPI, PyTorch Geometric, LangChain, and OpenAI**
