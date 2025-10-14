# สถาปัตยกรรมระบบ Symptom Recommendation Engine

เอกสารฉบับนี้อธิบายสถาปัตยกรรมของระบบแนะนำอาการทางการแพทย์แบบละเอียด ตั้งแต่การ train โมเดลไปจนถึงการทำงานของ API

---

## สารบัญ

1. [ภาพรวมระบบ](#ภาพรวมระบบ)
2. [สถาปัตยกรรมโมเดล ML](#สถาปัตยกรรมโมเดล-ml)
3. [กระบวนการ Training](#กระบวนการ-training)
4. [สถาปัตยกรรม Backend API](#สถาปัตยกรรม-backend-api)
5. [Data Flow แบบละเอียด](#data-flow-แบบละเอียด)
6. [ตัวอย่างการทำงานจริง](#ตัวอย่างการทำงานจริง)

---

## ภาพรวมระบบ

### จุดประสงค์
ระบบนี้ถูกออกแบบมาเพื่อ **แนะนำอาการทางการแพทย์ที่เป็นไปได้ต่อไป** โดยอิงจากข้อมูล:
- **Demographics**: เพศ (male/female) และอายุ (0-120 ปี)
- **Initial Symptoms**: อาการเริ่มต้นที่ผู้ป่วยมี (เช่น ไอ, ปวดท้อง)

คล้ายกับระบบ Netflix ที่แนะนำหนังจากประวัติการรับชม แต่เราแนะนำอาการจากข้อมูลผู้ป่วยในอดีต

### เทคโนโลยีหลัก
- **PyTorch**: สำหรับสร้าง Deep Learning Model
- **PyTorch Geometric**: สำหรับ Graph Neural Networks (GAT)
- **FastAPI**: สำหรับสร้าง REST API
- **Pandas/NumPy**: สำหรับจัดการข้อมูล

---

## สถาปัตยกรรมโมเดล ML

### 🎯 โมเดล Hybrid แบบ 3 สาย (Triple-Path Architecture)

โมเดลนี้รวมเทคนิค 3 แบบเข้าด้วยกัน:

```
Input (Gender, Age, Symptoms)
         ↓
    ┌────┴────┐
    │ Preprocess│
    └────┬────┘
         ↓
    ┌────┴────────────────────────────┐
    │                                 │
┌───┴───┐  ┌──────┴──────┐  ┌────┴────┐
│  CF   │  │     CB      │  │  Graph  │
│(NCF)  │  │  (Cosine)   │  │  (GAT)  │
└───┬───┘  └──────┬──────┘  └────┬────┘
    │             │              │
    └─────────┬───┴──────────────┘
              ↓
         Weighted Fusion
         (0.4 + 0.4 + 0.2)
              ↓
         Top-K Rankings
              ↓
        Recommendations
```

---

### 1️⃣ Collaborative Filtering (CF) - Neural Collaborative Filtering (NCF)

**หลักการ**: เรียนรู้จากพฤติกรรมผู้ป่วยที่คล้ายกัน เช่น ผู้ป่วยที่มีอาการเริ่มต้นเหมือนกันมักมีอาการต่อไปคล้ายกัน

#### Architecture:

```python
# NCF = GMF + MLP

# 1. GMF (Generalized Matrix Factorization)
patient_embed = Embedding(patient_idx)      # [batch, 128]
symptom_embed = Embedding(symptom_idx)      # [batch, 128]
gmf_vector = patient_embed * symptom_embed  # Element-wise product
gmf_score = Linear(gmf_vector)              # [batch, 1]

# 2. MLP (Multi-Layer Perceptron)
mlp_input = Concat([patient_embed, symptom_embed, demographics])  # [batch, 272]
mlp_hidden = Linear(272 → 256) → ReLU → BatchNorm → Dropout
           → Linear(256 → 128) → ReLU → BatchNorm → Dropout
           → Linear(128 → 64)  → ReLU → BatchNorm → Dropout
           → Linear(64 → 32)   → ReLU → BatchNorm → Dropout
mlp_score = Linear(32 → 1)

# 3. Combine
cf_score = gmf_score + mlp_score
```

**Input Features**:
- **Patient Embedding**: เลข index ของผู้ป่วย (ใช้ embedding 128 มิติ)
- **Symptom Embedding**: เลข index ของอาการ (ใช้ embedding 128 มิติ)
- **Demographics Embedding**: เพศ + อายุ (encode เป็น 16 มิติ)

**ข้อดี**:
- จับ pattern ที่ซับซ้อนได้ดีผ่าน MLP
- GMF ช่วยจับความสัมพันธ์แบบ linear
- เหมาะกับข้อมูลที่มี user-item interaction มาก

---

### 2️⃣ Content-Based Filtering (CB)

**หลักการ**: เปรียบเทียบความคล้ายคลึงระหว่างอาการโดยตรงผ่าน embeddings

#### Algorithm:

```python
# 1. Embed query symptoms
query_embeds = SymptomEmbedding(query_symptoms)  # [batch, num_query, 128]
query_avg = Mean(query_embeds, dim=1)            # [batch, 1, 128]

# 2. Embed candidate symptoms
candidate_embeds = SymptomEmbedding(candidates)  # [batch, num_candidates, 128]

# 3. Cosine Similarity
query_norm = Normalize(query_avg, p=2)           # L2 normalization
candidate_norm = Normalize(candidate_embeds, p=2)

cb_scores = CosineSim(query_norm, candidate_norm)
          = (query_norm * candidate_norm).sum(dim=-1)  # [batch, num_candidates]
```

**ตัวอย่าง**:
```
Query: ["ไอ"]
  → Embedding: [0.23, -0.45, 0.12, ...]

Candidates:
  "เสมหะ"    → Embedding: [0.25, -0.40, 0.15, ...] → Cosine = 0.89 (คล้ายมาก)
  "น้ำมูกไหล" → Embedding: [0.20, -0.38, 0.10, ...] → Cosine = 0.82
  "ปวดท้อง"  → Embedding: [-0.10, 0.30, -0.25, ...] → Cosine = 0.15 (ไม่คล้าย)
```

**ข้อดี**:
- ทำงานได้แม้กับผู้ป่วยใหม่ (Cold Start Problem)
- Interpretable - เห็นว่าอาการไหนคล้ายกัน
- Fast inference

---

### 3️⃣ Graph Neural Network (GAT - Graph Attention Network)

**หลักการ**: สร้าง graph ที่ nodes คืออาการ และ edges คือความถี่ที่อาการเกิดร่วมกัน (co-occurrence)

#### Graph Construction:

```python
# สร้าง Graph จากข้อมูล training
G = NetworkX.Graph()
G.add_nodes(all_symptoms)  # 338 nodes

# สำหรับแต่ละผู้ป่วย
for patient in patients:
    symptoms = patient.yes_symptoms  # เช่น ["ไอ", "เสมหะ", "เจ็บคอ"]

    # เพิ่ม edge ระหว่างอาการทุกคู่
    for (symptom_i, symptom_j) in combinations(symptoms, 2):
        edge = (symptom_i, symptom_j)
        edge_weights[edge] += 1  # นับจำนวนครั้งที่เกิดร่วมกัน

# ตัวอย่าง edges:
# ("ไอ", "เสมหะ") → weight = 156 (เกิดร่วมกัน 156 ครั้ง)
# ("ไอ", "เจ็บคอ") → weight = 203
# ("ปวดท้อง", "คลื่นไส้") → weight = 89
```

#### GAT Architecture:

```python
# Input: Symptom embeddings X ∈ R^(338 × 128)
X = SymptomEmbeddings.weight  # [338, 128]

# Layer 1: Multi-head attention (4 heads)
H1 = GAT_Layer1(X, edge_index, heads=4)  # [338, 128*4]
H1 = ELU(H1)
H1 = Dropout(H1, p=0.3)

# Layer 2: Single-head attention
H2 = GAT_Layer2(H1, edge_index, heads=1)  # [338, 128]

# H2 คือ symptom embeddings ที่ถูก propagate ผ่าน graph
```

**Attention Mechanism**:

สำหรับแต่ละ node (อาการ), GAT จะคำนวณ attention weights กับ neighbors:

```python
# สำหรับ node "ไอ"
neighbors = ["เสมหะ", "เจ็บคอ", "น้ำมูกไหล", ...]

# คำนวณ attention scores
for neighbor in neighbors:
    # Attention coefficient
    alpha = softmax(
        LeakyReLU(
            W1 @ embed("ไอ") + W2 @ embed(neighbor)
        )
    )

# Update embedding ของ "ไอ"
new_embed("ไอ") = Σ (alpha_neighbor × embed(neighbor))
```

**ข้อดี**:
- จับ **co-occurrence patterns** ได้ดี เช่น "ไอ" มักมากับ "เสมหะ"
- Attention weights บอกว่าอาการไหนสำคัญกับอาการไหน
- ไม่จำกัดแค่ direct connections (ข้ามได้หลาย hops)

---

### 🔀 Fusion Strategy

รวม scores จาก 3 สายเข้าด้วยกัน:

```python
# 1. Normalize scores to [0, 1]
cf_norm = Sigmoid(cf_scores)              # CF score
cb_norm = (cb_scores + 1) / 2            # Cosine [-1,1] → [0,1]
graph_norm = (graph_scores + 1) / 2      # Cosine [-1,1] → [0,1]

# 2. Learnable fusion weights (เริ่มต้นที่ [0.4, 0.4, 0.2])
weights = [w_cf, w_cb, w_graph]
weights = Softmax(weights)  # ทำให้ผลรวม = 1

# 3. Weighted sum
final_scores = (
    weights[0] * cf_norm +      # 40% จาก CF
    weights[1] * cb_norm +      # 40% จาก Content-Based
    weights[2] * graph_norm     # 20% จาก Graph
)

# 4. Top-K selection
top_k_indices = ArgTopK(final_scores, k=5)
recommendations = [idx_to_symptom[idx] for idx in top_k_indices]
```

**ทำไมใช้ weights แบบนี้?**
- **CF (40%)**: สำคัญที่สุดเพราะเรียนรู้จากพฤติกรรมผู้ป่วยจริง
- **CB (40%)**: สำคัญพอๆ กัน เพราะจับความคล้ายตามเนื้อหาได้ดี
- **Graph (20%)**: เป็น supplement เพื่อเพิ่ม context ของ co-occurrence

**Learnable Weights**: น้ำหนักเหล่านี้จะถูกปรับระหว่าง training ผ่าน `nn.Parameter`

---

## กระบวนการ Training

### 📊 Data Preprocessing (ไฟล์ `train.py`)

#### Step 1: โหลดและ Parse ข้อมูล

```python
# โหลด CSV
df = pd.read_csv('AI symptom picker data.csv')
# Columns: gender, age, summary (JSON), search_term

# Parse JSON summary
for row in df:
    summary = json.loads(row['summary'])

    # Extract yes_symptoms
    yes_symptoms = []
    for sym_obj in summary['yes_symptoms']:
        symptom = sym_obj['text']           # เช่น "ไอ"
        answers = sym_obj['answers']        # เช่น ["ระยะเวลา 1-3 สัปดาห์"]

        # Parse severity จาก answers
        severity = parse_severity(answers)  # 1.0 - 3.0

        yes_symptoms.append(symptom)
        symptom_details[symptom] = {
            'answers': answers,
            'severity': severity
        }

    # Extract no_symptoms
    no_symptoms = [sym['text'] for sym in summary['no_symptoms']]

    # Extract from search_term
    search_symptoms = row['search_term'].split(',')
```

**Severity Parsing**:

```python
def parse_severity(answers):
    """แปลง text คำตอบเป็น severity score"""
    severity_map = {
        'ปวดจนไม่สามารถทำงานได้': 3.0,           # รุนแรงมาก
        'ปวดจนไม่สามารถทำกิจกรรมใดใดได้เลย': 3.0,
        'ปวดมาก': 2.5,
        'ปวดปานกลาง': 2.0,                        # ปานกลาง
        'ส่งผลต่อการดำเนินกิจวัตรประจำวันบ้าง': 1.5,
        'ปวดเล็กน้อย': 1.0,                       # เล็กน้อย
        'เล็กน้อย': 1.0,
    }

    for pattern, score in severity_map.items():
        if pattern in ' '.join(answers):
            return score

    return 1.0  # Default
```

---

#### Step 2: สร้าง Vocabulary

```python
# รวมอาการทั้งหมดที่พบ
all_symptoms = set()

for patient in patients:
    all_symptoms.update(patient['yes_symptoms'])
    all_symptoms.update(patient['no_symptoms'])
    all_symptoms.update(patient['search_symptoms'])

# สร้าง mapping
symptom_to_idx = {sym: idx for idx, sym in enumerate(sorted(all_symptoms))}
idx_to_symptom = {idx: sym for sym, idx in symptom_to_idx.items()}

# ตัวอย่าง:
# symptom_to_idx = {
#     "ไอ": 0,
#     "เสมหะ": 1,
#     "ปวดท้อง": 2,
#     ...
# }
```

**ผลลัพธ์**: Vocabulary size = 338 symptoms

---

#### Step 3: สร้าง Interaction Matrix

```python
# Matrix: [num_patients × num_symptoms]
interaction_matrix = np.zeros((1000, 338), dtype=np.float32)

for patient_idx, patient in enumerate(patients):
    # Yes symptoms → positive weight (weighted by severity)
    for symptom in patient['yes_symptoms']:
        sym_idx = symptom_to_idx[symptom]
        severity = patient['symptom_details'][symptom]['severity']
        interaction_matrix[patient_idx, sym_idx] = severity  # 1.0 - 3.0

    # No symptoms → negative weight
    for symptom in patient['no_symptoms']:
        sym_idx = symptom_to_idx[symptom]
        interaction_matrix[patient_idx, sym_idx] = -1.0

# ตัวอย่าง row ของผู้ป่วย #5:
# [0, 2.5, 0, 1.0, 0, ..., -1.0, 0, ...]
#  ^   ^      ^             ^
#  |   |      |             |
#  |   |      |             no_symptom
#  |   |      yes_symptom (severity=1.0)
#  |   yes_symptom (severity=2.5)
#  ไม่มีอาการนี้
```

---

#### Step 4: สร้าง Co-occurrence Graph

```python
# สร้าง NetworkX graph
G = nx.Graph()
G.add_nodes_from(range(338))  # 338 symptoms

edge_weights = defaultdict(float)

for patient in patients:
    symptoms = patient['yes_symptoms']
    symptom_indices = [symptom_to_idx[s] for s in symptoms]

    # สร้าง edge ทุกคู่ของอาการที่เกิดร่วมกัน
    for i in range(len(symptom_indices)):
        for j in range(i+1, len(symptom_indices)):
            edge = tuple(sorted([symptom_indices[i], symptom_indices[j]]))
            edge_weights[edge] += 1.0  # เพิ่มน้ำหนัก

# Add edges
for (u, v), weight in edge_weights.items():
    G.add_edge(u, v, weight=weight)

# แปลงเป็น PyTorch Geometric format
edge_index = []
edge_attr = []

for u, v, data in G.edges(data=True):
    # Undirected → เพิ่มทั้ง 2 ทิศทาง
    edge_index.append([u, v])
    edge_index.append([v, u])

    weight = data['weight']
    edge_attr.append([weight])
    edge_attr.append([weight])

edge_index = torch.tensor(edge_index).t()  # [2, num_edges]
edge_attr = torch.tensor(edge_attr)        # [num_edges, 1]

graph_data = Data(
    edge_index=edge_index,
    edge_attr=edge_attr,
    num_nodes=338
)
```

**ตัวอย่าง Graph**:
```
Nodes: 338 (อาการทั้งหมด)
Edges: ~5,000-10,000 (ขึ้นอยู่กับ co-occurrence)

ตัวอย่าง edges:
  (0, 1): weight=156    # "ไอ" ↔ "เสมหะ" (เกิดร่วมกัน 156 ครั้ง)
  (0, 45): weight=203   # "ไอ" ↔ "เจ็บคอ"
  (2, 89): weight=89    # "ปวดท้อง" ↔ "คลื่นไส้"
```

---

#### Step 5: Encode Demographics

```python
age_bins = [0, 20, 30, 40, 100]  # 4 age groups

for patient in patients:
    # Gender one-hot encoding
    gender_vec = np.zeros(2)
    gender_vec[0 if patient['gender'] == 'male' else 1] = 1.0
    # male: [1, 0], female: [0, 1]

    # Age bin one-hot encoding
    age_bin_vec = np.zeros(4)
    bin_idx = np.digitize(patient['age'], age_bins) - 1
    bin_idx = max(0, min(3, bin_idx))  # Clip to [0, 3]
    age_bin_vec[bin_idx] = 1.0

    # ตัวอย่าง:
    # age=26 → bin_idx=1 (20-30) → [0, 1, 0, 0]
    # age=55 → bin_idx=3 (40+)   → [0, 0, 0, 1]

    patient['gender_vec'] = gender_vec
    patient['age_bin_vec'] = age_bin_vec
```

**Demographics Encoding**:
- Gender (2D) + Age Bin (4D) = **6D input**
- ผ่าน Linear layer → **16D demographics embedding**

---

### 🎓 Training Loop

#### Dataset Class

```python
class SymptomDataset(Dataset):
    def __getitem__(self, idx):
        patient = self.data[idx]

        # Split symptoms: query vs target
        pos_symptoms = patient['yes_symptoms']

        if len(pos_symptoms) > 1:
            split_idx = len(pos_symptoms) // 2
            query_symptoms = pos_symptoms[:split_idx]      # ใช้เป็น input
            target_symptoms = pos_symptoms[split_idx:]     # ใช้เป็น target
        else:
            query_symptoms = pos_symptoms
            target_symptoms = pos_symptoms

        # Encode to indices
        query_indices = [self.symptom_to_idx[s] for s in query_symptoms]
        target_indices = [self.symptom_to_idx[s] for s in target_symptoms]
        neg_indices = [self.symptom_to_idx[s] for s in patient['no_symptoms']]

        # Pad query to fixed length (10)
        query_indices += [0] * (10 - len(query_indices))

        # Create binary labels
        labels = torch.zeros(num_symptoms)  # [338]
        for idx in target_indices:
            labels[idx] = 1.0  # Positive symptoms

        return {
            'patient_idx': idx,
            'gender': patient['gender_vec'],
            'age_bin': patient['age_bin_vec'],
            'query_symptoms': torch.tensor(query_indices),
            'target_symptoms': target_indices,
            'neg_symptoms': neg_indices,
            'labels': labels
        }
```

**ตัวอย่าง batch**:
```
ผู้ป่วยมีอาการ: ["ไอ", "เสมหะ", "เจ็บคอ", "น้ำมูกไหล"]

Split:
  Query:  ["ไอ", "เสมหะ"]              → Input
  Target: ["เจ็บคอ", "น้ำมูกไหล"]     → Ground truth

Labels: [0, 0, ..., 1, ..., 1, ...]  (1 ที่ตำแหน่งของ target symptoms)
```

---

#### Training Function

```python
def train_epoch(model, dataloader, optimizer, criterion_bce, edge_index, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Move to device
        patient_idx = batch['patient_idx'].to(device)      # [32]
        gender = batch['gender'].to(device)                # [32, 2]
        age_bin = batch['age_bin'].to(device)              # [32, 4]
        query_symptoms = batch['query_symptoms'].to(device)  # [32, 10]
        labels = batch['labels'].to(device)                # [32, 338]

        optimizer.zero_grad()

        # Forward pass
        scores = model(
            patient_idx=patient_idx,
            gender=gender,
            age_bin=age_bin,
            query_symptoms=query_symptoms,
            edge_index=edge_index
        )  # → [32, 338]

        # Compute loss
        loss = criterion_bce(scores, labels)  # Binary Cross-Entropy

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

**Loss Function**: Binary Cross-Entropy (BCE)

```python
BCE = -[y * log(ŷ) + (1-y) * log(1-ŷ)]

# ตัวอย่าง:
# Target: y = [0, 1, 0, 1, 0]
# Pred:   ŷ = [0.1, 0.9, 0.2, 0.8, 0.15]
#
# Loss = -(
#     0*log(0.1) + 1*log(0.9) +      # symptom 0 (negative, correct)
#     1*log(0.9) +                    # symptom 1 (positive, correct)
#     0*log(0.2) + 1*log(0.8) +      # ...
#     0*log(0.15)
# )
```

---

#### Evaluation Function

```python
def evaluate(model, dataloader, edge_index, device, k=5):
    model.eval()
    all_precisions = []
    all_recalls = []

    with torch.no_grad():
        for batch in dataloader:
            # Forward
            scores = model(...)  # [batch, 338]
            labels = batch['labels']

            for i in range(len(scores)):
                # Mask query symptoms (ห้ามแนะนำอาการที่ input มา)
                query_mask = (query_symptoms[i] != 0)
                scores[i][query_symptoms[i][query_mask]] = -float('inf')

                # Get top-K predictions
                top_k_indices = torch.topk(scores[i], k=5).indices
                true_indices = torch.where(labels[i] > 0)[0]

                # Calculate metrics
                tp = len(set(top_k_indices) & set(true_indices))
                precision = tp / k
                recall = tp / len(true_indices) if len(true_indices) > 0 else 0

                all_precisions.append(precision)
                all_recalls.append(recall)

    return np.mean(all_precisions), np.mean(all_recalls)
```

**Metrics**:
- **Precision@5**: จาก 5 อาการที่แนะนำ กี่อาการที่ถูกต้อง
  ```
  Precision@5 = (จำนวนอาการที่ถูกต้องใน top-5) / 5
  ```

- **Recall@5**: จากอาการที่ควรแนะนำทั้งหมด แนะนำได้กี่อาการ
  ```
  Recall@5 = (จำนวนอาการที่ถูกต้องใน top-5) / (จำนวนอาการที่ควรแนะนำทั้งหมด)
  ```

**ตัวอย่าง**:
```
Ground Truth: ["เสมหะ", "เจ็บคอ", "น้ำมูกไหล"]  (3 อาการ)
Predicted Top-5: ["เสมหะ", "ไข้", "เจ็บคอ", "ปวดศีรษะ", "คลื่นไส้"]

ถูกต้อง: ["เสมหะ", "เจ็บคอ"]  (2 อาการ)

Precision@5 = 2/5 = 0.40
Recall@5 = 2/3 = 0.67
```

---

#### Main Training Loop

```python
# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

# Initialize
model = SymptomRecommender(num_symptoms=338, num_patients=1000)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
criterion = nn.BCELoss()

best_val_recall = 0.0

for epoch in range(EPOCHS):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, edge_index, device)

    # Validate
    val_precision, val_recall = evaluate(model, val_loader, edge_index, device, k=5)

    # Update learning rate
    scheduler.step(val_recall)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val P@5: {val_precision:.4f}")
    print(f"  Val R@5: {val_recall:.4f}")

    # Save best model
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        torch.save(model.state_dict(), 'model.pth')
        print(f"  → Saved best model!")
```

**Learning Rate Scheduling**:
- เริ่มต้น: `lr = 1e-3`
- ถ้า Recall@5 ไม่ดีขึ้นใน 5 epochs → ลด lr เหลือครึ่ง
- ช่วยให้ model converge ดีขึ้น

---

### 💾 Artifacts ที่ได้จาก Training

หลัง training เสร็จจะได้ไฟล์:

1. **`model.pth`**: Model weights (state_dict)
   - ขนาด: ~10-20 MB
   - เก็บ weights ของทุก layer

2. **`symptom_to_idx.pkl`**: Vocabulary mapping
   ```python
   {
       "ไอ": 0,
       "เสมหะ": 1,
       "ปวดท้อง": 2,
       ...
   }
   ```

3. **`idx_to_symptom.pkl`**: Reverse mapping
   ```python
   {
       0: "ไอ",
       1: "เสมหะ",
       2: "ปวดท้อง",
       ...
   }
   ```

4. **`age_bins.pkl`**: Age bin boundaries
   ```python
   [0, 20, 30, 40, 100]
   ```

5. **`graph.pt`**: Co-occurrence graph
   - PyTorch Geometric Data object
   - เก็บ edge_index และ edge_attr

6. **`model_config.pkl`**: Model configuration
   ```python
   {
       'num_symptoms': 338,
       'num_patients': 1000,
       'symptom_embed_dim': 128
   }
   ```

---

## สถาปัตยกรรม Backend API

### 🚀 FastAPI Application (ไฟล์ `app.py`)

#### Startup Process

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """โหลด model และ artifacts เมื่อ server start"""
    global model, symptom_to_idx, idx_to_symptom, age_bins, edge_index, device

    # 1. โหลด artifacts
    with open('symptom_to_idx.pkl', 'rb') as f:
        symptom_to_idx = pickle.load(f)

    with open('idx_to_symptom.pkl', 'rb') as f:
        idx_to_symptom = pickle.load(f)

    with open('age_bins.pkl', 'rb') as f:
        age_bins = pickle.load(f)

    with open('model_config.pkl', 'rb') as f:
        model_config = pickle.load(f)

    graph_data = torch.load('graph.pt', weights_only=False)
    edge_index = graph_data.edge_index

    # 2. Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SymptomRecommender(
        num_symptoms=model_config['num_symptoms'],
        num_patients=model_config['num_patients'],
        symptom_embed_dim=model_config['symptom_embed_dim']
    ).to(device)

    # 3. Load trained weights
    model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
    model.eval()  # Set to evaluation mode

    edge_index = edge_index.to(device)

    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Vocabulary size: {len(symptom_to_idx)}")

    yield  # Server is running

    # Cleanup (if needed)
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)
```

**สิ่งที่เกิดขึ้นเมื่อ start server**:
```
1. [Load] symptom_to_idx.pkl      → RAM: ~50 KB
2. [Load] idx_to_symptom.pkl      → RAM: ~50 KB
3. [Load] age_bins.pkl            → RAM: ~1 KB
4. [Load] model_config.pkl        → RAM: ~1 KB
5. [Load] graph.pt                → RAM: ~5 MB
6. [Init] SymptomRecommender      → RAM: ~20 MB
7. [Load] model.pth weights       → RAM: ~20 MB
8. [Move] to device (CPU/GPU)

Total RAM: ~45 MB
Ready to serve requests! 🚀
```

---

### 📡 API Endpoints

#### 1. `POST /recommend` - แนะนำอาการ

```python
@app.post("/recommend", response_model=RecommendResponse)
async def recommend_symptoms(request: RecommendRequest):
    # Validate input
    if request.gender.lower() not in ['male', 'female']:
        raise HTTPException(400, "Gender must be 'male' or 'female'")

    if not request.symptoms:
        raise HTTPException(400, "At least one symptom required")

    # Check vocabulary
    known_symptoms = []
    unknown_symptoms = []

    for symptom in request.symptoms:
        if symptom in symptom_to_idx:
            known_symptoms.append(symptom)
        else:
            unknown_symptoms.append(symptom)

    if not known_symptoms:
        raise HTTPException(
            400,
            f"None of the symptoms are in vocabulary. Unknown: {unknown_symptoms}"
        )

    # Get recommendations
    recommendations = model.recommend(
        patient_idx=0,  # Dummy index for inference
        gender=request.gender.lower(),
        age=request.age,
        query_symptoms=known_symptoms,
        symptom_to_idx=symptom_to_idx,
        idx_to_symptom=idx_to_symptom,
        edge_index=edge_index,
        age_bins=age_bins,
        top_k=request.top_k,
        device=device
    )

    return RecommendResponse(
        recommendations=recommendations,
        query_symptoms=known_symptoms,
        unknown_symptoms=unknown_symptoms
    )
```

#### 2. `GET /health` - Health check

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "vocabulary_size": len(symptom_to_idx)
    }
```

#### 3. `GET /symptoms?limit=50` - ดู vocabulary

```python
@app.get("/symptoms")
async def list_symptoms(limit: int = 50):
    symptoms = list(symptom_to_idx.keys())[:limit]
    return {
        "symptoms": symptoms,
        "total": len(symptom_to_idx),
        "showing": len(symptoms)
    }
```

---

## Data Flow แบบละเอียด

### 🔄 Request → Response Journey

#### Example Request:

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

---

### Step-by-Step Execution:

#### **Step 1: Request Validation** ⚡️ ~0.1ms

```python
# FastAPI Pydantic validation
request = RecommendRequest(
    gender="male",      # ✓ Valid
    age=26,             # ✓ Valid (0-120)
    symptoms=["ไอ", "เสมหะ"],  # ✓ Valid list
    top_k=5             # ✓ Valid (1-20)
)

# Check gender
if request.gender not in ['male', 'female']:
    raise HTTPException(400, "Invalid gender")

# Check symptoms list
if not request.symptoms:
    raise HTTPException(400, "No symptoms provided")
```

---

#### **Step 2: Vocabulary Lookup** ⚡️ ~0.1ms

```python
known_symptoms = []
unknown_symptoms = []

for symptom in request.symptoms:  # ["ไอ", "เสมหะ"]
    if symptom in symptom_to_idx:
        known_symptoms.append(symptom)
    else:
        unknown_symptoms.append(symptom)

# Result:
# known_symptoms = ["ไอ", "เสมหะ"]
# unknown_symptoms = []
```

---

#### **Step 3: Encode Demographics** ⚡️ ~0.2ms

```python
# 3.1 Gender encoding
gender_vec = torch.zeros(1, 2, device=device)  # [1, 2]
gender_vec[0, 0 if gender == 'male' else 1] = 1.0

# male → [1.0, 0.0]
# Result: [[1.0, 0.0]]

# 3.2 Age bin encoding
age = 26
age_bin_vec = torch.zeros(1, 4, device=device)  # [1, 4]
bin_idx = np.digitize(age, age_bins) - 1
# age_bins = [0, 20, 30, 40, 100]
# 26 falls in bin [20, 30) → bin_idx = 1

age_bin_vec[0, bin_idx] = 1.0
# Result: [[0.0, 1.0, 0.0, 0.0]]
```

---

#### **Step 4: Encode Query Symptoms** ⚡️ ~0.3ms

```python
query_symptoms = ["ไอ", "เสมหะ"]
query_indices = []

for symptom in query_symptoms:
    idx = symptom_to_idx[symptom]
    query_indices.append(idx)

# symptom_to_idx = {"ไอ": 0, "เสมหะ": 1, ...}
# Result: query_indices = [0, 1]

query_tensor = torch.tensor([query_indices], device=device)
# Result: [[0, 1]]  shape=[1, 2]
```

---

#### **Step 5: Model Forward Pass** ⚡️ ~20-50ms (CPU) / ~5-10ms (GPU)

```python
# 5.1 Demographics Encoding
demog_input = torch.cat([gender_vec, age_bin_vec], dim=-1)  # [1, 6]
demog_embed = model.demog_encoder(demog_input)               # [1, 16]

# Linear(6 → 16) + ReLU + Dropout
# Result: [[0.23, -0.45, 0.12, ..., 0.67]]  (16 values)
```

```python
# 5.2 Query Symptom Embeddings
query_embeds = model.symptom_embeddings(query_tensor)  # [1, 2, 128]

# query_tensor = [[0, 1]]  (ไอ, เสมหะ)
# Result shape: [1, 2, 128]
# [
#   [
#     [0.23, -0.45, 0.12, ..., 0.89],  # embedding ของ "ไอ"
#     [0.25, -0.40, 0.15, ..., 0.85]   # embedding ของ "เสมหะ"
#   ]
# ]
```

```python
# 5.3 GAT Propagation
gat_embeddings = model.forward_gat(edge_index)  # [338, 128]

# Layer 1: Multi-head attention
x = model.symptom_embeddings.weight  # [338, 128]
h1 = model.gat1(x, edge_index)       # [338, 512] (4 heads × 128)
h1 = F.elu(h1)
h1 = F.dropout(h1, 0.3)

# Layer 2: Single-head attention
gat_embeddings = model.gat2(h1, edge_index)  # [338, 128]

# ตอนนี้แต่ละอาการได้ embedding ที่ผ่าน graph propagation แล้ว
```

```python
# 5.4 Generate Candidate Symptoms
candidate_indices = [idx for idx in range(338) if idx not in [0, 1]]
# ไม่เอาอาการที่ query มา (ไอ=0, เสมหะ=1)
# candidate_indices = [2, 3, 4, ..., 337]  (336 อาการ)

candidate_tensor = torch.tensor([candidate_indices], device=device)  # [1, 336]
```

```python
# 5.5 Compute CF Scores
patient_idx_tensor = torch.tensor([0], device=device)  # Dummy patient

# Expand for all candidates
patient_idx_expanded = patient_idx_tensor.unsqueeze(1).expand(-1, 336).reshape(-1)  # [336]
demog_embed_expanded = demog_embed.unsqueeze(1).expand(-1, 336, -1).reshape(-1, 16)  # [336, 16]
candidate_flat = candidate_tensor.reshape(-1)  # [336]

# NCF forward
patient_embed = model.patient_embeddings(patient_idx_expanded)  # [336, 128]
symptom_embed = model.symptom_embeddings(candidate_flat)        # [336, 128]

# GMF
gmf_vector = patient_embed * symptom_embed   # [336, 128]
gmf_score = model.gmf_linear(gmf_vector)     # [336, 1]

# MLP
mlp_input = torch.cat([patient_embed, symptom_embed, demog_embed_expanded], dim=-1)  # [336, 272]
mlp_score = model.mlp(mlp_input)  # [336, 1]

# Combine
cf_scores = gmf_score + mlp_score  # [336, 1]
cf_scores = cf_scores.reshape(1, 336)  # [1, 336]
```

```python
# 5.6 Compute CB Scores
query_avg = query_embeds.mean(dim=1, keepdim=True)  # [1, 1, 128]
# Average of ["ไอ", "เสมหะ"] embeddings

candidate_embeds = model.symptom_embeddings(candidate_tensor)  # [1, 336, 128]

# Cosine similarity
query_norm = F.normalize(query_avg, p=2, dim=-1)            # [1, 1, 128]
candidate_norm = F.normalize(candidate_embeds, p=2, dim=-1)  # [1, 336, 128]

cb_scores = (query_norm * candidate_norm).sum(dim=-1)  # [1, 336]
```

```python
# 5.7 Compute Graph Scores
query_gat_embeds = gat_embeddings[query_tensor]  # [1, 2, 128]
query_gat_avg = query_gat_embeds.mean(dim=1, keepdim=True)  # [1, 1, 128]

candidate_gat_embeds = gat_embeddings[candidate_tensor]  # [1, 336, 128]

query_gat_norm = F.normalize(query_gat_avg, p=2, dim=-1)
candidate_gat_norm = F.normalize(candidate_gat_embeds, p=2, dim=-1)

graph_scores = (query_gat_norm * candidate_gat_norm).sum(dim=-1)  # [1, 336]
```

```python
# 5.8 Fusion
cf_norm = torch.sigmoid(cf_scores)        # [0, 1]
cb_norm = (cb_scores + 1) / 2             # [-1, 1] → [0, 1]
graph_norm = (graph_scores + 1) / 2       # [-1, 1] → [0, 1]

# Learnable weights
weights = F.softmax(model.fusion_weights, dim=0)  # [0.4, 0.4, 0.2]

final_scores = (
    weights[0] * cf_norm +
    weights[1] * cb_norm +
    weights[2] * graph_norm
)  # [1, 336]

# Example values:
# Symptom "เจ็บคอ" (idx=45):
#   CF:    0.82
#   CB:    0.75
#   Graph: 0.68
#   Final: 0.4*0.82 + 0.4*0.75 + 0.2*0.68 = 0.764
```

---

#### **Step 6: Top-K Selection** ⚡️ ~0.5ms

```python
top_k = 5
top_scores, top_local_indices = torch.topk(final_scores[0], k=top_k)

# top_scores: [0.853, 0.829, 0.764, 0.721, 0.698]
# top_local_indices: [43, 12, 45, 89, 102]
#   (indices ใน candidate_indices array)

# Map back to actual symptom indices
top_indices = [candidate_indices[idx.item()] for idx in top_local_indices]
# top_indices: [45, 14, 47, 91, 104]

# Map to symptom names
recommendations = [idx_to_symptom[idx] for idx in top_indices]
# recommendations = ["เจ็บคอ", "น้ำมูกไหล", "ไข้", "ปวดศีรษะ", "คัดจมูก"]
```

---

#### **Step 7: Response Construction** ⚡️ ~0.1ms

```python
response = RecommendResponse(
    recommendations=["เจ็บคอ", "น้ำมูกไหล", "ไข้", "ปวดศีรษะ", "คัดจมูก"],
    query_symptoms=["ไอ", "เสมหะ"],
    unknown_symptoms=[]
)

# Convert to JSON
response_json = {
    "recommendations": ["เจ็บคอ", "น้ำมูกไหล", "ไข้", "ปวดศีรษะ", "คัดจมูก"],
    "query_symptoms": ["ไอ", "เสมหะ"],
    "unknown_symptoms": []
}
```

---

### ⏱️ Total Latency Breakdown

| Step | Description | Time (CPU) | Time (GPU) |
|------|-------------|------------|------------|
| 1 | Request Validation | 0.1 ms | 0.1 ms |
| 2 | Vocabulary Lookup | 0.1 ms | 0.1 ms |
| 3 | Encode Demographics | 0.2 ms | 0.1 ms |
| 4 | Encode Query Symptoms | 0.3 ms | 0.2 ms |
| 5 | Model Forward Pass | 20-50 ms | 5-10 ms |
| 6 | Top-K Selection | 0.5 ms | 0.3 ms |
| 7 | Response Construction | 0.1 ms | 0.1 ms |
| **Total** | | **21-51 ms** | **6-11 ms** |

**Bottleneck**: Model forward pass (特に CF score computation)

---

## ตัวอย่างการทำงานจริง

### 📝 Scenario 1: ผู้ป่วยชายอายุ 26 ปี มีอาการ "ไอ"

#### Request:
```json
{
  "gender": "male",
  "age": 26,
  "symptoms": ["ไอ"],
  "top_k": 5
}
```

#### Processing:

1. **Demographics Encoding**:
   - Gender: male → `[1.0, 0.0]`
   - Age: 26 → bin 1 (20-30) → `[0, 1, 0, 0]`
   - Demographics embedding: `[0.23, -0.45, ..., 0.67]` (16D)

2. **Symptom Embedding**:
   - "ไอ" → index 0 → embedding `[0.23, -0.45, 0.12, ...]` (128D)

3. **Scoring** (top candidates):

   | Symptom | CF Score | CB Score | Graph Score | Final Score |
   |---------|----------|----------|-------------|-------------|
   | เสมหะ | 0.89 | 0.92 | 0.88 | **0.897** |
   | เจ็บคอ | 0.85 | 0.78 | 0.85 | **0.828** |
   | น้ำมูกไหล | 0.81 | 0.75 | 0.82 | **0.793** |
   | ไข้ | 0.76 | 0.72 | 0.79 | **0.755** |
   | ปวดศีรษะ | 0.71 | 0.68 | 0.70 | **0.697** |

4. **Why these recommendations?**
   - **เสมหะ**: มักเกิดร่วมกับ "ไอ" (co-occurrence สูง), semantic similarity สูง
   - **เจ็บคอ**: อาการระบบทางเดินหายใจเหมือนกัน
   - **น้ำมูกไหล**: อาการหวัด/ไข้หวัดที่พบร่วมกับไอ
   - **ไข้**: อาการทั่วไปของการติดเชื้อ
   - **ปวดศีรษะ**: อาการแสดงทั่วไป

#### Response:
```json
{
  "recommendations": ["เสมหะ", "เจ็บคอ", "น้ำมูกไหล", "ไข้", "ปวดศีรษะ"],
  "query_symptoms": ["ไอ"],
  "unknown_symptoms": []
}
```

---

### 📝 Scenario 2: ผู้ป่วยหญิงอายุ 35 ปี มีอาการ "ปวดท้อง", "คลื่นไส้"

#### Request:
```json
{
  "gender": "female",
  "age": 35,
  "symptoms": ["ปวดท้อง", "คลื่นไส้"],
  "top_k": 5
}
```

#### Processing:

1. **Demographics**:
   - Gender: female → `[0.0, 1.0]`
   - Age: 35 → bin 2 (30-40) → `[0, 0, 1, 0]`

2. **Query Embeddings**:
   - "ปวดท้อง" → `[0.15, 0.42, -0.23, ...]`
   - "คลื่นไส้" → `[0.18, 0.38, -0.20, ...]`
   - Average: `[0.165, 0.40, -0.215, ...]`

3. **Top Recommendations**:

   | Symptom | Reason |
   |---------|--------|
   | อาเจียน | มักเกิดร่วมกับคลื่นไส้ (co-occurrence 89%) |
   | ท้องเสีย | อาการระบบทางเดินอาหารเหมือนกัน |
   | จุกแน่นท้อง | semantic similarity กับปวดท้อง |
   | ไข้ | อาการทั่วไปของการติดเชื้อทางเดินอาหาร |
   | ปวดศีรษะ | อาการแสดงทั่วไป |

#### Response:
```json
{
  "recommendations": ["อาเจียน", "ท้องเสีย", "จุกแน่นท้อง", "ไข้", "ปวดศีรษะ"],
  "query_symptoms": ["ปวดท้อง", "คลื่นไส้"],
  "unknown_symptoms": []
}
```

---

### 📝 Scenario 3: Unknown Symptom Handling

#### Request:
```json
{
  "gender": "male",
  "age": 45,
  "symptoms": ["ท้องแสบ", "ปวดท้อง"],  // "ท้องแสบ" ไม่อยู่ใน vocab
  "top_k": 5
}
```

#### Processing:

1. **Vocabulary Check**:
   - "ท้องแสบ" → ❌ Not in vocabulary
   - "ปวดท้อง" → ✅ In vocabulary (index 42)

2. **Proceed with known symptoms**: `["ปวดท้อง"]`

3. **Generate recommendations**: (same as normal)

#### Response:
```json
{
  "recommendations": ["คลื่นไส้", "จุกแน่นท้อง", "อาเจียน", "ท้องเสีย", "ท้องอืด"],
  "query_symptoms": ["ปวดท้อง"],
  "unknown_symptoms": ["ท้องแสบ"]  // ⚠️ Warning: unknown symptom
}
```

---

### 📝 Scenario 4: All Unknown Symptoms (Error)

#### Request:
```json
{
  "gender": "female",
  "age": 28,
  "symptoms": ["ท้องแสบ", "ปวดหัว"],  // Both not in vocab
  "top_k": 5
}
```

#### Response:
```json
{
  "detail": "None of the provided symptoms are in the vocabulary. Unknown: ['ท้องแสบ', 'ปวดหัว']"
}
```
**HTTP Status**: 400 Bad Request

---

## สรุป

### 🎯 จุดเด่นของสถาปัตยกรรม

1. **Hybrid Approach**:
   - CF: เรียนรู้จากพฤติกรรมผู้ป่วย
   - CB: จับความคล้ายของอาการ
   - Graph: จับ co-occurrence patterns

2. **Scalable**:
   - รองรับผู้ป่วยจำนวนมาก (batch processing)
   - Fast inference (~20-50ms บน CPU)

3. **Interpretable**:
   - ดูได้ว่าแต่ละสายให้คะแนนเท่าไร
   - Attention weights บอกความสัมพันธ์

4. **Production-Ready**:
   - FastAPI with async support
   - Error handling ครบถ้วน
   - API documentation (Swagger UI)

---

### 🔧 การปรับแต่งในอนาคต

1. **More Training Data**:
   - เพิ่มข้อมูลผู้ป่วย → ปรับปรุง CF accuracy

2. **Text Embeddings**:
   - ใช้ BERT/Thai-BERT สำหรับ symptom text
   - จับ semantic similarity ดีขึ้น

3. **Severity Modeling**:
   - แนะนำอาการพร้อม severity level

4. **Multi-modal Input**:
   - รับรูปภาพ/เสียงประกอบ

5. **Explainability**:
   - อธิบายว่าทำไมแนะนำอาการนี้

---

### 📚 เอกสารอ้างอิง

- **NCF**: He et al. (2017) "Neural Collaborative Filtering"
- **GAT**: Veličković et al. (2018) "Graph Attention Networks"
- **PyTorch Geometric**: Fey & Lenssen (2019)
- **FastAPI**: Sebastián Ramírez

---

**เอกสารนี้อธิบายสถาปัตยกรรมและการทำงานของระบบแนะนำอาการทางการแพทย์อย่างละเอียด** 🎓
