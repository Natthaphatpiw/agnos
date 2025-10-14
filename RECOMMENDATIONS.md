# Overfitting Analysis Results & Recommendations

## Analysis Summary

**Date:** 2024
**Model:** SymptomRecommender (Hybrid: CF + CB + Graph)
**Dataset:** 1,000 patients, 338 symptoms

---

## Results

### Performance Metrics

| Split      | Precision@5 | Recall@5 | Precision@10 | Recall@10 |
|------------|-------------|----------|--------------|-----------|
| Train      | 8.58%       | 28.76%   | 4.85%        | 32.02%    |
| Validation | 7.20%       | 26.17%   | 4.40%        | 30.67%    |
| Test       | 8.00%       | 27.50%   | 4.40%        | 30.00%    |

### Overfitting Indicators

| Gap Type           | Precision@5 | Recall@5 |
|--------------------|-------------|----------|
| Train vs Val       | +1.37%      | +2.59%   |
| Train vs Test      | +0.58%      | +1.26%   |

**Maximum Gap:** 2.6% (threshold: 10%)

---

## Diagnosis

### ✅ **Overfitting Status: GOOD**

- Model generalizes well to unseen data
- Train-validation gap is well below threshold (2.6% < 10%)
- No signs of memorization

### ⚠️ **Performance Status: LOW**

- Recall@5 ~27% (target: >50%)
- Precision@5 ~8% (target: >30%)
- Model is underfitting rather than overfitting

---

## Root Cause Analysis

### Why Performance is Low

1. **Sparse Data Problem**
   - 338 symptoms × 1,000 patients = 338,000 possible interactions
   - Actual interactions: ~2,500 (< 1% density)
   - Many symptoms have very few examples

2. **Over-Regularization**
   - Current settings prevent overfitting successfully
   - But also prevent the model from learning complex patterns
   - Dropout (0.3) + Weight Decay (1e-5) is too conservative

3. **Limited Training**
   - 50 epochs might not be enough
   - Early stopping may trigger too early

---

## Recommendations

### **PRIMARY: Fine-tune Current Model (Recommended)**

Since the model is NOT overfitting, we can safely:

#### 1. Reduce Regularization
```python
# Current
dropout = 0.3
weight_decay = 1e-5

# Recommended
dropout = 0.2          # Less dropout
weight_decay = 5e-6    # Less weight decay
```

#### 2. Train Longer
```python
# Current
epochs = 50
patience = 5

# Recommended
epochs = 80            # More training
patience = 15          # More patient early stopping
```

#### 3. Adjust Learning Rate Schedule
```python
# More patient scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    patience=8,   # Instead of 5
    factor=0.5
)
```

**How to use:**
```bash
python train_improved.py
```

**Expected improvement:**
- Recall@5: 27% → 35-40%
- Precision@5: 8% → 12-15%

---

### **ALTERNATIVE 1: Data Augmentation**

If fine-tuning doesn't help enough, try augmenting data:

#### Technique 1: Symptom Synonyms
```python
symptom_synonyms = {
    "ไอ": ["ho", "cough", "ไอแห้ง"],
    "ปวดท้อง": ["abdominal pain", "stomachache", "ปวดหน้าท้อง"],
    # ...
}

# Duplicate samples with synonyms
augmented_data = []
for patient in patients:
    augmented_data.append(patient)

    # Create variant with synonyms
    variant = patient.copy()
    for i, symptom in enumerate(variant['symptoms']):
        if symptom in symptom_synonyms:
            variant['symptoms'][i] = random.choice(symptom_synonyms[symptom])

    augmented_data.append(variant)
```

#### Technique 2: Mix Patients
```python
# Combine symptoms from similar patients
def mix_patients(patient1, patient2, alpha=0.5):
    """Mix two patients to create synthetic patient"""
    mixed_symptoms = set(patient1['symptoms'])

    # Add some symptoms from patient2
    for symptom in patient2['symptoms']:
        if random.random() < alpha:
            mixed_symptoms.add(symptom)

    return create_patient(mixed_symptoms)
```

**Expected improvement:**
- Training samples: 1,000 → 3,000-5,000
- Recall@5: 27% → 40-45%

---

### **ALTERNATIVE 2: Ensemble Methods**

Combine multiple models to improve performance:

```python
class EnsembleRecommender:
    def __init__(self):
        self.model1 = SymptomRecommender(seed=42)
        self.model2 = SymptomRecommender(seed=123)
        self.model3 = SymptomRecommender(seed=456)

    def recommend(self, query):
        scores1 = self.model1.predict(query)
        scores2 = self.model2.predict(query)
        scores3 = self.model3.predict(query)

        # Average scores
        final_scores = (scores1 + scores2 + scores3) / 3

        return get_top_k(final_scores)
```

**Expected improvement:**
- Recall@5: 27% → 32-35%
- More stable predictions

---

### **ALTERNATIVE 3: Collect More Data**

Most effective but requires time:

| Data Size | Expected Recall@5 | Expected Precision@5 |
|-----------|-------------------|----------------------|
| 1,000     | 27%              | 8%                   |
| 5,000     | 45-50%           | 15-20%               |
| 10,000    | 55-60%           | 25-30%               |
| 50,000+   | 65-70%           | 35-40%               |

---

## Action Plan

### Phase 1: Quick Wins (1-2 hours)

1. ✅ Run improved training
   ```bash
   python train_improved.py
   ```

2. ✅ Compare results
   ```bash
   python check_overfitting.py
   ```

3. ✅ If Recall@5 > 35%, deploy improved model

### Phase 2: If Phase 1 Insufficient (1 week)

1. Implement data augmentation
2. Retrain with augmented data
3. Target: Recall@5 > 40%

### Phase 3: Long-term (ongoing)

1. Collect more real patient data
2. Continuously retrain as data grows
3. Monitor performance monthly

---

## Monitoring Checklist

After deploying improved model, monitor:

- [ ] Recall@5 on validation set (target: >35%)
- [ ] Train-val gap (keep <10%)
- [ ] API response time (keep <100ms)
- [ ] User feedback on recommendation quality

---

## Conclusion

**Current Status:**
- ✅ Model is NOT overfitting (good generalization)
- ⚠️ Model performance is low (underfitting)

**Next Step:**
- Use `train_improved.py` to reduce regularization
- This will allow model to learn more complex patterns
- Monitor to ensure overfitting doesn't start

**Expected Outcome:**
- Recall@5: 27% → 35-40%
- Maintain good generalization (gap <10%)

---

**Note:** The low performance is primarily due to sparse data (1,000 samples for 338 symptoms), not model architecture. The current hybrid approach is sound, it just needs more data or reduced regularization to perform better.
