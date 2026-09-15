# 🛡️ HALAL-BERT: Fine-Tuned BERT-LoRA Dietary Detection System & Web Application

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Umair1710%2FBert--Lora--Finedtuned--Hala__Haram__Detection-yellow.svg)](https://huggingface.co/Umair1710/Bert-Lora-Finedtuned-Hala_Haram_Detection)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20PEFT%20LoRA-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An ultra-modern, recruiter-ready full-stack web application integrating **Umair Naveed's** Parameter-Efficient Fine-Tuned (PEFT) LoRA model on `bert-base-uncased` for fine-grained Halal/Haram food classification.

Developed as a **Final Year Project (FYP) Core Implementation Engine**.

---

## 🌟 Key Features & Capabilities

- **State-of-the-Art Dietary NLP**: Powered by `Umair1710/Bert-Lora-Finedtuned-Hala_Haram_Detection`.
- **0.98 Haram Recall Benchmark**: Designed specifically with high sensitivity to prohibited dietary triggers (pork derivatives, wine, alcohol, animal shortening, cochineal) with near-zero false-Halal risk.
- **Contextual Counterfactual Nuance**: Understands fine distinctions that collapse naive keyword search (e.g. *raw wine* = **HARAM** vs *white wine vinegar* = **HALAL**; *pork bacon* = **HARAM** vs *soy bacon* = **HALAL**).
- **Interactive Testing Studio**:
  1. **Single Item Scanner**: Instant natural language inference with animated probability gauge.
  2. **Multi-Ingredient Label Inspector**: Paste full commercial packaging labels; parses individual ingredients and computes whole-product compliance.
  3. **Differential Counterfactual Test Lab**: Side-by-side comparison of trigger words vs contextual negations.
  4. **Recruiter Benchmark Presets**: 1-click test buttons for complex edge cases.
- **Research & Metrics Showcase**: Live metrics breakdown (95.81% accuracy, 0.9464 Macro F1, 296K active parameters vs 110M frozen base).
- **Recruiter Report Exporter**: Instant summary generator and raw JSON API inspector.

---

## 📊 Model Performance Matrix

Evaluated across an uncorrupted, isolated test footprint:

| Metric Type | Base Un-Tuned BERT Baseline | Fine-Tuned LoRA Adapter | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 24.96% | **95.81%** | **+70.85% Absolute** |
| **Test F1-Macro** | 0.1998 | **0.9464** | **+373% Gain** |
| **Test Precision-Macro** | 0.1248 | **0.9302** | Optimal Specificity |
| **Critical Haram Recall** | 0.5000 | **0.9800** | **Critical Safety Guarantee** |
| **Active Parameters** | 110M (100%) | **296,450 (0.27%)** | Ultra-Lightweight Footprint |

---

## 🚀 Quickstart: Running Locally

### Prerequisites
- Python 3.10+
- `pip`

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
./run.sh
# or
python3 app.py
```
Open your browser and navigate to:
```
http://localhost:7860
```
*(Interactive API documentation is available at `http://localhost:7860/docs`)*

---

## 🌐 Deploying to Hugging Face Spaces (Free Live URL for Recruiters)

You can host this entire web app live on Hugging Face Spaces for free with your own URL:

1. **Create a Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Enter Space name (e.g. `halal-haram-detector`).
   - Under **Space SDK**, select **Docker** (Blank template).
   - Click **Create Space**.

2. **Push the Code**:
   In your local terminal, initialize git and push to your space:
   ```bash
   git init
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/halal-haram-detector
   git add .
   git commit -m "Deploy HALAL-BERT Web Application"
   git push space main --force
   ```

3. **Live Web App**:
   Hugging Face Spaces will automatically build the `Dockerfile` and launch the web app with a permanent link like:
   `https://huggingface.co/spaces/YOUR_USERNAME/halal-haram-detector`

You can share this live link directly on your resume, LinkedIn, or portfolio for recruiters!

---

## 🔌 API Endpoints

### 1. Single Item Classification
- **Endpoint**: `POST /api/predict`
- **Request Body**:
  ```json
  { "text": "gourmet salad with white wine vinegar" }
  ```
- **Response**:
  ```json
  {
    "text": "gourmet salad with white wine vinegar",
    "prediction": "HALAL",
    "label_id": 0,
    "confidence": 99.98,
    "probabilities": { "halal": 99.98, "haram": 0.02 },
    "latency_ms": 28.4,
    "engine": "BERT-LoRA (cpu)"
  }
  ```

### 2. Multi-Ingredient Label Audit
- **Endpoint**: `POST /api/batch-predict`
- **Request Body**:
  ```json
  { "text": "Enriched flour, water, sugar, pork gelatin, salt" }
  ```
- **Response**:
  ```json
  {
    "overall_prediction": "HARAM",
    "overall_confidence": 99.85,
    "total_ingredients_scanned": 5,
    "haram_ingredients_detected": 1,
    "flagged_ingredients": ["pork gelatin (99.85%)"],
    "ingredient_breakdown": [ ... ]
  }
  ```

### 3. Model Telemetry & Benchmarks
- **Endpoint**: `GET /api/model-info`

---

## ✍️ Author & Citation

**Author:** Umair Naveed ([@Umair1710](https://huggingface.co/Umair1710))

If you use this system or dataset methodology in your research, please cite:

```bibtex
@misc{umair2026berthelalharam,
  author       = {Umair Naveed},
  title        = {BERT-LoRA Fine-Tuned Halal/Haram Detection System},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Umair1710/Bert-Lora-Finedtuned-Hala_Haram_Detection}},
  note         = {Final Year Project (FYP) Core Implementation Engine}
}
```
