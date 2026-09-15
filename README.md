---
title: HALAL-BERT Classifier
emoji: 🛡️
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
---

# 🛡️ HALAL-BERT: Fine-Tuned BERT-LoRA Dietary Detection System

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Umair1710%2FBert--Lora--Finedtuned--Hala__Haram__Detection-yellow.svg)](https://huggingface.co/Umair1710/Bert-Lora-Finedtuned-Hala_Haram_Detection)
[![Gradio Space](https://img.shields.io/badge/Demo-Gradio%20Space-orange.svg)](https://huggingface.co/spaces)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20PEFT%20LoRA-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An ultra-modern, recruiter-ready dietary classification application powered by **Umair Naveed's** Parameter-Efficient Fine-Tuned (PEFT) LoRA model on `bert-base-uncased`.

Developed as a **Final Year Project (FYP) Core Implementation Engine**.

---

## 🌟 Key Features & Capabilities

- **State-of-the-Art Dietary NLP**: Powered by `Umair1710/Bert-Lora-Finedtuned-Hala_Haram_Detection`.
- **0.98 Critical Haram Recall Benchmark**: Engineered with high sensitivity to prohibited dietary triggers (pork derivatives, wine, alcohol, animal shortening, cochineal) with near-zero false-Halal risk.
- **Contextual Counterfactual Nuance**: Understands fine distinctions that collapse naive keyword search (e.g. *raw wine* = **HARAM** vs *white wine vinegar* = **HALAL**; *pork bacon* = **HARAM** vs *soy bacon* = **HALAL**).
- **Interactive Testing Studio**:
  1. **Single Item Scanner**: Instant natural language inference with animated probability gauge.
  2. **Multi-Ingredient Label Inspector**: Paste full commercial packaging labels; parses individual ingredients and computes whole-product compliance.
  3. **Differential Counterfactual Test Lab**: Side-by-side comparison of trigger words vs contextual negations.
  4. **Recruiter Benchmark Presets**: 1-click test buttons for complex edge cases.
- **Research & Metrics Showcase**: Live metrics breakdown (95.81% accuracy, 0.9464 Macro F1, 296K active parameters vs 110M frozen base).

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

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:7860`.

---

## ☁️ Deploying to Hugging Face Spaces (100% Free - No Credit Card Needed)

1. Go to **[huggingface.co/new-space](https://huggingface.co/new-space)**.
2. Under **Space Name**, type: `halal-haram-classifier`.
3. Under **Space SDK**, select **Gradio** (Default Free 16 GB RAM CPU tier).
4. Click **Create Space**.
5. Connect your GitHub repository or push via git:
   ```bash
   git remote add space https://huggingface.co/spaces/Umair1710/halal-haram-classifier
   git push space main --force
   ```
6. Hugging Face will automatically run `python app.py` and provide you with a permanent public link:
   `https://huggingface.co/spaces/Umair1710/halal-haram-classifier`

---

## ✍️ Author & Citation

**Author:** Umair Naveed ([@Umair1710](https://huggingface.co/Umair1710))

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
