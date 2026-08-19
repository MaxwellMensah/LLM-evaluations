# 🛡️ Fraud Detection LLM Evaluation & Visualizer Framework

A lightweight, local framework for running automated evaluation and comparative performance visualization of **Base vs. Fine-Tuned** LLMs (e.g., `llama3.2:latest` vs. `fraud-model-v4:latest`). 

Built specifically to run **100% locally on Ollama**, bypassing heavy RAGAS framework constraints and eliminating JSON schema parsing failures from smaller models.

---

## 📁 Repository Structure
```text
.
├── Modelfile                        # Ollama configuration/system prompt definition for fine-tuned model
├── Deployfile                       # Deployment automation configuration
├── fraud_detection_dataset_V4.jsonl # Holdout evaluation dataset (ShareGPT format)
│
├── eval_llms_v5.py                  # [MAIN] Lightweight evaluation script (replaces heavy RAGAS pipeline)
├── eval_llm_visualizer.py           # [MAIN] Interactive Plotly visualizer script
│
├── llm_evals.py                     # Legacy evaluation script (Version 1)
├── ragas_eval_v4.py                 # Legacy RAGAS evaluation script (Version 4)
├── eval_ragas_old.py                # Deprecated RAGAS template (kept for historical benchmark comparison)
│
├── eval_scores_aggregated.csv       # Summary metric averages across test models
├── eval_scores_base.csv             # Raw per-question scores for the Base Model
├── eval_scores_ft.csv               # Raw per-question scores for the Fine-Tuned Model
├── eval_scores_combined.csv         # Merged dataset used for delta analysis & visualizer
│
├── faithfulness_relevancy_scatter.html # Interactive Plotly scatter plot (Scatter comparison)
├── per_question_deltas_heatmap.html    # Interactive Plotly heatmap (Slice optimization deltas)
├── confusion_matrix.html              # Interactive Plotly confusion matrix (Classification accuracy)
└── methodology.md                   # Auto-generated analytical report & methodology summary

```

---

## 💡 Key Architectural Changes: Why No RAGAS?

Traditional evaluation suites like RAGAS rely on multi-stage prompt chaining and strict JSON schema outputs. For local models ($\le 8\text{B}$ parameters), JSON parsing fails between 15% to 30% of the time—causing script crashes and `NaN` evaluation gaps.

Our custom evaluation pipeline replaces RAGAS with zero-failure local metrics:

1. **Faithfulness (0.0–1.0):** Evaluated by a local Judge LLM (`llama3.2:latest`) asking whether the answer strictly derives from context. Outputs direct numeric string matching without requiring structured JSON.
2. **Answer Relevancy (0.0–1.0):** Computed via direct **Cosine Similarity** between question embeddings and answer embeddings via `nomic-embed-text:latest`. Completely bypasses LLM inference for 100% deterministic reliability.

---

## 🚀 Quickstart Guide

### Prerequisites

* **Ollama** running locally on default port `http://localhost:11434`.
* Installed models:
```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
# Ensure your custom fine-tuned model tag is available:
# fraud-model-v4:latest

```



### Python Dependencies

Install required dependencies:

```bash
pip install pandas requests tqdm plotly scikit-learn numpy

```

---

## 🏃 Running the Evaluation & Visualization Pipeline

### Step 1: Run LLM Evaluation Pipeline

Run `eval_llms_v5.py` to sample test instances from `fraud_detection_dataset_V4.jsonl`, generate responses from both models, and score performance metrics.

```bash
python eval_llms_v5.py

```

> **Note:** I set `SAMPLE_SIZE = 300` in `eval_llms_v5.py` for a quick test, since its computationally demanding. For full benchmarks, use a higher number

**Output Artifacts Generated:**

* `eval_scores_aggregated.csv`
* `eval_scores_base.csv`
* `eval_scores_ft.csv`
* `eval_scores_combined.csv`

---

### Step 2: Generate Interactive Visualizations & Summary Report

Run `eval_llm_visualizer.py` to ingest the evaluation outputs and generate interactive Plotly HTML dashboards and markdown documentation:

```bash
python eval_llm_visualizer.py

```

**Output Graphics & Documentation:**

* **`faithfulness_relevancy_scatter.html`**: Large-canvas interactive scatter plot comparing base vs fine-tuned distributions with crosshair mean overlays.
* **`per_question_deltas_heatmap.html`**: Heatmap slice highlighting per-question performance improvements.
* **`confusion_matrix.html`**: Detailed confusion matrix tracking risk prediction classification ("High Risk" vs "Low Risk").
* **`methodology.md`**: Executive markdown report with comparative statistical summaries.

---

## 📊 Example Baseline Benchmark Performance

| Model Variant Profile | Average Faithfulness | Average Answer Relevancy | Sample Size ($N$) |
| --- | --- | --- | --- |
| **Base Baseline (`llama3.2:latest`)** | 0.6380 | 0.6152 | 300 |
| **Fine-Tuned Candidate (`fraud-model-v4:latest`)** | **0.7973** | **0.7457** | 300 |
| **Net Structural System Delta ($\Delta$)** | **+0.1593** | **+0.1305** | — |

```