# LLM Evaluation Methodology & Verification Report

**Generated Evaluation Timestamp:** 2026-05-26 03:51:05

## 📊 Experimental Dataset Context
- **Sample Population ($N$):** 300 transaction sequences evaluated serially.
- **Ground Truth Target Breakdown:** High Risk: 141 records | Low Risk: 159 records.

## 📈 Aggregated Statistical Performance Indicators

| Model Variant Profile | Average Faithfulness (Grounded Logic) | Average Answer Relevancy |
| :--- | :---: | :---: |
| **Base Baseline (`llama3.2:latest`)** | 0.6380 | 0.6152 |
| **Fine-Tuned Candidate (`fraud-model-v4:latest`)** | 0.7973 | 0.7457 |
| **Net Structural System Delta ($\Delta$)** | **++0.1593** | **++0.1305** |
