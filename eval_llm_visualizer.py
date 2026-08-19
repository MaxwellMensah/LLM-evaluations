# #!/usr/bin/env python3
# """
# Comprehensive evaluation visualisation for fraud detection LLM.
# Tailored for custom cosine/structural scoring outputs.
# """

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Set style for professional presentation graphics
# sns.set_theme(style="whitegrid")
# plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# # ----------------------------------------------------------------------
# # 1. Load Custom Evaluation Data
# # ----------------------------------------------------------------------
# COMBINED_CSV = "eval_scores_combined.csv"
# AGGREGATED_CSV = "eval_scores_aggregated.csv"

# if not os.path.exists(COMBINED_CSV):
#     print(f"✗ Error: {COMBINED_CSV} not found! Run your evaluation script first.")
#     exit(1)

# df = pd.read_csv(COMBINED_CSV)
# print(f"✓ Successfully loaded {len(df)} evaluated rows from {COMBINED_CSV}")

# # Clean up any potential column naming variations from custom scripts
# # Expected base names: faithfulness_base, faithfulness_ft, answer_relevancy_base, answer_relevancy_ft
# # Expected text columns: question, ground_truth, generated_answer_base, generated_answer_ft

# # ----------------------------------------------------------------------
# # 2. Data Cleaning & Label Extraction
# # ----------------------------------------------------------------------
# def extract_label(text):
#     """Safely extracts risk decisions for the confusion matrix validation."""
#     text = str(text).strip()
#     if "High Risk" in text or "Conclusion: High Risk" in text: 
#         return "High Risk"
#     if "Low Risk" in text or "Conclusion: Low Risk" in text: 
#         return "Low Risk"
#     return "Unknown"

# # Extract metrics safely from text predictions
# df["ground_truth_label"] = df["ground_truth"].apply(extract_label)
# df["predicted_label_base"] = df["generated_answer_base"].apply(extract_label)
# df["predicted_label_ft"] = df["generated_answer_ft"].apply(extract_label)

# print(f"DEBUG - Unique Ground Truth Labels:  {df['ground_truth_label'].unique()}")
# print(f"DEBUG - Unique Fine-Tuned Labels:   {df['predicted_label_ft'].unique()}")

# # Filter out unparseable rows exclusively for classification graphics
# valid_ft_df = df[
#     (df["ground_truth_label"].isin(["Low Risk", "High Risk"])) & 
#     (df["predicted_label_ft"].isin(["Low Risk", "High Risk"]))
# ].copy()

# # ----------------------------------------------------------------------
# # 3. Visualization 1: Faithfulness vs Answer Relevancy Scatter Plot
# # ----------------------------------------------------------------------
# plt.figure(figsize=(10, 6.5))
# plt.scatter(df["answer_relevancy_base"], df["faithfulness_base"], 
#             alpha=0.4, label="Base Model (llama3.2:latest)", c='#1f77b4', marker='o', edgecolors='w', s=50)
# plt.scatter(df["answer_relevancy_ft"], df["faithfulness_ft"], 
#             alpha=0.6, label="Fine-Tuned Model (fraud-model-v4)", c='#d62728', marker='X', edgecolors='w', s=60)

# # Add centroid crosshairs for visual average references
# plt.axhline(df["faithfulness_base"].mean(), color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
# plt.axvline(df["answer_relevancy_base"].mean(), color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
# plt.axhline(df["faithfulness_ft"].mean(), color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5)
# plt.axvline(df["answer_relevancy_ft"].mean(), color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5)

# plt.xlabel("Answer Relevancy Score")
# plt.ylabel("Faithfulness Score")
# plt.title("Per-Question Quality Comparison: Base vs. Fine-Tuned")
# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none')
# plt.tight_layout()
# plt.savefig("faithfulness_relevancy_scatter.png", dpi=150)
# plt.close()
# print("Saved performance metric scatter plot: faithfulness_relevancy_scatter.png")

# # ----------------------------------------------------------------------
# # 4. Visualization 2: Heatmap of Performance Deltas (First 25 Rows)
# # ----------------------------------------------------------------------
# df["faithfulness_delta"] = df["faithfulness_ft"] - df["faithfulness_base"]
# df["relevancy_delta"] = df["answer_relevancy_ft"] - df["answer_relevancy_base"]

# plt.figure(figsize=(11, 7.5))
# delta_subset = df[["faithfulness_delta", "relevancy_delta"]].iloc[:25]
# sns.heatmap(delta_subset, annot=True, cmap="RdYlGn", center=0, fmt=".3f", 
#             cbar_kws={"label": "Net Improvement Score (Fine-Tuned - Base)"})
# plt.title("Sample Slice Optimization Analysis (First 25 Evaluation Points)")
# plt.xlabel("Metrics Layer")
# plt.ylabel("Transaction Reference Index")
# plt.yticks(np.arange(25) + 0.5, [f"Log Query {i+1}" for i in range(25)], rotation=0)
# plt.tight_layout()
# plt.savefig("per_question_deltas_heatmap.png", dpi=150)
# plt.close()
# print("Saved improvement delta map: per_question_deltas_heatmap.png")

# # ----------------------------------------------------------------------
# # 5. Visualization 3: Confusion Matrix for Risk Classification
# # ----------------------------------------------------------------------
# if not valid_ft_df.empty:
#     plt.figure(figsize=(7, 6))
#     labels_order = ["Low Risk", "High Risk"]
#     cm = confusion_matrix(valid_ft_df["ground_truth_label"], valid_ft_df["predicted_label_ft"], labels=labels_order)
    
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
#     disp.plot(cmap="Reds", ax=plt.gca(), values_format='d')
#     plt.title("Confusion Matrix: Fine-Tuned Risk Predictions")
#     plt.grid(False)
#     plt.tight_layout()
#     plt.savefig("confusion_matrix.png", dpi=150)
#     plt.close()
#     print("Saved classification matrix visual: confusion_matrix.png")
# else:
#     print("⚠ Warning: Skipping Confusion Matrix. Format keywords ('High Risk' / 'Low Risk') absent from texts.")

# # ----------------------------------------------------------------------
# # 6. Dynamic Methodology & Technical Writeup Production
# # ----------------------------------------------------------------------
# base_faith_avg = df["faithfulness_base"].mean()
# ft_faith_avg = df["faithfulness_ft"].mean()
# base_rel_avg = df["answer_relevancy_base"].mean()
# ft_rel_avg = df["answer_relevancy_ft"].mean()

# total_samples = len(df)
# high_risk_count = sum(1 for label in df["ground_truth_label"] if label == "High Risk")
# low_risk_count = sum(1 for label in df["ground_truth_label"] if label == "Low Risk")

# methodology_content = f"""# LLM Evaluation Methodology & Verification Report

# **Generated Evaluation Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# ## 📊 Experimental Dataset Context
# - **Source Artifact:** Comprehensive pipeline validation holdout extracted from `fraud_detection_dataset_V4.jsonl`.
# - **Sample Population ($N$):** {total_samples} transaction sequences evaluated serially.
# - **Ground Truth Target Breakdown:** High Risk: {high_risk_count} records | Low Risk: {low_risk_count} records | Complex/Unparsed: {total_samples - (high_risk_count + low_risk_count)} records.

# ## 🛠️ Verification Topology & Systems Profile
# - **Base Baseline Engine:** `llama3.2:latest` (Standard Unaligned Base Weights)
# - **Production Fine-Tuned Candidate:** `fraud-model-v4:latest` 
# - **Vector Embedding Core Engine:** `nomic-embed-text:latest`
# - **Hardware Profile Context:** Processing localized completely on host execution resources (CPU-Bound execution constraints mapped out natively).

# ## 📈 Aggregated Statistical Performance Indicators

# | Model Variant Profile | Average Faithfulness (Grounded Logic) | Average Answer Relevancy |
# | :--- | :---: | :---: |
# | **Base Baseline (`llama3.2:latest`)** | {base_faith_avg:.4f} | {base_re_avg if 'base_re_avg' in locals() else base_rel_avg:.4f} |
# | **Fine-Tuned Candidate (`fraud-model-v4:latest`)** | {ft_faith_avg:.4f} | {ft_rel_avg:.4f} |
# | **Net Structural System Delta ($\Delta$)** | **+{ft_faith_avg - base_faith_avg:+.4f}** | **+{ft_rel_avg - base_rel_avg:+.4f}** |

# ## 💡 Principal Analytical Insights
# 1. **Structural Reasoning Synthesis:** The fine-tuned candidate (`fraud-model-v4:latest`) shows a major improvement in **Faithfulness (+{(ft_faith_avg - base_faith_avg)*100:.1f}%)**. This confirms the adapter weights successfully restricted the model's tendency to hallucinate generic compliance boilerplate, forcing it to stick strictly to the structured payload evidence.
# 2. **Relevancy Vector Alignment:** Unlike standard generalized fine-tuning runs where precision formatting drops alignment vectors, your adapter model unlocked a **+{(ft_rel_avg - base_rel_avg)*100:.1f}% surge in Relevancy**. This proves that formatting enforcement (Step 1 -> Step 2 -> Conclusion) directly improved conversational precision.
# """

# with open("methodology.md", "w", encoding="utf-8") as f:
#     f.write(methodology_content)
# print("Saved production quality methodology review: methodology.md")
# print("\n✅ All visual plots and statistical report modules built successfully.")


#!/usr/bin/env python3
"""
Comprehensive evaluation visualisation for fraud detection LLM.
Built entirely in Plotly for high-resolution interactive charts.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------
# 1. Load Custom Evaluation Data
# ----------------------------------------------------------------------
COMBINED_CSV = "eval_scores_combined.csv"

if not os.path.exists(COMBINED_CSV):
    print(f"✗ Error: {COMBINED_CSV} not found! Run your evaluation script first.")
    exit(1)

df = pd.read_csv(COMBINED_CSV)
print(f"✓ Successfully loaded {len(df)} evaluated rows from {COMBINED_CSV}")

# ----------------------------------------------------------------------
# 2. Data Cleaning & Label Extraction
# ----------------------------------------------------------------------
def extract_label(text):
    """Safely extracts risk decisions for the confusion matrix validation."""
    text = str(text).strip()
    if "High Risk" in text or "Conclusion: High Risk" in text: 
        return "High Risk"
    if "Low Risk" in text or "Conclusion: Low Risk" in text: 
        return "Low Risk"
    return "Unknown"

df["ground_truth_label"] = df["ground_truth"].apply(extract_label)
df["predicted_label_ft"] = df["generated_answer_ft"].apply(extract_label)

# Filter out unparseable rows exclusively for classification graphics
valid_ft_df = df[
    (df["ground_truth_label"].isin(["Low Risk", "High Risk"])) & 
    (df["predicted_label_ft"].isin(["Low Risk", "High Risk"]))
].copy()

# ----------------------------------------------------------------------
# 3. Plotly Chart 1: Large Interactive Scatter Plot
# ----------------------------------------------------------------------
print("Generating Scatter Plot...")
fig_scatter = go.Figure()

# Base Model Scatter Traces
fig_scatter.add_trace(go.Scatter(
    x=df["answer_relevancy_base"],
    y=df["faithfulness_base"],
    mode='markers',
    name='Base Model (llama3.2:latest)',
    marker=dict(size=10, color='#1f77b4', opacity=0.5, line=dict(width=1, color='White')),
    text=[f"Q: {q[:60]}..." for q in df["question"]],
    hovertemplate="<b>Base Model Performance</b><br>Relevancy: %{x:.4f}<br>Faithfulness: %{y:.4f}<br>%{text}<extra></extra>"
))

# Fine-Tuned Model Scatter Traces
fig_scatter.add_trace(go.Scatter(
    x=df["answer_relevancy_ft"],
    y=df["faithfulness_ft"],
    mode='markers',
    name='Fine-Tuned Model (fraud-model-v4:latest)',
    marker=dict(size=12, color='#d62728', symbol='x', opacity=0.8, line=dict(width=1, color='White')),
    text=[f"Q: {q[:60]}..." for q in df["question"]],
    hovertemplate="<b>Fine-Tuned Performance</b><br>Relevancy: %{x:.4f}<br>Faithfulness: %{y:.4f}<br>%{text}<extra></extra>"
))

# Calculate and draw Mean Performance crosshairs
base_f_mean = df["faithfulness_base"].mean()
base_r_mean = df["answer_relevancy_base"].mean()
ft_f_mean = df["faithfulness_ft"].mean()
ft_r_mean = df["answer_relevancy_ft"].mean()

# Add lines representing structural shifting benchmarks
fig_scatter.add_hline(y=base_f_mean, line_dash="dash", line_color="#1f77b4", opacity=0.5, annotation_text="Base Faith Mean")
fig_scatter.add_vline(x=base_r_mean, line_dash="dash", line_color="#1f77b4", opacity=0.5, annotation_text="Base Rel Mean")
fig_scatter.add_hline(y=ft_f_mean, line_dash="dash", line_color="#d62728", opacity=0.5, annotation_text="FT Faith Mean")
fig_scatter.add_vline(x=ft_r_mean, line_dash="dash", line_color="#d62728", opacity=0.5, annotation_text="FT Rel Mean")

fig_scatter.update_layout(
    title="Quality Comparison: Base vs. Fine-Tuned (Hover to Inspect Questions)",
    xaxis_title="Answer Relevancy Score",
    yaxis_title="Faithfulness Score",
    width=1200,   # Large presentation canvas width
    height=800,   # Large presentation canvas height
    template="plotly_white",
    xaxis=dict(range=[-0.05, 1.05]),
    yaxis=dict(range=[-0.05, 1.05]),
    legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
)

fig_scatter.write_html("faithfulness_relevancy_scatter.html")
print("Saved: faithfulness_relevancy_scatter.html")

# ----------------------------------------------------------------------
# 4. Plotly Chart 2: Heatmap of Performance Deltas (First 25 Rows)
# ----------------------------------------------------------------------
print("Generating Delta Optimization Heatmap...")
df["faithfulness_delta"] = df["faithfulness_ft"] - df["faithfulness_base"]
df["relevancy_delta"] = df["answer_relevancy_ft"] - df["answer_relevancy_base"]

delta_subset = df[["faithfulness_delta", "relevancy_delta"]].iloc[:25].copy()
y_labels = [f"Log Query {i+1}" for i in range(25)]

fig_heatmap = px.imshow(
    delta_subset,
    labels=dict(x="Metrics Layer", y="Transaction Index", color="Net Delta Value"),
    x=["Faithfulness Delta", "Relevancy Delta"],
    y=y_labels,
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    aspect="auto"
)

fig_heatmap.update_layout(
    title="Sample Slice Optimization Analysis (First 25 Evaluation Points)",
    width=1000,
    height=900,
    template="plotly_white"
)
fig_heatmap.update_traces(text=np.round(delta_subset.values, 3), showlegend=False, hoverongaps=False, texttemplate="%{text}")

fig_heatmap.write_html("per_question_deltas_heatmap.html")
print("Saved: per_question_deltas_heatmap.html")

# ----------------------------------------------------------------------
# 5. Plotly Chart 3: Confusion Matrix for Risk Classification
# ----------------------------------------------------------------------
if not valid_ft_df.empty:
    print("Generating Confusion Matrix...")
    labels_order = ["Low Risk", "High Risk"]
    cm = confusion_matrix(valid_ft_df["ground_truth_label"], valid_ft_df["predicted_label_ft"], labels=labels_order)
    
    fig_cm = px.imshow(
        cm,
        x=labels_order,
        y=labels_order,
        labels=dict(x="Predicted Risk Level", y="True Risk Level", color="Case Count"),
        color_continuous_scale="Reds"
    )
    
    fig_cm.update_layout(
        title="Confusion Matrix: Fine-Tuned Adapter Decisions",
        width=700,
        height=650,
        template="plotly_white"
    )
    fig_cm.update_traces(text=cm, showlegend=False, texttemplate="%{text}", textfont=dict(size=16))
    
    fig_cm.write_html("confusion_matrix.html")
    print("Saved: confusion_matrix.html")
else:
    print("⚠ Warning: Skipping Confusion Matrix drawing due to unmatched formatting strings.")

# ----------------------------------------------------------------------
# 6. Technical Writeup Generation
# ----------------------------------------------------------------------
total_samples = len(df)
high_risk_count = sum(1 for label in df["ground_truth_label"] if label == "High Risk")
low_risk_count = sum(1 for label in df["ground_truth_label"] if label == "Low Risk")

methodology_content = f"""# LLM Evaluation Methodology & Verification Report

**Generated Evaluation Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Experimental Dataset Context
- **Sample Population ($N$):** {total_samples} transaction sequences evaluated serially.
- **Ground Truth Target Breakdown:** High Risk: {high_risk_count} records | Low Risk: {low_risk_count} records.

## 📈 Aggregated Statistical Performance Indicators

| Model Variant Profile | Average Faithfulness (Grounded Logic) | Average Answer Relevancy |
| :--- | :---: | :---: |
| **Base Baseline (`llama3.2:latest`)** | {base_f_mean:.4f} | {base_r_mean:.4f} |
| **Fine-Tuned Candidate (`fraud-model-v4:latest`)** | {ft_f_mean:.4f} | {ft_r_mean:.4f} |
| **Net Structural System Delta ($\Delta$)** | **+{ft_f_mean - base_f_mean:+.4f}** | **+{ft_r_mean - base_r_mean:+.4f}** |
"""

with open("methodology.md", "w", encoding="utf-8") as f:
    f.write(methodology_content)
print("Saved report: methodology.md")

print("\n✅ All interactive Plotly visualizations and markdown reports built successfully.")