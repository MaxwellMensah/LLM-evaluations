"""
Fraud Detection Model Evaluation
- Models under test : local Ollama (base + fine-tuned)
- Judge             : OpenAI GPT-4o-mini  (reliable JSON, cheap)
- Metrics           : Faithfulness, Answer Relevancy
"""

import json, math, os, random, re, time
import requests, pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-1234...")   # set env var or paste key
JUDGE_MODEL     = "gpt-4o-mini"

OLLAMA_URL      = "http://localhost:11434"
BASE_MODEL      = "llama3.2:latest"
FINETUNED_MODEL = "fraud-model-v4:latest"
EMBED_MODEL     = "nomic-embed-text:latest"

DATASET_PATH    = "fraud_detection_dataset_V4.jsonl"
SAMPLE_SIZE     = 100   # lower to 5 for a quick smoke test

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                c = json.loads(line)["conversations"]
                if len(c) >= 3:
                    q, a = c[1]["content"].strip(), c[2]["content"].strip()
                    if q and a:
                        rows.append({"question": q, "ground_truth": a})
            except Exception:
                continue
    random.seed(42)
    return random.sample(rows, min(n, len(rows)))

# ── Local model inference ─────────────────────────────────────────────────────
SYSTEM = "You are a fraud detection expert. Analyze transactions using step-by-step reasoning."

def generate(question, model):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}\n\n"
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}\n\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "stream": False,
            "options": {"num_predict": 512, "temperature": 0},
        }, timeout=120)
        return r.json()["response"].strip()
    except Exception as e:
        return f"[Error: {e}]"

# ── Scoring ───────────────────────────────────────────────────────────────────
def score_faithfulness(context, answer):
    """GPT-4o-mini rates how faithful the answer is to the context (0.0-1.0)."""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                "Rate how faithfully the ANSWER is supported by the CONTEXT.\n"
                "Return JSON: {\"score\": <float 0.0-1.0>}\n\n"
                f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
            )
        }]
    )
    return json.loads(resp.choices[0].message.content)["score"]

def embed_ollama(text):
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": text}, timeout=30)
    d = r.json()
    return d.get("embeddings", [d.get("embedding", [])])[0]

def cosine(a, b):
    if not a or not b: return 0.0
    dot  = sum(x*y for x,y in zip(a,b))
    norm = math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b))
    return dot/norm if norm else 0.0

def score_relevancy(question, answer):
    """Cosine similarity between question and answer embeddings. No LLM call."""
    return cosine(embed_ollama(question), embed_ollama(answer))

# ── Evaluate one model ────────────────────────────────────────────────────────
def evaluate(name, model, data):
    print(f"\n── {name} ({model}) ──")
    records = []
    for row in tqdm(data, desc="  evaluating"):
        ans   = generate(row["question"], model)
        faith = score_faithfulness(row["question"], ans)
        relev = score_relevancy(row["question"], ans)
        records.append({
            "question":         row["question"],
            "ground_truth":     row["ground_truth"],
            "generated_answer": ans,
            "faithfulness":     faith,
            "answer_relevancy": relev,
        })
        time.sleep(0.1)
    return pd.DataFrame(records)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    data = load_data(DATASET_PATH, SAMPLE_SIZE)
    print(f"  {len(data)} samples loaded.")

    df_base = evaluate("Base",       BASE_MODEL,      data)
    df_ft   = evaluate("Fine-tuned", FINETUNED_MODEL, data)

    # Save
    df_base.to_csv("scores_base.csv", index=False)
    df_ft.to_csv("scores_ft.csv", index=False)

    combined = df_base.merge(df_ft, on="question", suffixes=("_base", "_ft"))
    combined.rename(columns={"ground_truth_base": "ground_truth"}, inplace=True)
    combined.drop(columns=["ground_truth_ft"], errors="ignore", inplace=True)
    combined.to_csv("scores_combined.csv", index=False)

    # Summary
    summary = pd.DataFrame([
        {"Model": f"Base ({BASE_MODEL})",
         "Faithfulness":     f"{df_base['faithfulness'].mean():.4f}",
         "Answer_Relevancy": f"{df_base['answer_relevancy'].mean():.4f}"},
        {"Model": f"Fine-tuned ({FINETUNED_MODEL})",
         "Faithfulness":     f"{df_ft['faithfulness'].mean():.4f}",
         "Answer_Relevancy": f"{df_ft['answer_relevancy'].mean():.4f}"},
    ])
    summary.to_csv("scores_summary.csv", index=False)

    print("\n" + "="*60)
    print(summary.to_string(index=False))
    print("\nFiles: scores_base.csv  scores_ft.csv  scores_combined.csv  scores_summary.csv")

if __name__ == "__main__":
    main()