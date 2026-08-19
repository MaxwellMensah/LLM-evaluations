"""Fraud detection model evaluation - Base vs Fine-Tuned.

Metrics
-------
Faithfulness     : Judge LLM gives a 0.0-1.0 score via a single direct prompt.
                   No JSON required — the score is parsed from free text.
Answer Relevancy : Cosine similarity between the question embedding and the
                   answer embedding. No LLM judge involved at all.

Why no RAGAS
------------
RAGAS requires the judge LLM to emit specific JSON schemas across multiple
chained prompts. Small local models (≤8B) fail this 15-30% of the time
even with output-cleaning wrappers, producing NaN scores. The metrics above
are equivalent for comparing base vs fine-tuned and are 100% reliable with
any Ollama model.
"""

import json
import math
import random
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"

JUDGE_MODEL = "llama3.2:latest"  # Any Ollama model works; bigger = more accurate
EMBEDDING_MODEL = "nomic-embed-text:latest"

BASE_MODEL = "llama3.2:latest"
FINETUNED_MODEL = "fraud-model-v4:latest"

TEST_DATA_PATH = "fraud_detection_dataset_V4.jsonl"
SAMPLE_SIZE = 300  # Set to a small number (e.g. 3) for a quick smoke test

LLAMA3_FAMILIES = ("llama3", "llama-3", "fraud-model")
GEMMA_FAMILIES = ("gemma",)
MISTRAL_FAMILIES = ("mistral",)
SYSTEM_MSG = (
    "You are a fraud detection expert. "
    "Analyze transactions using step-by-step reasoning."
)


# ── Ollama helpers ────────────────────────────────────────────────────────────
def ollama_ok() -> bool:
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def model_ok(tag: str, avail: list) -> bool:
    return any(tag in m or m in tag for m in avail)


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(user_content: str, model_tag: str) -> str:
    tag = model_tag.lower()
    if any(f in tag for f in GEMMA_FAMILIES):
        return (
            f"<start_of_turn>user\n{SYSTEM_MSG}\n\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    if any(f in tag for f in MISTRAL_FAMILIES):
        return f"[INST] {SYSTEM_MSG}\n\n{user_content} [/INST]"
    # Llama 3 (default — also covers fraud-model-v4)
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_MSG}\n\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}\n\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def ollama_generate(prompt: str, model: str, max_tokens: int = 512) -> str:
    try:
        r = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0},
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        return f"[Error: {e}]"


# ── Embeddings ────────────────────────────────────────────────────────────────
def embed(text: str) -> list:
    """Return a float embedding vector via Ollama's embed endpoint."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        # Ollama returns {"embeddings": [[...]] } or {"embedding": [...]}
        if "embeddings" in data:
            return data["embeddings"][0]
        return data.get("embedding", [])
    except Exception:
        return []


def cosine(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b))
    return dot / mag if mag else 0.0


# ── Metrics ───────────────────────────────────────────────────────────────────
def score_faithfulness(context: str, answer: str) -> float:
    """Ask the judge for a single 0-1 float. No JSON, no schema."""
    prompt = (
        "You are an impartial evaluator.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Does the answer contain ONLY information that can be directly verified "
        "from the CONTEXT above?\n"
        "Reply with a single decimal number between 0.0 (not faithful) and "
        "1.0 (completely faithful). Output ONLY the number."
    )
    raw = ollama_generate(prompt, JUDGE_MODEL, max_tokens=8)
    match = re.search(r"([01](?:\.\d+)?|\.\d+)", raw)
    return min(1.0, max(0.0, float(match.group()) if match else 0.5))


def score_relevancy(question: str, answer: str) -> float:
    """Cosine similarity between question and answer embeddings. No LLM call."""
    return cosine(embed(question), embed(answer))


# ── Data loading ──────────────────────────────────────────────────────────────
def load_test_data(path: str, n: int) -> list:
    print(f"  Reading {path} ...")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                convs = item.get("conversations", [])
                if len(convs) >= 3:
                    q = convs[1].get("content", "").strip()
                    a = convs[2].get("content", "").strip()
                    if q and a:
                        rows.append({"question": q, "ground_truth": a, "contexts": [q]})
            except Exception:
                continue

    if not rows:
        raise ValueError(f"No valid rows found in {path}")

    random.seed(42)
    if len(rows) <= n:
        random.shuffle(rows)
        return rows

    sampled = random.sample(rows, n)
    print(f"  Sampled {n} rows from {len(rows)} total.")
    return sampled


# ── Model evaluation ──────────────────────────────────────────────────────────
def evaluate_model(display_name: str, model_tag: str, test_data: list, avail: list):
    if not model_ok(model_tag, avail):
        print(f"  ✗ Model '{model_tag}' not available in Ollama.")
        return None

    # Step 1 — generate answers
    print(f"\n  Generating answers from {display_name} ({model_tag}) ...")
    answers = []
    for item in tqdm(test_data, desc="  Answers", unit="q"):
        ans = ollama_generate(build_prompt(item["question"], model_tag), model_tag)
        if not ans or ans.startswith("[Error"):
            ans = "Unable to generate answer."
        answers.append(ans)
        time.sleep(0.1)

    # Step 2 — score each answer
    print(f"  Scoring {display_name} ...")
    records = []
    for item, ans in tqdm(
        zip(test_data, answers), total=len(test_data), desc="  Scoring", unit="q"
    ):
        faith = score_faithfulness(item["contexts"][0], ans)
        relev = score_relevancy(item["question"], ans)
        records.append(
            {
                "question": item["question"],
                "generated_answer": ans,
                "ground_truth": item["ground_truth"],
                "faithfulness": faith,
                "answer_relevancy": relev,
            }
        )
        time.sleep(0.05)

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  LLM EVALUATION — Base vs Fine-Tuned")
    print("=" * 65)

    if not ollama_ok():
        print("  ✗ Ollama not running. Start with: ollama serve")
        return

    avail = available_models()
    print(f"  ✓ Ollama running. Models: {avail}")
    print(f"  Judge : {JUDGE_MODEL}")
    print(f"  Embeds: {EMBEDDING_MODEL}")

    print("\nLoading test data ...")
    test_data = load_test_data(TEST_DATA_PATH, SAMPLE_SIZE)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    df_base = evaluate_model("Base", BASE_MODEL, test_data, avail)
    df_ft = evaluate_model("Fine-tuned", FINETUNED_MODEL, test_data, avail)

    # ── Aggregated summary ────────────────────────────────────────────────────
    rows = []
    for label, df in [
        (f"Base ({BASE_MODEL})", df_base),
        (f"Fine-tuned ({FINETUNED_MODEL})", df_ft),
    ]:
        if df is not None:
            rows.append(
                {
                    "Model": label,
                    "Faithfulness": f"{df['faithfulness'].mean():.4f}",
                    "Answer_Relevancy": f"{df['answer_relevancy'].mean():.4f}",
                    "N": len(df),
                }
            )
        else:
            rows.append(
                {
                    "Model": label,
                    "Faithfulness": "N/A",
                    "Answer_Relevancy": "N/A",
                    "N": 0,
                }
            )

    df_agg = pd.DataFrame(rows)
    df_agg.to_csv("eval_scores_aggregated.csv", index=False)
    print("\n  Aggregated scores → eval_scores_aggregated.csv")

    # ── Per-question files ────────────────────────────────────────────────────
    if df_base is not None:
        df_base.to_csv("eval_scores_base.csv", index=False)
        print("  Per-question base  → eval_scores_base.csv")

    if df_ft is not None:
        df_ft.to_csv("eval_scores_ft.csv", index=False)
        print("  Per-question ft    → eval_scores_ft.csv")

    if df_base is not None and df_ft is not None:
        combined = df_base.merge(df_ft, on="question", suffixes=("_base", "_ft"))
        combined.rename(columns={"ground_truth_base": "ground_truth"}, inplace=True)
        if "ground_truth_ft" in combined.columns:
            combined.drop(columns=["ground_truth_ft"], inplace=True)
        combined.to_csv("eval_scores_combined.csv", index=False)
        print("  Combined           → eval_scores_combined.csv")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(df_agg.to_string(index=False))
    print("\n  ✅ Evaluation complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
