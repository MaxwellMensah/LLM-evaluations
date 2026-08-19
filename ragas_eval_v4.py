"""
RAGAS evaluation for base vs fine-tuned fraud detection models.
Uses local Ollama with Llama-3.1 chat template (matching training).
Loads exactly 100 randomized evaluation data rows directly from a JSONL file.
"""

import json
import time
import random
import requests
import pandas as pd
from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"

# Judge model (evaluator)
JUDGE_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Models to compare
BASE_MODEL = "llama3.2:latest"
FINETUNED_MODEL = "fraud-model-v4:latest"

# Target dataset file path
TEST_DATA_PATH = "fraud_detection_dataset_V4.jsonl"
QUICK_TEST_MODE = True  # Turned off to allow the full 100 sample scale run
QUICK_TEST_SIZE = 3

# ----------------------------------------------------------------------
# Helper: build prompt using Llama-3.1 chat template (exactly as training)
# ----------------------------------------------------------------------
def build_prompt(user_content: str) -> str:
    system_msg = "You are a fraud detection expert. Analyze transactions using step-by-step reasoning."
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}\n\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}\n\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

# ----------------------------------------------------------------------
# Ollama connectivity and model checks
# ----------------------------------------------------------------------
def check_ollama_running() -> bool:
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except:
        return False

def get_available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except:
        return []

def model_available(tag: str, available: list) -> bool:
    return any(tag in m or m in tag for m in available)

def build_judge_llm(model: str):
    return LangchainLLMWrapper(
        ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=256,
            timeout=480,
        )
    )

def build_embeddings(model: str):
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
    )

# ----------------------------------------------------------------------
# Generate answer using the correct prompt format
# ----------------------------------------------------------------------
def generate_answer(question: str, model_tag: str) -> str:
    """Send a properly formatted prompt to Ollama and return the response."""
    prompt = build_prompt(question)
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_tag,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 512, "temperature": 0}
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        return f"Error: {e}"

# ----------------------------------------------------------------------
# Test data loading from .jsonl format with 100 Random Samples Extraction
# ----------------------------------------------------------------------
def load_test_data(path: str) -> list:
    raw_data = []
    print(f"  Reading dataset lines from {path}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # skip empty buffer lines
                
                try:
                    item = json.loads(line)
                    
                    # Sanitize and strictly enforce RAGAS schema shapes
                    processed_item = {
                        "question": str(item.get("question", "")),
                        "answer": str(item.get("answer", item.get("ground_truth", "")))
                    }
                    
                    # Extract contexts list dynamically
                    contexts_raw = item.get("contexts", [])
                    if isinstance(contexts_raw, list):
                        processed_item["contexts"] = [str(c) for c in contexts_raw]
                    else:
                        processed_item["contexts"] = [str(contexts_raw)] if contexts_raw else [""]
                        
                    raw_data.append(processed_item)
                    
                except json.JSONDecodeError as je:
                    print(f"  ✗ Warning: Skipping malformed JSON on line {line_idx}: {je}")
                    
        if not raw_data:
            raise ValueError(f"Dataset {path} parsed successfully but contains 0 valid evaluation lines.")
            
        # ------------------------------------------------------------------
        # Random Sampling Integration
        # ------------------------------------------------------------------
        # Seed 42 guarantees reproducibility across runs
        random.seed(42)
        target_sample_size = 100
        
        if len(raw_data) <= target_sample_size:
            print(f"  ⚠️ Warning: Dataset line size ({len(raw_data)}) is <= requested size ({target_sample_size}). Returning all.")
            random.shuffle(raw_data)
            return raw_data
        else:
            print(f"  🎲 Successfully sampled {target_sample_size} random rows out of {len(raw_data)} entries.")
            return random.sample(raw_data, target_sample_size)

    except FileNotFoundError:
        print(f"\n  ✗ Error: File critical baseline '{path}' was not found.")
        print("  Ensure 'fraud_detection_dataset_V4.jsonl' sits in the working execution folder.")
        raise

# ----------------------------------------------------------------------
# Evaluate a single model
# ----------------------------------------------------------------------
def evaluate_model(display_name: str, model_tag: str, test_data: list, judge_llm, embeddings, available_models: list):
    if not model_available(model_tag, available_models):
        print(f"  ✗ Model '{model_tag}' not available.")
        return {}

    print(f"\n  Generating answers from {display_name}...")
    answers = []
    for item in tqdm(test_data, desc="  Generating", unit="q"):
        ans = generate_answer(item["question"], model_tag)
        answers.append(ans)
        time.sleep(0.2)

    data_dict = {
        "question": [d["question"] for d in test_data],
        "answer": answers,
        "contexts": [d["contexts"] for d in test_data],
        "ground_truth": [d["answer"] for d in test_data],
    }
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),
        "ground_truth": Value("string"),
    })
    dataset = Dataset.from_dict(data_dict, features=features)

    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)

    n = len(test_data)
    est_min = (n * 6 * 35) // 60
    print(f"\n  Starting RAGAS evaluation ({n} queries, serial).")
    print(f"  Estimated time: {est_min}–{est_min*2} min\n")

    # Set workers to 1 to carefully feed our local hardware queue without overloading VRAM
    run_cfg = RunConfig(timeout=1200, max_retries=1, max_wait=60, max_workers=1)
    t0 = time.time()
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=embeddings,
            run_config=run_cfg,
        )
        print(f"\n  ✓ Done in {(time.time()-t0)/60:.1f} min")
        return result
    except Exception as e:
        print(f"\n  ✗ Evaluation failed: {e}")
        return {}

def safe_score(result, key):
    if not result:
        return "N/A"
    
    try:
        val = result[key]
        return f"{val:.4f}" if val is not None else "N/A"
    except:
        return "N/A"

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("\n" + "="*65)
    print("  RAGAS EVALUATION – Base vs Fine‑Tuned (Llama-3.1 chat template)")
    print("="*65)

    if not check_ollama_running():
        print("  ✗ Ollama not running. Start: ollama serve")
        return

    available = get_available_models()
    print(f"  ✓ Ollama OK. Models: {available}")

    # Judge LLM Setup
    judge_model = JUDGE_MODEL
    if not model_available(judge_model, available):
        judge_model = "llama3.2:latest"
    print(f"\nBuilding judge LLM ({judge_model})...")
    judge_llm = build_judge_llm(judge_model)

    # Embeddings Setup
    embed_model = EMBEDDING_MODEL
    if not model_available(embed_model, available):
        embed_model = "llama3.2:latest"
    print(f"\nBuilding embeddings ({embed_model})...")
    embeddings = build_embeddings(embed_model)

    # Test data loading via randomized JSONL loader
    print(f"\nLoading test data from JSONL dataset...")
    test_data = load_test_data(TEST_DATA_PATH)
    
    if QUICK_TEST_MODE:
        test_data = test_data[:QUICK_TEST_SIZE]
        print(f"  ⚡ Quick mode override: restricted to first {QUICK_TEST_SIZE} queries.")

    # Evaluate base model
    print(f"\nEvaluating BASE ({BASE_MODEL})")
    result_base = evaluate_model("Base", BASE_MODEL, test_data, judge_llm, embeddings, available)

    # Evaluate fine‑tuned model
    print(f"\nEvaluating FINE‑TUNED ({FINETUNED_MODEL})")
    result_ft = evaluate_model("Fine-tuned", FINETUNED_MODEL, test_data, judge_llm, embeddings, available)

    # Results compilation
    df = pd.DataFrame({
        "Model": [f"Base ({BASE_MODEL})", f"Fine-tuned ({FINETUNED_MODEL})"],
        "Faithfulness": [safe_score(result_base, "faithfulness"), safe_score(result_ft, "faithfulness")],
        "Answer_Relevancy": [safe_score(result_base, "answer_relevancy"), safe_score(result_ft, "answer_relevancy")]
    })
    
    df.to_csv("ragas_scores.csv", index=False)
    print("\n" + "="*65)
    print(df.to_string(index=False))
    print("\n  → ragas_scores.csv saved")
    print("="*65)

if __name__ == "__main__":
    main()