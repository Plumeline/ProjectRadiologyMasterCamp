"""
Phase 2 - Offline batch inference with MedGemma 4B (run ONCE on Colab, GPU).

Loads MedGemma 4B in 4-bit, runs every synthetic case through a given prompt,
parses the model's JSON, and writes one results file the rest of the repo reads.
The web app and evaluation never load the model: they consume these cached files.

Usage (in a Colab cell, after auth + uploading the repo):
    !python eval/run_inference_batch.py --prompt baseline
    !python eval/run_inference_batch.py --prompt improved

Outputs:
    eval/cached_predictions/predictions_baseline.json
    eval/cached_predictions/predictions_improved.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "google/medgemma-4b-it"
REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_CSV = REPO_ROOT / "data" / "synthetic_cases.csv"
PROMPT_FILES = {
    "baseline": REPO_ROOT / "prompts" / "baseline_prompt.txt",
    "improved": REPO_ROOT / "prompts" / "improved_prompt.txt",
}
OUT_DIR = REPO_ROOT / "eval" / "cached_predictions"

VALID_CLASSES = {"normal", "suspected_opacity", "uncertain"}
WARNING_TEXT = (
    "Educational prototype only. Not for diagnosis. "
    "A qualified clinician must verify the image."
)


def load_model():
    """Load MedGemma 4B in 4-bit. Fits a T4 (16GB) or a 10GB local card."""
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def extract_json(text):
    """Pull the first JSON object out of the model's free text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def fallback_uncertain(reason):
    """Schema-valid 'uncertain' result when the model fails to return JSON."""
    return {
        "image_quality": "poor",
        "predicted_class": "uncertain",
        "confidence": 0.0,
        "visual_evidence": [],
        "justification": f"Model output could not be parsed as valid JSON ({reason}).",
        "limitations": ["unparseable model output"],
        "warning": WARNING_TEXT,
    }


def normalize(parsed):
    """Make sure the parsed dict is schema-complete; coerce obvious issues."""
    if parsed is None:
        return fallback_uncertain("no JSON found"), False
    out = dict(parsed)
    out.setdefault("image_quality", "limited")
    out.setdefault("predicted_class", "uncertain")
    out.setdefault("confidence", 0.0)
    out.setdefault("visual_evidence", [])
    out.setdefault("justification", "")
    out.setdefault("limitations", [])
    out["warning"] = WARNING_TEXT  # always enforce the mandatory warning

    if out["predicted_class"] not in VALID_CLASSES:
        out["predicted_class"] = "uncertain"
    try:
        out["confidence"] = float(out["confidence"])
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    return out, True


@torch.no_grad()
def run_one(model, processor, prompt_text, image_path):
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Analyze this frontal chest X-ray. Return only the JSON."},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    generated = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,  # deterministic -> reproducible baseline
    )
    decoded = processor.decode(generated[0][input_len:], skip_special_tokens=True)
    return decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["baseline", "improved"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="debug: only N cases")
    args = ap.parse_args()

    prompt_text = PROMPT_FILES[args.prompt].read_text(encoding="utf-8")
    cases = pd.read_csv(CASES_CSV)
    if args.limit:
        cases = cases.head(args.limit)

    print(f"Loading {MODEL_ID} in 4-bit...")
    model, processor = load_model()
    print("Model loaded. Running inference.")

    results = []
    for i, row in cases.iterrows():
        img_path = REPO_ROOT / row["image_path"]
        t0 = time.time()
        try:
            raw = run_one(model, processor, prompt_text, img_path)
            parsed = extract_json(raw)
            result, json_ok = normalize(parsed)
        except Exception as e:  # never let one image kill the whole run
            raw = f"ERROR: {e}"
            result, json_ok = fallback_uncertain(str(e)), False

        results.append({
            "case_id": row["case_id"],
            "image_path": row["image_path"],
            "ground_truth": row["label"],
            "split": row["split"],
            "prompt_mode": args.prompt,
            "model_id": MODEL_ID,
            "json_valid": json_ok,
            "raw_output": raw,
            "prediction": result,
        })
        dt = time.time() - t0
        print(f"[{i+1:>2}/{len(cases)}] {row['case_id']} "
              f"gt={row['label']:<17} pred={result['predicted_class']:<17} "
              f"conf={result['confidence']:.2f} json_ok={json_ok} ({dt:.1f}s)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"predictions_{args.prompt}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    n_valid = sum(r["json_valid"] for r in results)
    n_correct = sum(r["prediction"]["predicted_class"] == r["ground_truth"] for r in results)
    print(f"\nWrote {out_path}")
    print(f"JSON valid: {n_valid}/{len(results)}   "
          f"Accuracy vs labels: {n_correct}/{len(results)} = {n_correct/len(results):.2%}")


if __name__ == "__main__":
    main()
