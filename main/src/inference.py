from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Directory holding the offline MedGemma batch outputs (Phase 2).
_CACHE_DIR = Path(__file__).resolve().parents[1] / "eval" / "cached_predictions"
# Lazily loaded: {mode: {image_filename: prediction_dict}}
_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def _load_cache(mode: str) -> dict[str, dict[str, Any]]:
    """Load and index the cached MedGemma outputs for a given prompt mode.

    Returns a map from image filename -> normalized prediction dict.
    Cached in module memory so the file is read once per process.
    """
    if mode in _CACHE:
        return _CACHE[mode]

    cache_file = _CACHE_DIR / f"predictions_{mode}.json"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cached predictions for mode '{mode}' at {cache_file}. "
            f"Run eval/run_inference_batch.py --prompt {mode} on GPU first."
        )

    records = json.loads(cache_file.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = Path(rec["image_path"]).name  # match on filename only
        index[key] = rec
    _CACHE[mode] = index
    return index


def _lookup_key(image_path: str | Path, index: dict[str, dict[str, Any]]) -> str | None:
    """Resolve an image path to a cache key, tolerant of API upload renaming.

    The API saves uploads as 'uploaded_<originalstem><suffix>'. We try the exact
    filename first, then strip a leading 'uploaded_' prefix, then match on stem.
    """
    name = Path(image_path).name
    if name in index:
        return name
    # strip the API's 'uploaded_' prefix if present
    if name.startswith("uploaded_"):
        stripped = name[len("uploaded_"):]
        if stripped in index:
            return stripped
    # last resort: match on stem (filename without extension)
    stem = Path(name).stem
    if stem.startswith("uploaded_"):
        stem = stem[len("uploaded_"):]
    for key in index:
        if Path(key).stem == stem:
            return key
    return None


def cached_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Return the pre-computed MedGemma prediction for this image and mode.

    The heavy VLM inference was run once offline (Phase 2). Here we just look
    up the result and shape it to the same schema as toy_predict, so the API,
    the app and the evaluation are all CPU-only.
    """
    start = time.perf_counter()
    index = _load_cache(mode)
    key = _lookup_key(image_path, index)

    if key is None:
        # Image has no cached result: fall back to a safe 'uncertain'.
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "image_quality": "poor",
            "predicted_class": "uncertain",
            "confidence": 0.0,
            "visual_evidence": [],
            "justification": f"No cached MedGemma prediction found for {Path(image_path).name}.",
            "limitations": ["missing cached prediction"],
            "warning": WARNING,
            "model_name": f"medgemma-4b-it-{mode}",
            "prompt_version": f"{mode}_v1",
            "latency_ms": latency_ms,
        }

    record = index[key]
    pred = dict(record["prediction"])  # the JSON block MedGemma produced

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": pred.get("image_quality", "limited"),
        "predicted_class": pred.get("predicted_class", "uncertain"),
        "confidence": round(float(pred.get("confidence", 0.0)), 3),
        "visual_evidence": pred.get("visual_evidence", []),
        "justification": pred.get("justification", ""),
        "limitations": pred.get("limitations", []),
        "warning": WARNING,  # always enforce the local warning text
        "model_name": record.get("model_id", "google/medgemma-4b-it"),
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Unified entry point.

    mode='toy'              -> deterministic filename-based toy predictor
    mode='baseline'/'improved' -> cached MedGemma outputs, else toy fallback
    """
    if mode == "toy":
        return toy_predict(image_path, mode="baseline")
    try:
        return cached_predict(image_path, mode=mode)
    except FileNotFoundError:
        # No cache available (e.g. fresh checkout before Phase 2): stay runnable.
        return toy_predict(image_path, mode=mode)
