from __future__ import annotations

from typing import Any

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_KEYS = {"image_quality", "predicted_class", "confidence", "visual_evidence", "justification", "limitations", "warning"}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

import numpy as np
from PIL import Image



def check_image_quality(image_path):
    """
    image_path : chemin de l'image à vérifier

    La fonctions sert à vérifier manuellement si l'image est de mauvaise qualité AVANT de la donner
    comme input au VLM. La fonction détecte les cas les plus évidents de mauvaise qualité, et permet
    d'aider la VLM en rajoutant un filet de sécurité.
    Elle permet aussi d'éviter de passer beaucoup de temps à prédire pour une image qui est de manière
    évidente de très mauvaise qualité.
    """

    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)

    issues = []

    # 1. Exposition : luminosité moyenne
    mean_brightness = arr.mean()
    if mean_brightness < 30:
        issues.append("(trop sombre)")
    elif mean_brightness > 220:
        issues.append("(trop claire)")

    # 2. Contraste : écart-type des pixels
    std = arr.std()
    if std < 20:
        issues.append("contraste trop faible")

    # 3. Résolution minimale
    w, h = img.size
    """
    if w < 224 or h < 224:
        issues.append(f"résolution insuffisante ({w}x{h})")"""

    # 4. Ratio d'aspect : une CXR frontale est ~carré ou légèrement portrait
    #le ratio doit idéalement être inchangé (ou peu varier) pour éviter les incohérences
    ratio = w / h
    if ratio < 0.7 or ratio > 1.4:
        issues.append(f"ratio d'aspect inhabituel ({ratio:.2f})")

    # 5. Saturation des pixels (pixels brûlés ou noirs)
    pct_black = (arr < 5).mean()
    pct_white = (arr > 250).mean()
    if pct_black > 0.5:
        issues.append(f"trop de pixels noirs ({pct_black:.0%})")
    if pct_white > 0.3:
        issues.append(f"trop de pixels saturés ({pct_white:.0%})")

    quality = "mauvaise" if len(issues) >= 2 else "moyenne" if issues else "bonne"
    return {"quality": quality, "issues": issues}



def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
    try:
        conf = float(pred.get("confidence", -1))
        if not 0 <= conf <= 1:
            errors.append("confidence outside [0,1]")
    except Exception:
        errors.append("confidence is not numeric")
    if not pred.get("warning"):
        errors.append("warning missing")
    return not errors, errors


def apply_safety_guardrails(pred: dict[str, Any]) -> dict[str, Any]:
    valid, errors = validate_prediction(pred)
    if not valid:
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(float(pred.get("confidence", 0.0) or 0.0), 0.5)
        pred.setdefault("limitations", []).append("guardrail triggered: invalid output schema")
    if pred.get("image_quality") in {"limited", "poor"} and float(pred.get("confidence", 0)) < 0.6:
        pred["predicted_class"] = "uncertain"
    pred["warning"] = WARNING_TEXT
    pred["guardrail_errors"] = errors
    return pred
