from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Any

# Configuration issue du cahier des charges
ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_KEYS = {"image_quality", "predicted_class", "confidence", "visual_evidence", "justification", "limitations", "warning"}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."


def check_image_quality(image_path: str) -> dict[str, Any]:
    """
    Vérifie la qualité technique de la radiographie (CXR) avant soumission au VLM.
    Filtre de sécurité amont pour éviter le traitement d'images inexploitables.
    """
    img = Image.open(image_path).convert("L")  # niveaux de gris
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

    # 3. Résolution minimale (Réactivée)
    w, h = img.size
    if w < 224 or h < 224:
        issues.append(f"résolution insuffisante ({w}x{h})")

    # 4. Ratio d'aspect : une CXR frontale est ~carrée ou légèrement portrait
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

    # Mapping avec les termes attendus par la fonction de validation en aval
    if len(issues) >= 2:
        quality = "poor"
    elif issues:
        quality = "limited"
    else:
        quality = "good"

    return {"quality": quality, "issues": issues}


def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Valide la conformité structurelle et clinique du JSON produit par le VLM.
    """
    errors: list[str] = []
    
    # Vérification des clés requises
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
        
    # Vérification de la classe
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
        
    # Vérification du niveau de confiance
    try:
        conf = float(pred.get("confidence", -1))
        if not (0 <= conf <= 1):
            errors.append("confidence outside [0,1]")
    except (TypeError, ValueError):
        errors.append("confidence is not numeric")
        
    # Vérification du message d'avertissement
    if not pred.get("warning"):
        errors.append("warning missing")
        
    return not errors, errors


def apply_safety_guardrails(pred: dict[str, Any]) -> dict[str, Any]:
    """
    Applique le filet de sécurité (post-processing). Force la prudence si le 
    schéma est invalide ou si la qualité de l'image est dégradée.
    """
    # 1. Validation de la prédiction brute
    valid, errors = validate_prediction(pred)
    
    if not valid:
        # Repli sécuritaire en cas de sortie VLM corrompue
        pred["predicted_class"] = "uncertain"
        try:
            current_conf = float(pred.get("confidence", 0.0))
            pred["confidence"] = min(current_conf, 0.5)
        except (TypeError, ValueError):
            pred["confidence"] = 0.0
            
        # Injection propre de l'erreur dans les limitations
        current_limits = pred.get("limitations")
        if isinstance(current_limits, list):
            current_limits.append("guardrail triggered: invalid output schema")
        elif isinstance(current_limits, str):
            pred["limitations"] = [current_limits, "guardrail triggered: invalid output schema"]
        else:
            pred["limitations"] = ["guardrail triggered: invalid output schema"]

    # 2. Règle d'incertitude liée à la qualité d'image
    # Si la qualité est basse (limited/poor) et le modèle peu sûr (< 60%), on force 'uncertain'
    try:
        conf_value = float(pred.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf_value = 0.0

    if pred.get("image_quality") in {"limited", "poor"} and conf_value < 0.6:
        pred["predicted_class"] = "uncertain"

    # 3. Forçage réglementaire des mentions obligatoires
    pred["warning"] = WARNING_TEXT
    pred["guardrail_errors"] = errors
    
    return pred


# À completer selon les tests
CXR_LEXICON = {"poumon", "poumons", "thorax", "thoracique", "plèvre", "cœur", "diaphragme", "médiastin", "lung", "chest", "opacity", "infiltrat"}

def check_hallucination_lexicon(pred: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Vérifie si le texte généré contient au moins un mot-clé du domaine thoracique.
    Évite qu'un VLM extrapole sur une image totalement hors-sujet.
    """
    text_to_check = f"{pred.get('visual_evidence', '')} {pred.get('justification', '')}".lower()
    
    # Si le modèle est sûr de lui mais n'utilise aucun terme technique lié au thorax
    if pred.get("predicted_class") in {"normal", "suspected_opacity"}:
        if not any(word in text_to_check for word in CXR_LEXICON):
            return False, "Le modèle semble hors-sujet ou l'image n'est pas une radiographie thoracique."
            
    return True, None

CRITICAL_KEYWORDS = {"grave", "foudroyant", "masse", "pneumothorax", "épanchement", "détresse"}

def check_semantic_coherence(pred: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Empêche le modèle de classer en 'normal' si le texte explicatif 
    contient des indices de pathologie lourde.
    """
    justification = pred.get("justification", "").lower()
    predicted_class = pred.get("predicted_class")
    
    if predicted_class == "normal" and any(word in justification for word in CRITICAL_KEYWORDS):
        return False, "Incohérence sémantique détectée (mots critiques trouvés pour une classe 'normal')."
        
    return True, None

def apply_uncertainty_threshold(pred: dict[str, Any], lower_bound: float = 0.4, upper_bound: float = 0.65) -> dict[str, Any]:
    """
    Si la confiance est dans la zone grise, force la classe 'uncertain' 
    conformément aux consignes du cahier des charges.
    """
    try:
        conf = float(pred.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
        
    if lower_bound <= conf <= upper_bound:
        if pred.get("predicted_class") != "uncertain":
            pred["predicted_class"] = "uncertain"
            pred.setdefault("limitations", []).append(f"Classe modifiée en 'uncertain' : confiance dans la zone grise ({conf:.2f}).")
            
    return pred
