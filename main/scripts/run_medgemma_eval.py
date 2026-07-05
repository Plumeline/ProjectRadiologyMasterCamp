"""
run_medgemma_eval.py
--------------------
Fait tourner le vrai modèle MedGemma-4B-it sur toutes les images de
data/sample_images/ et sauvegarde les vraies réponses dans un fichier CSV.

Usage :
    .venv/bin/python scripts/run_medgemma_eval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from PIL import Image

# S'assurer que le dossier racine est dans le path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.preprocessing import basic_quality_flag

MODEL_ID    = "google/medgemma-4b-it"
SAMPLE_DIR  = ROOT / "data" / "sample_images"
OUTPUT_CSV  = ROOT / "data" / "medgemma_results.csv"

# Prompts selon le mode
PROMPTS = {
    "baseline": (
        "Analyze this chest X-ray. Tell me if there is suspected opacity. "
        "Provide: class (normal, suspected_opacity, or uncertain), confidence, and visual observations."
    ),
    "improved": (
        "You are an expert radiologist. Carefully examine this chest X-ray for consolidation, infiltrate, or effusion. "
        "Provide a structured analysis containing: "
        "1. Predicted class (normal, suspected_opacity, or uncertain) "
        "2. Confidence score (between 0.0 and 1.0) "
        "3. Specific visual evidence. "
        "If the image quality is poor or findings are ambiguous, flag it as uncertain."
    )
}

def extract_class_from_text(text: str) -> str:
    """Analyse succincte du texte de MedGemma pour en extraire la classe."""
    text_lower = text.lower()
    if "suspected_opacity" in text_lower or "opacity" in text_lower or "consolidation" in text_lower:
        return "suspected_opacity"
    elif "normal" in text_lower or "clear" in text_lower:
        return "normal"
    return "uncertain"

def main():
    # Détection hardware
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("💻 Accélération Metal GPU (MPS) détectée.")
    else:
        device = "cpu"
        dtype = torch.float32
        print("💻 Exécution sur CPU.")

    print(f"🔄 Chargement de MedGemma ({MODEL_ID})...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        ).to(device)
        print("✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"❌ Impossible de charger MedGemma (avez-vous fait login ?) : {e}")
        return

    images = sorted(SAMPLE_DIR.glob("*.png")) + sorted(SAMPLE_DIR.glob("*.jpg"))
    if not images:
        print(f"❌ Aucune image trouvée dans {SAMPLE_DIR}")
        return

    print(f"🚀 Début de l'analyse sur {len(images)} images (2 modes par image = {len(images)*2} runs)...")
    results = []

    for i, img_path in enumerate(images, 1):
        print(f"\n🖼️ [{i}/{len(images)}] Analyse de {img_path.name}...")
        quality = basic_quality_flag(img_path)
        
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"❌ Impossible de lire l'image {img_path.name} : {e}")
            continue

        for mode in ["baseline", "improved"]:
            start_time = time.time()
            prompt = PROMPTS[mode]
            
            try:
                # Génération VLM
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_new_tokens=200)
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                latency_ms = int((time.time() - start_time) * 1000)
                
                predicted_class = extract_class_from_text(generated_text)
                
                # Conserver les structures attendues
                results.append({
                    "filename": img_path.name,
                    "mode": mode,
                    "image_quality": quality,
                    "predicted_class": predicted_class,
                    "confidence": 0.85 if predicted_class != "uncertain" else 0.45, # MedGemma-it n'extrait pas toujours sa confiance proprement en valeur numérique
                    "medgemma_raw_output": generated_text,
                    "latency_ms": latency_ms,
                })
                print(f"  └─ Mode {mode:8s} -> {predicted_class.upper()} ({latency_ms} ms)")
                
            except Exception as e:
                print(f"  ❌ Erreur sur {img_path.name} [{mode}] : {e}")

    # Sauvegarde
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Analyse terminée ! Les résultats ont été enregistrés dans : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
