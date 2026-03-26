"""
build_soil_prompt.py
Build a context-engineered LLM prompt from your existing field data.
Pair images with Word reports in training_data/ folder.

Naming convention:
  sample_01.jpg  +  sample_01.docx  (same stem name)

Usage:
  python build_soil_prompt.py
"""

import os
import json
import sys
from datetime import datetime

import cv2
import numpy as np
from docx import Document

# ── Import our CV module ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_analysis import extract_cv_features_text


# ========================= CONFIG =========================
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
OUTPUT_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "as1726_complete_prompt.txt")
MAX_EXAMPLES = 8  # keep under LLM token limits


def parse_word_report(docx_path: str) -> str:
    """Extract soil-relevant paragraphs from a Word report."""
    try:
        doc = Document(docx_path)
        full_text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

        # Try to find soil-relevant lines
        lines = full_text.split("\n")
        soil_keywords = [
            "colour", "color", "moisture", "consistency", "sand", "clay",
            "silt", "gravel", "firm", "stiff", "soft", "dense", "moist",
            "dry", "wet", "brown", "grey", "red", "yellow", "fill",
            "topsoil", "bedrock", "residual", "alluvial"
        ]
        relevant_lines = []
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in soil_keywords):
                # Grab context: this line + up to 4 following lines
                context = lines[i:i + 5]
                relevant_lines.extend(context)
                break

        if relevant_lines:
            return "\n".join(relevant_lines)
        # Fallback: return first 800 chars
        return full_text[:800]
    except Exception as e:
        return f"ERROR reading Word file: {e}"


def build_prompt():
    """Main prompt builder."""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created {DATA_FOLDER}/ — add your image+docx pairs and re-run.")
        return

    # Find paired files
    image_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    examples = []
    for img_file in sorted(image_files)[:MAX_EXAMPLES]:
        base = os.path.splitext(img_file)[0]
        docx_file = f"{base}.docx"
        img_path = os.path.join(DATA_FOLDER, img_file)
        docx_path = os.path.join(DATA_FOLDER, docx_file)

        if not os.path.exists(docx_path):
            print(f"  SKIP {img_file} — no matching {docx_file}")
            continue

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"  SKIP {img_file} — cannot read image")
            continue

        cv_features = extract_cv_features_text(img_cv)
        ground_truth = parse_word_report(docx_path)

        example = (
            f"Example {len(examples) + 1}:\n"
            f"Image file: {img_file}\n"
            f"CV features: {cv_features}\n"
            f"Ground-truth AS1726 log from field report:\n"
            f"{ground_truth}\n"
        )
        examples.append(example)
        print(f"  PAIRED: {img_file} <-> {docx_file}")

    # ── Assemble the complete prompt ──
    system_prompt = f"""You are an expert Australian Geotechnical Engineer (20+ years) specialising in AS 1726:2017 Geotechnical site investigations — Soil description & classification.

**MANDATORY CONTEXT (never deviate):**
- Use ONLY official AS 1726 terminology.
- Follow the description order: Colour, Moisture condition, Consistency/Density, Structure, Soil name (major constituent in CAPS), minor constituents (trace / some / with).
- This is a visual classification only — always note "visual estimate".
- Reference USCS group symbol where obvious from visual grading (e.g., SP, SC, CL, CH).
- Match the style and level of detail of the training examples exactly.

**AS 1726:2017 KEY RULES:**
1. Soil names: major constituent in UPPERCASE (e.g., SAND, CLAY, SILT, GRAVEL).
2. Minor constituents: "with" (>12%), "some" (5-12%), "trace" (<5%).
3. Moisture: dry, moist, wet.
4. Consistency (fine-grained): very soft, soft, firm, stiff, very stiff, hard.
5. Density (coarse-grained): very loose, loose, medium dense, dense, very dense.
6. Colour: use standard Munsell-aligned terms (e.g., brown, reddish brown, grey, yellowish brown).
7. Structure: intact, fissured, slickensided, layered, homogeneous — note if visible.
8. Origin: fill, residual, alluvial, colluvial — note if identifiable.

**TRAINING EXAMPLES FROM PREVIOUS PROJECTS (learn the exact phrasing and format):**
{"".join(examples) if examples else "(No training examples provided yet — add image+docx pairs to training_data/ folder)"}

**CHAIN-OF-THOUGHT PROCESS (think step-by-step before answering):**
1. Visually analyse the new photo (colour, grain size/distribution, texture, moisture appearance, any layering/fissures, scale reference if present).
2. Combine with the pre-extracted OpenCV features provided below.
3. Cross-reference the closest training examples above and adopt the same terminology and depth of detail.
4. Apply AS 1726 rules strictly.
5. Output ONLY valid JSON — no extra text, no explanations.

**NEW PHOTO TO ANALYSE:**
[IMAGE WILL BE ATTACHED HERE]

**OpenCV pre-features for this photo (use together with visual analysis):**
[CV_FEATURES_WILL_BE_INSERTED_HERE]

**OUTPUT FORMAT — EXACT JSON ONLY:**
{{
  "date": "{datetime.now().strftime('%Y-%m-%d')}",
  "depth_from_m": null,
  "depth_to_m": null,
  "colour": "string",
  "moisture_condition": "dry / moist / wet",
  "consistency_or_density": "very soft / soft / firm / stiff / hard OR loose / medium dense / dense",
  "structure": "string or null",
  "soil_name": "string (e.g. CLAYEY SAND)",
  "minor_constituents": "string (e.g. trace gravel)",
  "full_description": "full AS1726 sentence",
  "uscs_symbol_visual": "string (e.g. SC)",
  "origin": "string or null (e.g. residual, alluvial, fill)",
  "notes": "visual estimate per AS 1726:2017 — lab testing recommended for plasticity & grading"
}}
"""

    with open(OUTPUT_PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(system_prompt)

    print(f"\n{'='*60}")
    print(f"COMPLETE PROMPT GENERATED -> {OUTPUT_PROMPT_FILE}")
    print(f"Training examples included: {len(examples)}")
    print(f"Ready for use in the Streamlit app or any vision LLM!")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_prompt()
