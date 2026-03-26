"""
cv_analysis.py
Core OpenCV computer vision module for soil sample analysis.
Implements color detection, grain/particle sizing, texture analysis,
and moisture estimation per AS 1726:2017 visual classification principles.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class SoilAnalysisResult:
    """Structured result from OpenCV soil analysis."""
    # Colour
    colour_description: str = ""
    colour_rgb: Tuple[int, int, int] = (0, 0, 0)
    colour_munsell_approx: str = ""
    # Grain / particle
    grain_description: str = ""
    grain_category: str = ""  # coarse / medium / fine
    avg_particle_area_px: float = 0.0
    particle_count: int = 0
    particle_size_distribution: dict = field(default_factory=dict)
    # Texture
    texture_variance: float = 0.0
    consistency: str = ""
    # Moisture
    brightness: float = 0.0
    moisture_condition: str = ""
    # Soil naming
    primary_soil_name: str = ""
    minor_constituents: str = ""
    full_as1726_description: str = ""
    uscs_visual_estimate: str = ""
    # Debug images
    grayscale_img: Optional[np.ndarray] = None
    threshold_img: Optional[np.ndarray] = None
    contour_img: Optional[np.ndarray] = None
    colour_histogram_img: Optional[np.ndarray] = None


# ── AS 1726 Colour Mapping (HSV-based) ──────────────────────────────────────

COLOUR_MAP_HSV = [
    # (hue_min, hue_max, sat_min, val_min, description, munsell_approx)
    (0, 10, 40, 40, "red", "2.5YR"),
    (10, 20, 40, 40, "reddish brown", "5YR"),
    (20, 30, 40, 40, "brown", "7.5YR"),
    (30, 45, 40, 40, "yellowish brown", "10YR"),
    (45, 65, 40, 40, "yellow", "2.5Y"),
    (65, 85, 20, 40, "olive", "5Y"),
    (85, 130, 20, 40, "greenish grey", "5GY"),
    (130, 160, 20, 40, "blue grey", "5PB"),
    (160, 180, 40, 40, "red", "10R"),
    # Low saturation catches
    (0, 180, 0, 0, "grey", "N"),
]


def classify_colour(mean_hsv: Tuple[float, float, float]) -> Tuple[str, str]:
    """Map mean HSV values to AS 1726 colour description."""
    h, s, v = mean_hsv

    # Very dark → black/dark grey
    if v < 50:
        return "dark grey to black", "N2"

    # Low saturation → grey tones
    if s < 25:
        if v > 170:
            return "light grey", "N7"
        elif v > 100:
            return "grey", "N5"
        else:
            return "dark grey", "N3"

    # Chromatic classification
    for h_min, h_max, s_min, v_min, desc, munsell in COLOUR_MAP_HSV[:-1]:
        if h_min <= h < h_max and s >= s_min and v >= v_min:
            # Refine lightness
            if v > 170:
                return f"light {desc}", munsell
            elif v < 80:
                return f"dark {desc}", munsell
            return desc, munsell

    return "grey", "N5"


# ── Particle / Grain Size Analysis ──────────────────────────────────────────

GRAIN_THRESHOLDS = {
    # area in pixels (approximate, depends on image resolution and scale)
    "boulder_cobble": 50000,
    "gravel": 5000,
    "coarse_sand": 1500,
    "medium_sand": 500,
    "fine_sand": 100,
    # below fine_sand → silt/clay (visually indistinguishable)
}


def analyse_particles(gray: np.ndarray) -> dict:
    """Detect and classify particles by contour area."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold for varied lighting
    adapt_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(adapt_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    min_area = 30
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Classify into size bins
    size_bins = {
        "boulder_cobble": 0,
        "gravel": 0,
        "coarse_sand": 0,
        "medium_sand": 0,
        "fine_sand": 0,
        "fines": 0,
    }

    areas = []
    for c in valid_contours:
        a = cv2.contourArea(c)
        areas.append(a)
        if a >= GRAIN_THRESHOLDS["boulder_cobble"]:
            size_bins["boulder_cobble"] += 1
        elif a >= GRAIN_THRESHOLDS["gravel"]:
            size_bins["gravel"] += 1
        elif a >= GRAIN_THRESHOLDS["coarse_sand"]:
            size_bins["coarse_sand"] += 1
        elif a >= GRAIN_THRESHOLDS["medium_sand"]:
            size_bins["medium_sand"] += 1
        elif a >= GRAIN_THRESHOLDS["fine_sand"]:
            size_bins["fine_sand"] += 1
        else:
            size_bins["fines"] += 1

    return {
        "contours": valid_contours,
        "areas": areas,
        "size_bins": size_bins,
        "threshold_img": cleaned,
    }


def determine_grain_description(size_bins: dict, avg_area: float) -> tuple[str, str, str, str]:
    total = sum(size_bins.values())
    if total == 0:
        return "homogeneous fine material", "fine", "Silty CLAY", "trace fine sand"

    pcts = {k: (v / total * 100) for k, v in size_bins.items()}
    coarse_pct = pcts["boulder_cobble"] + pcts["gravel"]
    sand_pct = pcts["coarse_sand"] + pcts["medium_sand"] + pcts["fine_sand"]
    fines_pct = pcts["fines"]

    primary = ""
    prefix = ""
    category = "fine"
    plasticity = "medium plasticity"
    minors = []

    if coarse_pct > 50:
        category = "coarse"
        primary = "GRAVEL"
        if sand_pct > 15: prefix = "Sandy "
        elif fines_pct > 15: prefix = "Silty " if pcts.get("fines", 0) < 20 else "Clayey "
        
        plasticity = "fine to coarse"
        if sand_pct > 5 and sand_pct <= 15: minors.append("trace sand")
        if fines_pct > 5 and fines_pct <= 15: minors.append("trace fines")
        
    elif sand_pct > 50:
        category = "medium"
        primary = "SAND"
        if coarse_pct > 15: prefix = "Gravelly "
        elif fines_pct > 15: prefix = "Silty " if pcts.get("fines", 0) < 20 else "Clayey "
        
        plasticity = "fine to medium grained"
        if coarse_pct > 5 and coarse_pct <= 15: minors.append("trace gravel")
        if fines_pct > 5 and fines_pct <= 15: minors.append("trace fines")
        
    else:
        category = "fine"
        primary = "CLAY"
        if sand_pct > 15: prefix = "Sandy "
        elif coarse_pct > 15: prefix = "Gravelly "
        else: prefix = "Silty "
        
        plasticity = "medium plasticity"
        if sand_pct > 5 and sand_pct <= 15: minors.append("trace fine sand")
        if coarse_pct > 5 and coarse_pct <= 15: minors.append("trace fine gravel")

    soil_name = f"{prefix}{primary}".strip()
    return plasticity, category, soil_name, " and ".join(minors)


# ── Texture / Consistency / Density ─────────────────────────────────────────

def estimate_consistency(texture_std: float, grain_category: str) -> str:
    """
    Estimate consistency (fine-grained) or density (coarse-grained)
    from grey-level standard deviation.
    """
    if grain_category == "fine":
        # Fine-grained soils → consistency
        if texture_std > 65:
            return "stiff to hard"
        elif texture_std > 50:
            return "firm to stiff"
        elif texture_std > 35:
            return "firm"
        elif texture_std > 20:
            return "soft to firm"
        else:
            return "very soft to soft"
    else:
        # Coarse-grained soils → density
        if texture_std > 60:
            return "dense"
        elif texture_std > 35:
            return "medium dense"
        else:
            return "loose"


# ── Moisture Estimation ─────────────────────────────────────────────────────

def estimate_moisture(brightness: float, saturation: float) -> str:
    """Estimate moisture condition from brightness + colour saturation."""
    if brightness < 80:
        return "wet"
    elif brightness < 130 and saturation > 40:
        return "moist"
    elif brightness < 150:
        return "moist"
    else:
        return "dry"


# ── USCS Visual Estimate ────────────────────────────────────────────────────

def estimate_uscs(grain_category: str, size_bins: dict, fines_prop: float) -> str:
    """Provide a rough USCS group symbol based on visual grading."""
    if grain_category == "fine":
        return "CL-ML (visual estimate)"
    elif grain_category == "coarse":
        if fines_prop > 0.12:
            return "GC-GM (visual estimate)"
        else:
            return "GP-GW (visual estimate)"
    else:  # medium (sand)
        if fines_prop > 0.12:
            return "SC-SM (visual estimate)"
        elif fines_prop > 0.05:
            return "SW-SM (visual estimate)"
        else:
            return "SP-SW (visual estimate)"


# ── Draw Colour Histogram ───────────────────────────────────────────────────

def draw_colour_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """Create an RGB histogram visualization."""
    hist_h = 200
    hist_w = 512
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8) + 20

    colours = ('b', 'g', 'r')
    colour_vals = ((200, 120, 60), (60, 200, 100), (60, 100, 220))

    for i, (col_name, col_val) in enumerate(zip(colours, colour_vals)):
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_h - 10, cv2.NORM_MINMAX)
        pts = np.column_stack([
            np.linspace(0, hist_w - 1, 256).astype(int),
            (hist_h - hist.flatten()).astype(int)
        ])
        cv2.polylines(hist_img, [pts], False, col_val, 2, cv2.LINE_AA)

    return hist_img


# ── Main Analysis Function ──────────────────────────────────────────────────

def analyse_soil_image(img_bgr: np.ndarray) -> SoilAnalysisResult:
    """
    Full OpenCV pipeline for soil sample analysis.
    Input: BGR image (OpenCV format).
    Returns: SoilAnalysisResult dataclass.
    """
    result = SoilAnalysisResult()

    # ── 1. Colour Analysis ──────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv)[:3]
    colour_desc, munsell = classify_colour(mean_hsv)
    mean_bgr = cv2.mean(img_bgr)[:3]
    result.colour_description = colour_desc
    result.colour_rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
    result.colour_munsell_approx = munsell

    # ── 2. Grayscale + Particle Detection ───────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    result.grayscale_img = gray

    particle_data = analyse_particles(gray)
    result.threshold_img = particle_data["threshold_img"]
    result.particle_count = len(particle_data["contours"])
    result.particle_size_distribution = particle_data["size_bins"]

    areas = particle_data["areas"]
    avg_area = float(np.mean(areas)) if areas else 0.0
    result.avg_particle_area_px = avg_area

    # Draw contours
    contour_vis = img_bgr.copy()
    cv2.drawContours(contour_vis, particle_data["contours"], -1, (0, 255, 120), 1)
    result.contour_img = contour_vis

    # ── 3. Grain Description & Soil Naming ──────
    grain_desc, grain_cat, soil_name, minors = determine_grain_description(
        particle_data["size_bins"], avg_area
    )
    result.grain_description = grain_desc
    result.grain_category = grain_cat
    result.primary_soil_name = soil_name
    result.minor_constituents = minors

    # ── 4. Texture / Consistency ────────────────
    texture_std = float(np.std(gray))
    result.texture_variance = texture_std
    result.consistency = estimate_consistency(texture_std, grain_cat)

    # ── 5. Moisture ─────────────────────────────
    brightness = float(np.mean(gray))
    mean_sat = mean_hsv[1]
    result.brightness = brightness
    result.moisture_condition = estimate_moisture(brightness, mean_sat)

    # ── 6. USCS Estimate ────────────────────────
    total_particles = sum(particle_data["size_bins"].values())
    fines_prop = particle_data["size_bins"].get("fines", 0) / max(total_particles, 1)
    result.uscs_visual_estimate = estimate_uscs(grain_cat, particle_data["size_bins"], fines_prop)

    # ── 7. Custom Prompted Log Description ─────────────
    # Format: Soil Name, grading/plasticity, colour, minor constituents, moisture
    parts = []
    parts.append(result.primary_soil_name)
    if result.grain_description:
        parts.append(result.grain_description) # stores plasticity/grading now
    if result.colour_description:
        parts.append(result.colour_description)
    if result.minor_constituents:
        parts.append(result.minor_constituents)
    if result.moisture_condition:
        parts.append(result.moisture_condition)
        
    result.full_as1726_description = ", ".join(parts)

    # ── 8. Colour Histogram ─────────────────────
    result.colour_histogram_img = draw_colour_histogram(img_bgr)

    return result


def extract_cv_features_text(img_bgr: np.ndarray) -> str:
    """Extract a text summary of CV features (for LLM prompt injection)."""
    r = analyse_soil_image(img_bgr)
    return (
        f"CV features -> Colour: {r.colour_description} (Munsell ~{r.colour_munsell_approx}), "
        f"Grain: {r.grain_description}, "
        f"Particles detected: {r.particle_count}, "
        f"Avg particle area: {int(r.avg_particle_area_px)} px, "
        f"Consistency/Density: {r.consistency}, "
        f"Moisture hint: {r.moisture_condition}, "
        f"Texture std: {r.texture_variance:.1f}"
    )
