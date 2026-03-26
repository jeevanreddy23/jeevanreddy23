import sys

with open("cv_analysis.py", "r", encoding="utf-8") as f:
    text = f.read()

new_func = """def determine_grain_description(size_bins: dict, avg_area: float) -> tuple[str, str, str, str]:
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
"""

rep1_start = text.find("def determine_grain_description(size_bins: dict, avg_area: float)")
rep1_end = text.find("# ── Texture / Consistency / Density")
if rep1_start != -1 and rep1_end != -1:
    text = text[:rep1_start] + new_func + "\n\n" + text[rep1_end:]

rep2_start = text.find("# ── 7. AS 1726 Full Description")
rep2_end = text.find("# ── 8. Colour Histogram")
if rep2_start != -1 and rep2_end != -1:
    new_desc = """# ── 7. Custom Prompted Log Description ─────────────
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

    """
    text = text[:rep2_start] + new_desc + text[rep2_end:]

with open("cv_analysis.py", "w", encoding="utf-8") as f:
    f.write(text)
print("Updated cv_analysis.py successfully.")
