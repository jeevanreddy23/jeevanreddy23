import cv2
import json
from cv_analysis import analyse_soil_image
from pdf_report import generate_borehole_log_pdf

# Run on one of the new dataset photos from Colleen Lane Wyee
img_path = r"C:\Users\rsing\Downloads\CG26-0101 Colleen Lane Wyee NSW\6. Site Photos\PXL_20260115_230846089.jpg"
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    print(f"Failed to load image at {img_path}")
else:
    # 1. Run strict AS1726 OpenCV analysis
    cv_res = analyse_soil_image(img_bgr)
    
    # Structure strictly per AS 1726:2017 Sections 6.1.4:
    # Colour, Moisture, Consistency/Density, Structure, Soil Name, Minor constituents, Origin
    
    output = {
        "Project": "CG26-0101 Colleen Lane Wyee NSW",
        "Source Image": img_path.split('\\')[-1],
        "Standard": "Strictly AS 1726:2017",
        "Classification": {
            "Colour (Table 6.1.4.f)": cv_res.colour_description.title(),
            "Moisture Condition (Table 6.1.6)": cv_res.moisture_condition.title(),
            "Consistency / Density (Table 6.1.7)": cv_res.consistency.title(),
            "Primary Soil Name (Table 6.1.4.d/e)": cv_res.primary_soil_name,
            "Minor Constituents (Table 6.1.4.e)": cv_res.minor_constituents,
            "Est. USCS Group Symbol": cv_res.uscs_visual_estimate
        },
        "Formatted Logging Sentence (AS1726 strict)": cv_res.full_as1726_description,
    }
    
    print(json.dumps(output, indent=4))
