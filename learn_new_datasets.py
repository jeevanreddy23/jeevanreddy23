import os
import cv2
import PyPDF2
from cv_analysis import analyse_soil_image, extract_cv_features_text

def get_pdf_text_chunk(pdf_path, identifier):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text and identifier in page_text:
                    # just grab a chunk around the identifier or the whole page
                    text += page_text + "\n"
            return text[:1000] # return first 1000 chars as context
    except Exception as e:
        return f"Could not read PDF: {e}"

def generate_example(img_path, pdf_path, identifier):
    if not os.path.exists(img_path) or not os.path.exists(pdf_path):
        return None
        
    img_cv = cv2.imread(img_path)
    if img_cv is None: return None
    
    cv_resText = extract_cv_features_text(img_cv)
    pdf_text = get_pdf_text_chunk(pdf_path, identifier)
    
    example = f"""
Example (New Dataset):
Image: {os.path.basename(img_path)}
CV features: {cv_resText}
Ground-Truth Log Text Extract (from {os.path.basename(pdf_path)}):
{pdf_text}
"""
    return example

ex1 = generate_example(
    r"C:\Users\rsing\Downloads\CG25-0354 Groundwater\Logs\BH01.JPEG",
    r"C:\Users\rsing\Downloads\CG25-0354 Groundwater\Logs\CG24-0354 BH01 to BH11.pdf",
    "BH01"
)

ex2 = generate_example(
    r"C:\Users\rsing\Downloads\CG24-0536 External Roads\Photo\TP22-1.JPEG",
    r"C:\Users\rsing\Downloads\CG24-0536 External Roads\Logs\CG24-0536 TP11 to TP17, TP22 and TP23.pdf",
    "TP22"
)

prompt_path = "as1726_complete_prompt.txt"
if os.path.exists(prompt_path):
    with open(prompt_path, "a", encoding="utf-8") as f:
        if ex1: f.write("\n" + ex1)
        if ex2: f.write("\n" + ex2)
    print("Successfully trained on BH01 (Groundwater) and TP22 (External Roads)!")
else:
    print("Could not find base prompt file.")
