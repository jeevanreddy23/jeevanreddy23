"""
soil_logger.py
Main Streamlit application for AS 1726 Soil Automatic Data Logger.
Integration of OpenCV analysis, PDF reporting, and LLM-based "Training" mode.
"""

import streamlit as st
import cv2
import numpy as np
import os
import io
import json
import base64
import pandas as pd
from PIL import Image
from datetime import datetime
from typing import Optional

# Local imports
from cv_analysis import analyse_soil_image, extract_cv_features_text
from pdf_report import generate_borehole_log_pdf

# Optional dependency: openai
try:
    import openai
except ImportError:
    openai = None

# --- UI Configuration ---
st.set_page_config(
    page_title="AS1726 Soil Logger",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium geotech aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #D4763C;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #B35F2B;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #D4763C !important;
    }
    .stAlert {
        background-color: #1A1D26;
        color: #FAFAFA;
        border: 1px solid #D4763C;
    }
    .css-1offfwp {
        background-color: #1A1D26 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1A1D26;
        border-right: 1px solid #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

# --- App Logic ---

def main():
    st.title("🪨 AS 1726 Soil Automatic Data Logger")
    st.write("Professional Geotechnical Site Investigation Support with Computer Vision & LLM")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio("Tool Mode", ["Analysis (Vision)", "Training (Prompt Builder)", "Batch Export"])
        
        st.markdown("---")
        st.subheader("🤖 LLM Integration")
        use_llm = st.toggle("Enable GPT-4o Enhanced Analysis", value=False)
        
        api_key = st.text_input("OpenAI API Key", type="password") if use_llm else ""
        
        st.subheader("📏 Calibration")
        scale_factor = st.number_input("Pixel-to-mm scale (optional)", value=1.0, help="For accurate grain sizing")
        
        st.markdown("---")
        st.caption("Aligned with AS 1726:2017 Geotechnical site investigations.")

    if mode == "Analysis (Vision)":
        run_analysis_tab(use_llm, api_key)
    elif mode == "Training (Prompt Builder)":
        run_training_tab()
    elif mode == "Batch Export":
        run_export_tab()

def run_analysis_tab(use_llm: bool, api_key: str):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Image Upload")
        uploaded_file = st.file_uploader("Upload soil sample or core photo", type=["jpg", "jpeg", "png", "heic"])
        
        # Metadata inputs
        st.markdown("### 📝 Log Metadata")
        proj_name = st.text_input("Project Name", "Caddies Creek Stage 2")
        bh_id = st.text_input("Borehole ID", "BH-01")
        depth_from = st.number_input("Depth From (m)", min_value=0.0, step=0.1, value=1.5)
        depth_to = st.number_input("Depth To (m)", min_value=0.0, step=0.1, value=2.0)

    if uploaded_file is not None:
        # Load and display
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary (e.g., RGBA or HEIC/Pillow)
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with col2:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        st.markdown("---")
        
        if st.button("🚀 Analyze Soil Sample"):
            with st.spinner("Processing computer vision features..."):
                # 1. Run OpenCV Analysis
                cv_result = analyse_soil_image(img_cv)
                
                # 2. GPT-4o Enhanced Pass
                llm_result = None
                if use_llm and api_key:
                    if not openai:
                        st.error("OpenAI library not found. Run `pip install openai`.")
                    else:
                        llm_result = run_llm_analysis(img_cv, api_key)
                
                # 3. Display Results
                display_results(cv_result, llm_result, proj_name, bh_id, depth_from, depth_to)

def run_llm_analysis(img_bgr, api_key):
    """Call GPT-4o vision with pre-extracted CV features."""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Load the prompt file
        prompt_path = "as1726_complete_prompt.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                full_prompt = f.read()
        else:
            # Fallback simple prompt
            full_prompt = "Analyze the soil in this image per AS1726:2017. Return valid JSON."

        # Insert CV features
        cv_feats = extract_cv_features_text(img_bgr)
        full_prompt = full_prompt.replace("[CV_FEATURES_WILL_BE_INSERTED_HERE]", cv_feats)

        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', img_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional geotechnical engineer."},
                {"role": "user", "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

def display_results(cv_res, llm_res, proj, bh, d_from, d_to):
    # Use LLM results if available, otherwise CV
    data = {}
    if llm_res:
        st.success("Analysis Complete (AI Enhanced)")
        data = llm_res
    else:
        st.success("Analysis Complete (Vision Only)")
        data = {
            "colour": cv_res.colour_description,
            "moisture_condition": cv_res.moisture_condition,
            "consistency_or_density": cv_res.consistency,
            "soil_name": cv_res.primary_soil_name,
            "minor_constituents": cv_res.minor_constituents,
            "full_description": cv_res.full_as1726_description,
            "uscs_symbol_visual": cv_res.uscs_visual_estimate
        }

    # UI Layout for Results
    c1, c2, c3 = st.columns(3)
    c1.metric("Colour", data.get("colour", "-").title())
    c2.metric("Moisture", data.get("moisture_condition", "-").title())
    c3.metric("Consistency/Density", data.get("consistency_or_density", "-").title())
    
    st.subheader("📄 Generated Soil Description")
    st.info(f"**{data.get('full_description', '-')}**")
    
    with st.expander("🔍 Behind the Scenes (CV Analysis Tools)"):
        st.write(f"Avg Particle Area: {cv_res.avg_particle_area_px:.1f} px")
        v1, v2, v3 = st.columns(3)
        v1.image(cv_res.grayscale_img, caption="Grayscale (Texture Analysis)")
        v2.image(cv_res.threshold_img, caption="Threshold (Particle Detection)")
        v3.image(cv_res.colour_histogram_img, caption="RGB Colour Profile")

    # Data Table
    log_entry = {
        "depth_from": d_from,
        "depth_to": d_to,
        "colour": data.get("colour", ""),
        "moisture": data.get("moisture_condition", ""),
        "consistency": data.get("consistency_or_density", ""),
        "uscs_symbol": data.get("uscs_symbol_visual", ""),
        "full_description": data.get("full_description", ""),
    }
    df = pd.DataFrame([log_entry])
    st.dataframe(df, use_container_width=True)

    # Export
    pdf_bytes = generate_borehole_log_pdf([log_entry], proj, bh)
    st.download_button("📥 Download PDF Log", pdf_bytes, f"{bh}_{d_from}-{d_to}m.pdf", "application/pdf")
    
    # Store for session
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(log_entry)

def run_training_tab():
    st.subheader("📚 In-Context Prompt Builder (Training)")
    st.write("Automatically pair your previous photos and Word reports to train the LLM.")
    
    if st.button("🏗️ Build Prompt (build_soil_prompt.py)"):
        with st.spinner("Analyzing training folder..."):
            # Run the local script
            import subprocess
            result = subprocess.run(["python", "build_soil_prompt.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Complete prompt generated!")
                st.code(result.stdout)
                
                if os.path.exists("as1726_complete_prompt.txt"):
                    with open("as1726_complete_prompt.txt", "r", encoding="utf-8") as f:
                        content = f.read()
                    st.download_button("Download Generated Prompt", content, "as1726_complete_prompt.txt")
            else:
                st.error("Failed to build prompt.")
                st.code(result.stderr)
    
    # Show current training pairs
    train_dir = "training_data"
    if os.path.exists(train_dir):
        files = os.listdir(train_dir)
        pairs = set([os.path.splitext(f)[0] for f in files if f.endswith(('.jpg', '.docx'))])
        st.write(f"Found **{len(pairs)}** valid training pairs in `./training_data/`")
        for p in sorted(list(pairs)):
            st.write(f"- ✅ {p}")

def run_export_tab():
    st.subheader("📂 Batch Data Export")
    if 'history' in st.session_state and st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Export Session Data (CSV)", csv, "batch_soil_logs.csv", "text/csv")
        
        if st.button("Generate Combined PDF Report"):
            pdf = generate_borehole_log_pdf(st.session_state.history)
            st.download_button("Download Combined PDF", pdf, "combined_log.pdf")
    else:
        st.info("No logs processed in this session yet.")

if __name__ == "__main__":
    main()
