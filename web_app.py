"""
web_app.py
FastAPI backend serving the premium Geotech web UI.
"""
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from io import BytesIO

# Import our custom CV logic and PDF generator
from cv_analysis import analyse_soil_image
from pdf_report import generate_borehole_log_pdf

app = FastAPI(title="Geotech Soil Data Logger API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
os.makedirs("temp_uploads", exist_ok=True)

# Mount static directory for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def analyze_soil(
    file: UploadFile = File(...),
    proj_name: str = Form(""),
    bh_id: str = Form(""),
    depth_from: float = Form(0.0),
    depth_to: float = Form(0.0)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse({"status": "error", "message": "Invalid image file."}, status_code=400)

        # 1. Run Analysis
        cv_res = analyse_soil_image(img_bgr)
        
        # 2. Return data
        data = {
            "status": "success",
            "colour": cv_res.colour_description,
            "moisture_condition": cv_res.moisture_condition,
            "consistency_or_density": cv_res.consistency,
            "uscs_symbol_visual": cv_res.uscs_visual_estimate,
            "full_description": cv_res.full_as1726_description,
            "avg_particle_area": round(cv_res.avg_particle_area_px, 1),
            "texture_variance": round(cv_res.texture_variance, 1),
            "particle_count": cv_res.particle_count
        }
        return JSONResponse(data)

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/generate_pdf")
async def generate_pdf(
    proj_name: str = Form(""),
    bh_id: str = Form(""),
    depth_from: float = Form(0.0),
    depth_to: float = Form(0.0),
    colour: str = Form(""),
    moisture: str = Form(""),
    consistency: str = Form(""),
    uscs_symbol: str = Form(""),
    full_description: str = Form("")
):
    try:
        log_entry = {
            "depth_from": depth_from,
            "depth_to": depth_to,
            "colour": colour,
            "moisture": moisture,
            "consistency": consistency,
            "uscs_symbol": uscs_symbol,
            "full_description": full_description,
        }

        pdf_bytes = generate_borehole_log_pdf([log_entry], project_name=proj_name, borehole_id=bh_id)
        
        pdf_path = f"temp_uploads/{bh_id}_Report.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
            
        return FileResponse(pdf_path, media_type='application/pdf', filename=f"{bh_id}_Report.pdf")

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
