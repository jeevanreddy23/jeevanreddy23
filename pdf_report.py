"""
pdf_report.py
Generate AS 1726 style borehole/test pit log PDF reports using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import numpy as np
from PIL import Image as PILImage


def _np_to_rl_image(np_img, width_mm=60):
    """Convert numpy image array to a ReportLab Image."""
    if len(np_img.shape) == 2:
        pil_img = PILImage.fromarray(np_img, mode='L')
    else:
        import cv2
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return RLImage(buf, width=width_mm * mm, height=width_mm * mm * pil_img.height / pil_img.width)


def generate_borehole_log_pdf(
    log_entries: list,
    project_name: str = "Geotechnical Investigation",
    borehole_id: str = "BH-01",
    location: str = "",
    rl_surface: float = 0.0,
    sample_image: np.ndarray = None,
) -> bytes:
    """
    Generate a professional borehole log PDF.

    log_entries: list of dicts with keys:
        depth_from, depth_to, colour, moisture, consistency, grain_desc,
        full_description, uscs_symbol, minor_constituents
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=20 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'BHTitle', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor('#2C3E50'),
        spaceAfter=4 * mm
    )
    subtitle_style = ParagraphStyle(
        'BHSubtitle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#5D6D7E'),
        spaceAfter=2 * mm
    )
    cell_style = ParagraphStyle(
        'Cell', parent=styles['Normal'],
        fontSize=8, leading=10, alignment=TA_LEFT
    )
    header_style = ParagraphStyle(
        'Header', parent=styles['Normal'],
        fontSize=8, leading=10, alignment=TA_CENTER,
        textColor=colors.white, fontName='Helvetica-Bold'
    )

    elements = []

    # ── Header ──
    elements.append(Paragraph(f"BOREHOLE / TEST PIT LOG — {borehole_id}", title_style))
    elements.append(Paragraph(
        f"Project: {project_name} &nbsp;|&nbsp; Location: {location or 'N/A'} &nbsp;|&nbsp; "
        f"RL Surface: {rl_surface:.1f} m &nbsp;|&nbsp; Date: {datetime.now().strftime('%Y-%m-%d')}",
        subtitle_style
    ))
    elements.append(Paragraph(
        "Classification per AS 1726:2017 — Visual estimate only. "
        "Lab confirmation recommended for grading and plasticity.",
        ParagraphStyle('Disclaimer', parent=styles['Normal'],
                       fontSize=7, textColor=colors.HexColor('#E74C3C'))
    ))
    elements.append(Spacer(1, 4 * mm))

    # ── Optional sample image ──
    if sample_image is not None:
        try:
            img_el = _np_to_rl_image(sample_image, width_mm=50)
            elements.append(img_el)
            elements.append(Spacer(1, 3 * mm))
        except Exception:
            pass

    # ── Log table ──
    col_widths = [18 * mm, 18 * mm, 28 * mm, 20 * mm, 22 * mm, 22 * mm, 55 * mm]
    headers = ["Depth\nFrom (m)", "Depth\nTo (m)", "Colour", "Moisture", "Cons./\nDensity",
               "USCS\n(visual)", "Soil Description (AS 1726)"]

    header_row = [Paragraph(h.replace("\n", "<br/>"), header_style) for h in headers]
    data = [header_row]

    for entry in log_entries:
        row = [
            Paragraph(str(entry.get("depth_from", "-")), cell_style),
            Paragraph(str(entry.get("depth_to", "-")), cell_style),
            Paragraph(entry.get("colour", ""), cell_style),
            Paragraph(entry.get("moisture", ""), cell_style),
            Paragraph(entry.get("consistency", ""), cell_style),
            Paragraph(entry.get("uscs_symbol", ""), cell_style),
            Paragraph(entry.get("full_description", ""), cell_style),
        ]
        data.append(row)

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        # Body
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(table)

    # ── Footer ──
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph(
        "Generated by AS 1726 Soil Automatic Data Logger | OpenCV + Streamlit | "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle('Footer', parent=styles['Normal'],
                       fontSize=7, textColor=colors.HexColor('#95A5A6'), alignment=TA_CENTER)
    ))

    doc.build(elements)
    return buffer.getvalue()
