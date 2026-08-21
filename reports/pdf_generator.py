import io
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)

from utils.logger import logger

class ForensicReportGenerator:
    """
    Generates downloadable, court/forensic-grade PDF intelligence reports
    documenting image authenticity, deep learning probabilities,
    filter detections, ELA compression, FFT spectrums, and Grad-CAM explainability.
    """

    def generate_report(
        self,
        results: Dict[str, Any],
        original_img: Image.Image,
        gradcam_img: Optional[Image.Image] = None,
        ela_img: Optional[Image.Image] = None,
        fft_img: Optional[Image.Image] = None
    ) -> bytes:
        """Builds a multi-section PDF document and returns the PDF bytes."""
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            alignment=0
        )
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569")
        )
        heading2_style = ParagraphStyle(
            'Heading2Custom',
            parent=styles['Heading2'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        verdict_style = ParagraphStyle(
            'VerdictText',
            parent=styles['Heading1'],
            fontSize=16,
            leading=20,
            textColor=colors.white,
            alignment=1
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("<b>AI IMAGE FORENSICS REPORT</b>", title_style))
        elements.append(Paragraph("Deep Learning, Computer Vision & Explainable AI Verification", subtitle_style))
        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceBefore=2, spaceAfter=8))

        # Metadata Header
        ensemble = results.get("ensemble", {})
        metadata = results.get("metadata", {})
        hashes = metadata.get("hashes", {})
        report_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        meta_table_data = [
            [
                Paragraph(f"<b>Timestamp:</b> {report_time}", body_style),
                Paragraph(f"<b>Dimensions:</b> {metadata.get('width', 0)} x {metadata.get('height', 0)} px", body_style)
            ],
            [
                Paragraph(f"<b>SHA-256:</b> <font size=7>{hashes.get('sha256', 'N/A')[:32]}...</font>", body_style),
                Paragraph(f"<b>Camera/Software:</b> {metadata.get('software', 'None')[:24]}", body_style)
            ]
        ]
        meta_table = Table(meta_table_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # 2. Executive Verdict Card
        verdict = ensemble.get("verdict", "REAL / NATURAL")
        color_hex = ensemble.get("verdict_color", "#10b981")
        verdict_color = colors.HexColor(color_hex)

        verdict_card_data = [
            [Paragraph(f"<b>FINAL VERDICT: {verdict.upper()}</b>", verdict_style)],
            [Paragraph(f"Authenticity Score: <b>{ensemble.get('authenticity_score', 0)}%</b> | Model Confidence: <b>{ensemble.get('overall_confidence', 0)}%</b>", ParagraphStyle('Sub', parent=verdict_style, fontSize=11, leading=14))]
        ]
        verdict_card = Table(verdict_card_data, colWidths=[540])
        verdict_card.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), verdict_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
        ]))
        elements.append(verdict_card)
        elements.append(Spacer(1, 10))

        # 3. Probability Distribution Breakdown Table
        probs = ensemble.get("probabilities", {})
        ai_pred = results.get("ai_detection", {})
        
        prob_table_data = [
            [
                Paragraph("<b>Category</b>", body_style),
                Paragraph("<b>Probability</b>", body_style),
                Paragraph("<b>Key Signal / Indicator</b>", body_style)
            ],
            [Paragraph("Real / Natural", body_style), f"{probs.get('real', 0)}%", "Natural optical sensor noise & 1/f spectral profile"],
            [Paragraph("AI-Generated", body_style), f"{probs.get('ai_generated', 0)}%", f"Generative artifact pattern ({ai_pred.get('top_subfamily', 'Synthetic')[:25]})"],
            [Paragraph("AI-Edited / Inpainted", body_style), f"{probs.get('ai_edited', 0)}%", "Localized noise and gradient inconsistencies"],
            [Paragraph("Filtered", body_style), f"{probs.get('filtered', 0)}%", f"{results.get('filter_detection', {}).get('filter_count', 0)} filters active"],
            [Paragraph("Manipulated / Spliced", body_style), f"{probs.get('manipulated', 0)}%", "ELA compression mismatch & edge gradient variance"]
        ]
        prob_table = Table(prob_table_data, colWidths=[140, 90, 310])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
        ]))
        elements.append(Paragraph("<b>Authenticity Probability Distribution</b>", heading2_style))
        elements.append(prob_table)
        elements.append(Spacer(1, 10))

        # 4. Visual Forensic Panels (Original, Grad-CAM, ELA, FFT)
        elements.append(Paragraph("<b>Visual Forensic & Explainability Panels</b>", heading2_style))
        
        def pil_to_flowable(pil_im, width=125, height=125):
            if pil_im is None:
                return Paragraph("N/A", body_style)
            buf = io.BytesIO()
            if isinstance(pil_im, np.ndarray):
                pil_im = Image.fromarray(pil_im)
            pil_im.convert("RGB").save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return RLImage(buf, width=width, height=height)

        visual_cells = [
            [
                pil_to_flowable(original_img),
                pil_to_flowable(gradcam_img),
                pil_to_flowable(ela_img),
                pil_to_flowable(fft_img)
            ],
            [
                Paragraph("<b>(a) Original Input</b>", body_style),
                Paragraph("<b>(b) Grad-CAM Heatmap</b>", body_style),
                Paragraph("<b>(c) Error Level Analysis</b>", body_style),
                Paragraph("<b>(d) 2D FFT Spectrum</b>", body_style)
            ]
        ]
        visual_table = Table(visual_cells, colWidths=[135, 135, 135, 135])
        visual_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(visual_table)
        elements.append(Spacer(1, 10))

        # 5. Explainable AI & Forensic Reasoning
        explain = results.get("explainability", {})
        elements.append(Paragraph("<b>Explainable AI & Forensic Reasoning</b>", heading2_style))
        elements.append(Paragraph(f"<b>Model Decision Rationale:</b> {explain.get('forensic_reasoning', 'Analysis complete.')}", body_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"<i>Disclaimer: {explain.get('disclaimer', '')}</i>", ParagraphStyle('Disc', parent=body_style, fontSize=7, textColor=colors.HexColor("#64748b"))))

        # Build Document
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
