"""
Phase 16: Doctor Report Generator.
Generates beautiful, self-contained clinical HTML diagnostic summaries.
Embeds demographics, prediction confidence bar charts, and Base64-encoded Grad-CAM overlays
so the resulting document is 100% portable and printable to PDF.
"""

import os
import logging
import base64
from io import BytesIO
from PIL import Image
import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("SkinCancerAI.ReportGenerator")


class ClinicalReportGenerator:
    """
    Assembles diagnostics and visual explanations into printable,
    clinician-friendly HTML summary documents.
    """

    # Medical dictionary mapping target labels to clinical names and diagnostic definitions
    DISEASE_INFO = {
        "akiec": {
            "name": "Actinic Keratosis / Intraepithelial Carcinoma (Bowen's Disease)",
            "severity": "Pre-malignant",
            "desc": "A rough, scaly patch on the skin caused by years of sun exposure. Can progress to invasive squamous cell carcinoma if left untreated.",
            "guideline": "Recommend cryotherapy, topical chemotherapy (5-fluorouracil), or photodynamic therapy. Monitor closely for signs of invasion."
        },
        "bcc": {
            "name": "Basal Cell Carcinoma",
            "severity": "Malignant (Locally Invasive)",
            "desc": "A type of skin cancer that begins in the basal cells. Rarely metastasizes but can be locally destructive to surrounding tissues.",
            "guideline": "Recommend surgical excision, Mohs micrographic surgery, or electrodessication and curettage. Good prognosis with complete excision."
        },
        "bkl": {
            "name": "Benign Keratosis-like Lesion (Solar Lentigo / Seborrheic Keratosis)",
            "severity": "Benign",
            "desc": "Non-cancerous skin growth that typically appears waxy, scaly, or 'stuck on' the skin. Common in older adults.",
            "guideline": "No immediate clinical treatment required unless symptomatic, irritated, or removed for cosmetic reasons."
        },
        "df": {
            "name": "Dermatofibroma",
            "severity": "Benign",
            "desc": "A common, harmless, firm red-to-brown nodule typically found on the lower extremities. Often displays a 'dimple sign' when squeezed.",
            "guideline": "Benign lesion. No intervention required. Reassure the patient."
        },
        "mel": {
            "name": "Melanoma",
            "severity": "Highly Malignant (Critical)",
            "desc": "The most serious type of skin cancer, developing in the pigment-producing melanocytes. High potential for metastasis and fatal outcome if treatment is delayed.",
            "guideline": "Urgent surgical excision with appropriate margins. Dermatological referral for urgent staging, lymph node biopsy, and systemic therapy if indicated."
        },
        "nv": {
            "name": "Melanocytic Nevus (Common Mole)",
            "severity": "Benign",
            "desc": "A common, benign accumulation of melanocytes. Can be congenital or acquired.",
            "guideline": "Benign mole. Regular self-examination recommended using the ABCDE guidelines. Reassess if changes occur."
        },
        "vasc": {
            "name": "Vascular Lesion (Cherry Angioma / Pyogenic Granuloma)",
            "severity": "Benign",
            "desc": "Harmless skin growths made of blood vessels. Can range from cherry angiomas to pyogenic granulomas that bleed easily.",
            "guideline": "Usually benign. Recommend conservative management. Excision or laser ablation may be used if recurrent bleeding or diagnostic uncertainty exists."
        },
        "unknown": {
            "name": "Unknown Category (Low Confidence)",
            "severity": "Undetermined (Requires Consultation)",
            "desc": "The classification model could not categorize this skin lesion with sufficient confidence. The visual textures do not match any trained disease patterns.",
            "guideline": "Low confidence prediction. This image does not sufficiently match any trained disease category. Please consult a dermatologist."
        }
    }

    def __init__(self, output_dir: str = "reports"):
        """
        Initializes the report generator.

        Args:
            output_dir (str): Folder where HTML reports will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ClinicalReportGenerator configured to save reports to: {self.output_dir}")

    def _convert_pil_to_base64(self, img: Image.Image) -> str:
        """Helper to convert a Pillow Image object to a base64 encoded string."""
        buffered = BytesIO()
        # Save as PNG inside the buffer
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def generate_report(
        self,
        patient_id: str,
        demographics: Dict[str, Any],
        predictions: Dict[str, float],
        original_image: Image.Image,
        heatmap_image: Image.Image,
        filename: str = "clinical_report.html",
        predicted_class: Optional[str] = None
    ) -> str:
        """
        Compiles patient parameters, logits, and heatmaps into a styled HTML clinical report.

        Args:
            patient_id (str): Unique patient/case identifier.
            demographics (Dict[str, Any]): Dictionary containing 'age', 'sex', and 'localization'.
            predictions (Dict[str, float]): Class mapping to probabilities (e.g. {'mel': 0.85, ...}).
            original_image (Image.Image): Pillow image of the raw skin lesion.
            heatmap_image (Image.Image): Pillow image of the Grad-CAM overlay.
            filename (str): Target filename for saving.
            predicted_class (Optional[str]): Explicit override class name.

        Returns:
            str: Absolute file path of the generated HTML report.
        """
        logger.info(f"Generating clinical report for case: {patient_id}...")

        # 1. Determine predicted class and fetch disease details
        predicted_cls = predicted_class if predicted_class is not None else max(predictions, key=predictions.get)
        disease_details = self.DISEASE_INFO.get(predicted_cls, {
            "name": "Unknown",
            "severity": "Unknown",
            "desc": "No detailed clinical profile available.",
            "guideline": "Consult specialist."
        })
        # Fetch confidence score (fallback to max probability if key is not present like 'unknown')
        confidence_val = predictions.get(predicted_cls, max(predictions.values()))

        # 2. Encode images as base64 strings
        orig_b64 = self._convert_pil_to_base64(original_image)
        heat_b64 = self._convert_pil_to_base64(heatmap_image)

        # 3. Compile date timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 4. Generate distribution bar metrics
        dist_bars = ""
        # Sort predictions in descending order of probability
        sorted_preds = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        for cls_code, prob in sorted_preds:
            cls_name = self.DISEASE_INFO.get(cls_code, {}).get("name", cls_code)
            percentage = prob * 100
            # Set colors depending on probability strength
            bar_color = "#3085d6" if cls_code != predicted_cls else "#d9534f" if cls_code == "mel" else "#f0ad4e" if cls_code == "bcc" else "#2ecc71"
            
            dist_bars += f"""
            <div class="metric-row">
                <div class="metric-label">{cls_name} ({cls_code.upper()})</div>
                <div class="metric-bar-container">
                    <div class="metric-bar" style="width: {percentage:.1f}%; background-color: {bar_color};"></div>
                </div>
                <div class="metric-value">{percentage:.2f}%</div>
            </div>
            """

        # 5. Assemble HTML structure with inline CSS styling for high-fidelity printing
        severity_class = "severity-high" if "Malignant" in disease_details["severity"] else \
                         "severity-warning" if "Undetermined" in disease_details["severity"] else \
                         "severity-benign"
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SkinCancerAI Clinical Diagnostics Summary - {patient_id}</title>
    <style>
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #f5f7fb;
            color: #333333;
            margin: 0;
            padding: 30px;
        }}
        .report-container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            padding: 40px;
        }}
        .header {{
            border-bottom: 2px solid #eef2f6;
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title h1 {{
            margin: 0;
            font-size: 24px;
            color: #1e3d59;
            font-weight: 700;
        }}
        .header-title p {{
            margin: 5px 0 0 0;
            font-size: 13px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .header-meta {{
            text-align: right;
            font-size: 13px;
            color: #7f8c8d;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 35px;
        }}
        .section-card {{
            background-color: #f8fafc;
            border-radius: 6px;
            padding: 20px;
            border: 1px solid #e2e8f0;
        }}
        .section-card h2 {{
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 16px;
            color: #1e3d59;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .info-label {{
            font-weight: 600;
            color: #4a5568;
        }}
        .info-value {{
            color: #2d3748;
        }}
        .severity-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .severity-high {{
            background-color: #fdf2f2;
            color: #9b1c1c;
            border: 1px solid #f8b4b4;
        }}
        .severity-benign {{
            background-color: #def7ec;
            color: #03543f;
            border: 1px solid #84e1bc;
        }}
        .severity-warning {{
            background-color: #fefcbf;
            color: #744210;
            border: 1px solid #faf089;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 35px;
        }}
        .image-card {{
            text-align: center;
        }}
        .image-card img {{
            max-width: 100%;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            margin-bottom: 8px;
        }}
        .image-card p {{
            margin: 0;
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
        }}
        .metric-row {{
            margin-bottom: 12px;
            font-size: 13px;
        }}
        .metric-label {{
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 4px;
        }}
        .metric-bar-container {{
            background-color: #edf2f7;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            display: inline-block;
            width: 80%;
            vertical-align: middle;
        }}
        .metric-bar {{
            height: 100%;
            border-radius: 5px;
        }}
        .metric-value {{
            display: inline-block;
            width: 18%;
            text-align: right;
            vertical-align: middle;
            font-weight: bold;
            color: #2d3748;
        }}
        .clinical-guideline {{
            margin-top: 15px;
            background-color: #fffaf0;
            border-left: 4px solid #dd6b20;
            padding: 12px 15px;
            border-radius: 0 4px 4px 0;
            font-size: 13px;
            color: #7b341e;
        }}
        .footer {{
            border-top: 1px solid #eef2f6;
            margin-top: 40px;
            padding-top: 20px;
            text-align: center;
            font-size: 11px;
            color: #94a3b8;
        }}
        .disclaimer {{
            font-style: italic;
            margin-top: 10px;
            color: #cbd5e1;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <div class="header-title">
                <h1>SkinCancerAI Clinical Diagnostics</h1>
                <p>Hybrid Multi-Modal Research System</p>
            </div>
            <div class="header-meta">
                <div><strong>Case ID:</strong> {patient_id}</div>
                <div><strong>Generated:</strong> {timestamp}</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="section-card">
                <h2>Patient Demographics</h2>
                <div class="info-row">
                    <span class="info-label">Age:</span>
                    <span class="info-value">{demographics.get('age', 'N/A')} years</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Sex:</span>
                    <span class="info-value" style="text-transform: capitalize;">{demographics.get('sex', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Anatomical Site:</span>
                    <span class="info-value" style="text-transform: capitalize;">{demographics.get('localization', 'N/A')}</span>
                </div>
            </div>

            <div class="section-card">
                <h2>Primary Diagnostic Prediction</h2>
                <div class="info-row">
                    <span class="info-label">Classification:</span>
                    <span class="info-value" style="font-weight: bold; color: #1e3d59;">{disease_details['name']}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Confidence Score:</span>
                    <span class="info-value" style="font-weight: bold; color: #d9534f;">{(confidence_val*100):.2f}%</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Severity Level:</span>
                    <span class="info-value">
                        <span class="severity-badge {severity_class}">{disease_details['severity']}</span>
                    </span>
                </div>
            </div>
        </div>

        <div class="image-gallery">
            <div class="image-card">
                <img src="data:image/png;base64,{orig_b64}" alt="Original Skin Lesion">
                <p>Original Dermoscopic Image</p>
            </div>
            <div class="image-card">
                <img src="data:image/png;base64,{heat_b64}" alt="Grad-CAM Hotspot Focus">
                <p>Gradient-weighted Class Activation Map (Grad-CAM)</p>
            </div>
        </div>

        <div class="section-card" style="margin-bottom: 35px;">
            <h2>Clinical Probability Distribution</h2>
            {dist_bars}
        </div>

        <div class="section-card">
            <h2>Pathological Description & Diagnostic Guidelines</h2>
            <p style="font-size: 14px; line-height: 1.5; color: #4a5568; margin-bottom: 10px;">
                <strong>Pathological Details:</strong> {disease_details['desc']}
            </p>
            <div class="clinical-guideline">
                <strong>Clinician Guidelines:</strong> {disease_details['guideline']}
            </div>
        </div>

        <div class="footer">
            <div>SkinCancerAI System - Developed for Scientific Publication & Diagnostic Assistance</div>
            <div class="disclaimer">
                Disclaimer: This report is compiled by an AI research model for academic study. 
                It is not an FDA-approved clinical decision tool and must be verified by a board-certified dermatologist.
            </div>
        </div>
    </div>
</body>
</html>
"""

        # Save to file
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Clinical report compiled successfully and saved to: {report_path}")
        return report_path
