import os
import sys
import warnings
import logging
from dataclasses import dataclass
from typing import List, Tuple, Any

# ==============================================================================
# 0. SETUP
# ==============================================================================
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["YOLO_VERBOSE"] = "False"

import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Optional Imports
try:
    from skimage.filters import threshold_sauvola
except ImportError:
    print("CRITICAL: scikit-image not installed. Run: pip install scikit-image")
    sys.exit(1)

try:
    from doclayout_yolo import YOLOv10
except ImportError:
    YOLOv10 = None

logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    # Paths
    input_pdf_path: str
    output_pdf_path: str
    model_doclayout_path: str
    model_stamp_path: str
    
    # Settings
    dpi: int = 150
    jpg_quality: int = 95
    conf_doc: float = 0.35
    conf_stamp: float = 0.25
    
    # Tuning Parameters
    dark_pixel_threshold: int = 140  
    
    # --- GHOST IMAGE TUNING ---
    max_text_contours: int = 15  
    max_avg_contour_area: int = 1500

    # Class Definitions
    keep_classes: Tuple[str, ...] = ("text", "title", "figure", "table", "list", "image", "heading")
    image_classes: Tuple[str, ...] = ("figure", "image", "picture", "photo")


# ==============================================================================
# 2. IMAGE UTILS & ADVANCED VALIDATION
# ==============================================================================

def get_pdf_page_as_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
    doc = fitz.open(pdf_path)
    if page_num < 1 or page_num > doc.page_count: raise ValueError("Page out of range")
    page = doc.load_page(page_num - 1)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4: return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3: return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

def intersect(box: List[int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, min(int(x1), w)), max(0, min(int(y1), h)), 
            max(0, min(int(x2), w)), max(0, min(int(y2), h)))

def has_true_dark_content(crop_bgr: np.ndarray, threshold: int) -> bool:
    if crop_bgr.size == 0: return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    darkest_percentile = np.percentile(gray, 1)
    return darkest_percentile < threshold

# --- GHOST IMAGE CLASSIFIER ---
def is_actually_text_contour(crop_bgr: np.ndarray, config: PipelineConfig) -> bool:
    if crop_bgr.size == 0: return False
    
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_box_area = h * w
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
    num_blobs = len(valid_contours)
    
    if num_blobs == 0: return False
    
    ink_area = sum(cv2.contourArea(c) for c in valid_contours)
    fill_ratio = ink_area / total_box_area
    avg_blob_area = ink_area / num_blobs
    
    if num_blobs > 20 and avg_blob_area < 800:
        return True
        
    if 3 <= num_blobs <= 30:
        if fill_ratio < 0.45: 
            return True
            
    return False


# ==============================================================================
# 3. ANALYSIS LOGIC
# ==============================================================================

def get_page_analysis(img: np.ndarray, model: Any, config: PipelineConfig):
    is_blank = True
    preserve_color_boxes = [] 
    h, w = img.shape[:2]
    
    try:
        results = model.predict(img, conf=config.conf_doc, imgsz=1024, verbose=False)[0]
        
        for b in results.boxes:
            cls_name = results.names[int(b.cls[0])].lower()
            box = [int(x) for x in b.xyxy[0].tolist()]
            
            is_keep_class = any(k in cls_name for k in config.keep_classes)
            is_img_class = any(k in cls_name for k in config.image_classes)
            
            if is_keep_class:
                x1, y1, x2, y2 = intersect(box, w, h)
                crop = img[y1:y2, x1:x2]
                
                if not has_true_dark_content(crop, config.dark_pixel_threshold):
                    continue
                
                is_blank = False
                
                final_is_image = is_img_class
                if is_img_class:
                    if is_actually_text_contour(crop, config):
                        final_is_image = False 

                if final_is_image:
                    preserve_color_boxes.append(box)
                    
    except Exception as e:
        print(f"Warning in analysis: {e}")
        pass
        
    return is_blank, preserve_color_boxes


# ==============================================================================
# 4. PROCESSING PIPELINE
# ==============================================================================

def load_models(config: PipelineConfig):
    print("Loading models...")
    doc_model = None
    if YOLOv10 is not None:
        try: doc_model = YOLOv10(model=config.model_doclayout_path, task="layout")
        except: pass
    if doc_model is None:
        doc_model = YOLO(config.model_doclayout_path)
    stamp_model = YOLO(config.model_stamp_path)
    return doc_model, stamp_model

def run_pipeline(config: PipelineConfig):
    os.makedirs(os.path.dirname(config.output_pdf_path), exist_ok=True)
    doc_model, stamp_model = load_models(config)
    
    try: doc = fitz.open(config.input_pdf_path)
    except Exception: return

    out_doc = fitz.open()
    total = doc.page_count
    
    print(f"\nProcessing: {config.input_pdf_path}")
    print("Mode: Grayscale Text | Color Images | Ghost Image Fix | Tracking Output Page Indices")

    # This counter tracks the page number in the NEW (Output) PDF
    current_output_page_idx = 0 
    
    removed = 0
    pages_with_stamps_output_idx = []

    for i in range(total):
        sys.stdout.write(f"\rPage {i+1}/{total}...")
        
        try:
            original_bgr = get_pdf_page_as_bgr(config.input_pdf_path, i+1, config.dpi)
            h, w = original_bgr.shape[:2]
            
            # 1. Analyze (Detects Blank & Identifies Real Images)
            is_blank, real_image_boxes = get_page_analysis(original_bgr, doc_model, config)
            
            if is_blank:
                removed += 1
                continue # Skip Page (Does NOT increment output page counter)
            
            # If we are here, we are KEEPING this page.
            # Increment the output page counter.
            current_output_page_idx += 1
            
            # 2. Check for Stamps (Log the CURRENT OUTPUT INDEX)
            try:
                stamp_results = stamp_model.predict(original_bgr, conf=config.conf_stamp, imgsz=640, verbose=False)[0]
                has_stamp = False
                for b in stamp_results.boxes:
                    if "stamp" in stamp_results.names[int(b.cls[0])].lower():
                        has_stamp = True
                        break
                
                if has_stamp:
                    # Log the index in the OUTPUT PDF (1-based)
                    pages_with_stamps_output_idx.append(current_output_page_idx)
            except Exception:
                pass
            
            # 3. Use Original Image for Base
            cleaned_bgr = original_bgr.copy()
            
            # 4. Extract Real Images (Preserve Color)
            preserved_regions = []
            for box in real_image_boxes:
                x1, y1, x2, y2 = intersect(box, w, h)
                if x2 > x1 and y2 > y1:
                    preserved_regions.append((cleaned_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)))

            # 5. Convert Background (Text/Layout) to Grayscale
            gray_layer = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
            final_composite = cv2.cvtColor(gray_layer, cv2.COLOR_GRAY2BGR)
            
            # 6. Paste Back Real Color Images
            for img_patch, (x1, y1, x2, y2) in preserved_regions:
                final_composite[y1:y2, x1:x2] = img_patch
            
            # 7. Save
            img_pil = Image.fromarray(cv2.cvtColor(final_composite, cv2.COLOR_BGR2RGB))
            import io
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG', quality=config.jpg_quality)
            pdf_page = out_doc.new_page(width=img_pil.width, height=img_pil.height)
            pdf_page.insert_image(fitz.Rect(0, 0, img_pil.width, img_pil.height), stream=img_byte_arr.getvalue())
            
        except Exception as e:
            print(f"Error on page {i}: {e}")
            continue

    out_doc.save(config.output_pdf_path, deflate=True, garbage=4)
    out_doc.close()
    
    print(f"\n\nProcessing Complete!")
    print(f"Total Pages in Output PDF: {current_output_page_idx}")
    print(f"Blank Pages Removed: {removed}")
    print("-" * 40)
    print(f"Output PDF Page Numbers containing Stamps ({len(pages_with_stamps_output_idx)} detected):")
    print(pages_with_stamps_output_idx)
    print("-" * 40)

if __name__ == "__main__":
    cfg = PipelineConfig(
        input_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\8_VISHWATMA SHRI ADINATH (C1374)_copy.pdf",
        output_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\data\PDF_restoration\pdfs\output-19-12_2.pdf",
        model_doclayout_path = r"D:\Cilans\PDF-RESTORATION\experiments\doclayout_yolo_docstructbench_imgsz1280_2501.pt",
        model_stamp_path = r"D:\Cilans\PDF-RESTORATION\experiments\finetunedyolo11m_896imgsz_50epochs.pt",
        dpi = 150
    )
    if os.path.exists(cfg.input_pdf_path):
        run_pipeline(cfg)
    else:
        print("Input file not found.")
