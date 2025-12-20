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
    
    # --- NEW TUNING PARAMETERS ---
    # 1. Blank Page Fix: Darkest pixel threshold
    # If the darkest 1% of pixels are brighter than this (0=Black, 255=White), it's blank/bleed-through.
    dark_pixel_threshold: int = 140  
    
    # 2. Text-As-Image Fix: Contour counting
    # If a detected "Image" has more than this many small blobs, it's actually Text.
    max_text_contours: int = 25      
    
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

# --- NEW VALIDATOR: Checks for "Dark Ink" ---
def has_true_dark_content(crop_bgr: np.ndarray, threshold: int) -> bool:
    """
    Returns True if the region contains pixels dark enough to be real ink/photo shadows.
    Filters out bleed-through which is usually light gray.
    """
    if crop_bgr.size == 0: return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Get the intensity of the darkest 1% of pixels
    # If this value is high (e.g., 180), the "darkest" thing in the box is light gray.
    darkest_percentile = np.percentile(gray, 1)
    
    return darkest_percentile < threshold

# --- NEW VALIDATOR: Checks if "Image" is actually Text ---
def is_actually_text_block(crop_bgr: np.ndarray, min_contours: int) -> bool:
    """
    Returns True if the region looks like a block of text (many small blobs),
    even if the AI detected it as an image.
    """
    if crop_bgr.size == 0: return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Binarize (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Count Contours (Blobs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter tiny noise
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
    
    # If we have 25+ distinct letters/shapes, it's text, not a single photo.
    return len(valid_contours) > min_contours


# ==============================================================================
# 3. ANALYSIS LOGIC
# ==============================================================================

def get_page_analysis(img: np.ndarray, model: Any, config: PipelineConfig):
    """
    Returns:
    1. is_blank (bool): True if no REAL content found.
    2. image_boxes (list): Validated images to preserve in color.
    """
    is_blank = True
    image_boxes = []
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
                
                # --- FIX 1: BLANK PAGE / BLEED THROUGH ---
                # Check if the content is "dark enough" to be real.
                if not has_true_dark_content(crop, config.dark_pixel_threshold):
                    # It's just faint bleed-through or paper texture. Ignore it.
                    continue
                
                # If we pass the dark check, the page is NOT blank.
                is_blank = False
                
                # --- FIX 2: TEXT MISCLASSIFIED AS IMAGE ---
                # Only add to "Preserve Color" list if it is an Image class AND NOT text.
                if is_img_class:
                    if is_actually_text_block(crop, config.max_text_contours):
                        # It detected an image, but it's full of letters.
                        # Treat as TEXT -> Do not preserve color -> Convert to Grayscale.
                        pass 
                    else:
                        # It's a real photo/diagram. Keep color.
                        image_boxes.append(box)
                        
    except Exception as e:
        print(f"Warning in analysis: {e}")
        pass
        
    return is_blank, image_boxes


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

def process_stamps(img: np.ndarray, model: Any, config: PipelineConfig) -> np.ndarray:
    # Detect stamps -> Clean them -> Paste back (still in color/BGR)
    h, w = img.shape[:2]
    try: results = model.predict(img, conf=config.conf_stamp, imgsz=640, verbose=False)[0]
    except: return img
    
    processed_img = img.copy()
    
    # Helper to clean stamp crop
    def clean_crop(c):
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        w_size = config.sauvola_window | 1
        thresh = threshold_sauvola(gray, window_size=w_size)
        binarized = gray.copy()
        binarized[gray > thresh] = 255
        # Quick noise removal
        kernel = np.ones((2,2), np.uint8)
        return cv2.cvtColor(cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel), cv2.COLOR_GRAY2BGR)

    for b in results.boxes:
        if "stamp" in results.names[int(b.cls[0])].lower():
            box = [int(x) for x in b.xyxy[0].tolist()]
            x1, y1, x2, y2 = intersect(box, w, h)
            if x2 > x1 and y2 > y1:
                roi = processed_img[y1:y2, x1:x2]
                processed_img[y1:y2, x1:x2] = clean_crop(roi)
    return processed_img

def run_pipeline(config: PipelineConfig):
    os.makedirs(os.path.dirname(config.output_pdf_path), exist_ok=True)
    doc_model, stamp_model = load_models(config)
    
    try: doc = fitz.open(config.input_pdf_path)
    except Exception: return

    out_doc = fitz.open()
    total = doc.page_count
    
    print(f"\nProcessing: {config.input_pdf_path}")
    print("Applying Fixes: Dark Ink Check (Blank Pages) & Contour Check (Text-as-Image)")

    kept = 0
    removed = 0

    for i in range(total):
        sys.stdout.write(f"\rPage {i+1}/{total}...")
        
        try:
            original_bgr = get_pdf_page_as_bgr(config.input_pdf_path, i+1, config.dpi)
            h, w = original_bgr.shape[:2]
            
            # 1. Analyze
            is_blank, image_boxes = get_page_analysis(original_bgr, doc_model, config)
            
            if is_blank:
                removed += 1
                continue # Skip Page
            
            kept += 1
            
            # 2. Clean Stamps
            cleaned_bgr = process_stamps(original_bgr, stamp_model, config)
            
            # 3. Extract Real Images (Color Preservation)
            preserved_regions = []
            for box in image_boxes:
                x1, y1, x2, y2 = intersect(box, w, h)
                if x2 > x1 and y2 > y1:
                    preserved_regions.append((cleaned_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)))

            # 4. Convert Everything to Grayscale
            gray_layer = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
            final_composite = cv2.cvtColor(gray_layer, cv2.COLOR_GRAY2BGR)
            
            # 5. Paste Back Color Images
            for img_patch, (x1, y1, x2, y2) in preserved_regions:
                final_composite[y1:y2, x1:x2] = img_patch
            
            # 6. Save
            img_pil = Image.fromarray(cv2.cvtColor(final_composite, cv2.COLOR_BGR2RGB))
            import io
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG', quality=config.jpg_quality)
            pdf_page = out_doc.new_page(width=img_pil.width, height=img_pil.height)
            pdf_page.insert_image(fitz.Rect(0, 0, img_pil.width, img_pil.height), stream=img_byte_arr.getvalue())
            
        except Exception:
            continue

    out_doc.save(config.output_pdf_path, deflate=True, garbage=4)
    out_doc.close()
    print(f"\n\nDone! Kept {kept} pages. Removed {removed} blank/noise pages.")

if __name__ == "__main__":
    cfg = PipelineConfig(
        input_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\5_RISHABHDEV EK PARISHILIN (C0851).PDF.pdf",
        output_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\data\PDF_restoration\pdfs\output-19-12_1.pdf",
        model_doclayout_path = r"D:\Cilans\PDF-RESTORATION\experiments\doclayout_yolo_docstructbench_imgsz1280_2501.pt",
        model_stamp_path = r"D:\Cilans\PDF-RESTORATION\experiments\finetunedyolo11m_896imgsz_50epochs.pt",
        dpi = 150
    )
    if os.path.exists(cfg.input_pdf_path):
        run_pipeline(cfg)
    else:
        print("Input file not found.")