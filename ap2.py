import os
import sys
import warnings
import logging
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

# ==============================================================================
# 0. ENVIRONMENT & SECURITY PRE-CONFIGURATION
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

# Optional Dependency Imports with Fallbacks
try:
    from skimage.filters import threshold_sauvola
except ImportError:
    print("CRITICAL: scikit-image not installed. Run: pip install scikit-image")
    sys.exit(1)

try:
    from doclayout_yolo import YOLOv10
except ImportError:
    YOLOv10 = None

try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

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
    
    # Output Directories
    temp_dir: str = "pipeline_temp"
    stamp_export_dir: str = "stamp_exports"
    blank_export_dir: str = "blank_exports"
    
    # PDF Settings
    dpi: int = 96
    upscale_factor: float = 2.0 
    jpg_quality: int = 100    
    
    # Detection Settings
    image_classes: Tuple[str, ...] = ("figure", "image", "picture", "photo")
    stamp_class: str = "stamp"
    conf_doclayout: float = 0.5
    conf_stamp: float = 0.25
    
    # --- CLEANING PARAMETERS ---
    denoise_h: float = 10.0 
    sauvola_window: int = 40  
    max_dot_area: int = 2
    blank_threshold_ratio: float = 0.005
    darkness_min: int = 10
    
    # --- GHOST IMAGE TUNING ---
    min_text_chars_in_figure: int = 15
    max_text_contours: int = 15
    max_avg_contour_area: int = 1500
    min_fill_ratio_for_text: float = 0.45


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def get_pdf_page_as_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
    doc = fitz.open(pdf_path)
    actual_page_idx = page_num - 1
    if actual_page_idx < 0 or actual_page_idx >= doc.page_count:
        raise ValueError(f"Page {page_num} out of range")
    
    page = doc.load_page(actual_page_idx)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4: return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3: return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

def get_box_area(box: List[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def get_intersection_area(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1: return 0.0
    return (x2 - x1) * (y2 - y1)

def intersect(box: List[int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, min(int(x1), w-1)), max(0, min(int(y1), h-1)), 
            max(0, min(int(x2), w)), max(0, min(int(y2), h)))

def merge_overlapping_boxes(boxes: List[List[int]]) -> List[List[int]]:
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x[0])
    merged = []
    while boxes:
        current = boxes.pop(0)
        cx1, cy1, cx2, cy2 = current
        was_merged = False
        i = 0
        while i < len(boxes):
            nx1, ny1, nx2, ny2 = boxes[i]
            # Check intersection
            if (min(cx2, nx2) - max(cx1, nx1) > 0) and (min(cy2, ny2) - max(cy1, ny1) > 0):
                cx1 = min(cx1, nx1); cy1 = min(cy1, ny1)
                cx2 = max(cx2, nx2); cy2 = max(cy2, ny2)
                boxes.pop(i)
                was_merged = True
            else:
                i += 1
        if was_merged: boxes.insert(0, [cx1, cy1, cx2, cy2])
        else: merged.append([cx1, cy1, cx2, cy2])
    return merged


# ==============================================================================
# 3. GHOST IMAGE VALIDATION
# ==============================================================================

def is_actually_text_contour(crop_bgr: np.ndarray, config: PipelineConfig) -> bool:
    if crop_bgr.size == 0: return False
    
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area = h * w
    
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except: return False
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
    num_blobs = len(valid_contours)
    
    if num_blobs == 0: return False
    
    ink_area = sum(cv2.contourArea(c) for c in valid_contours)
    fill_ratio = ink_area / total_area
    avg_blob_area = ink_area / num_blobs
    
    # Logic: Text has many small blobs OR is sparse (low fill ratio)
    if num_blobs > 20 and avg_blob_area < 800:
        return True
    
    if 3 <= num_blobs <= 30:
        if fill_ratio < config.min_fill_ratio_for_text: 
            return True
            
    return False


# ==============================================================================
# 4. IMAGE CLEANING UTILS
# ==============================================================================

def is_page_blank(image_bgr: np.ndarray, config: PipelineConfig) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() <= config.darkness_min: return False
    try:
        win = config.sauvola_window | 1
        thresh = threshold_sauvola(gray, window_size=win)
        binary = (gray <= thresh).astype(np.uint8)
    except: return False
    return (binary.sum() / binary.size) < config.blank_threshold_ratio

def clean_page_background(page_bgr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Standard cleaning: Denoise -> Binarize -> Fill Holes."""
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    
    # A. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=config.denoise_h, templateWindowSize=7, searchWindowSize=21)
    
    # B. Sauvola Binarize
    win = config.sauvola_window | 1
    thresh = threshold_sauvola(denoised, window_size=win, k=0.1)
    binary = np.where(denoised > thresh, 255, 0).astype(np.uint8)
    
    # C. Fill Holes
    inverted = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.bitwise_not(closed)
    
    # D. Remove tiny dots
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(cleaned), connectivity=8)
    final_clean = cleaned.copy()
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= config.max_dot_area:
            final_clean[labels == i] = 255 
            
    return cv2.cvtColor(final_clean, cv2.COLOR_GRAY2BGR)


# ==============================================================================
# 5. CORE PIPELINE
# ==============================================================================

def load_models(config: PipelineConfig) -> Tuple[Any, Any]:
    doc_model = None
    if YOLOv10 is not None:
        try: doc_model = YOLOv10(model=config.model_doclayout_path, task="layout")
        except: pass
    if doc_model is None: doc_model = YOLO(config.model_doclayout_path)
    stamp_model = YOLO(config.model_stamp_path)
    return doc_model, stamp_model

def process_single_page(
    page_bgr: np.ndarray, 
    doc_model: Any, 
    stamp_model: Any, 
    config: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray, List[List[int]], bool]:
    
    h, w = page_bgr.shape[:2]
    
    # 1. Detect Figures (Layout)
    raw_figs = []
    try:
        res = doc_model.predict(page_bgr, conf=config.conf_doclayout, imgsz=1024, verbose=False)[0]
        for b in res.boxes:
            cname = res.names[int(b.cls[0])].lower()
            if any(x in cname for x in config.image_classes):
                raw_figs.append([int(x) for x in b.xyxy[0].tolist()])
    except: pass
    
    # 2. Detect Stamps (For logging primarily)
    raw_stamps = []
    try:
        res = stamp_model.predict(page_bgr, conf=config.conf_stamp, imgsz=640, verbose=False)[0]
        for b in res.boxes:
            if config.stamp_class in res.names[int(b.cls[0])].lower():
                raw_stamps.append([int(x) for x in b.xyxy[0].tolist()])
    except: pass

    # --- 3. CONFLICT RESOLUTION: IMAGES vs STAMPS ---
    # The layout model often mistakes stamps for 'Figures/Images'.
    # If a 'Figure' overlaps significantly with a detected 'Stamp', 
    # we assume it IS a Stamp (and should be processed as text/binarized), NOT a Figure (preserved in color).
    
    filtered_figs = []
    
    for f_box in raw_figs:
        is_actually_stamp = False
        f_area = get_box_area(f_box)
        
        for s_box in raw_stamps:
            inter = get_intersection_area(f_box, s_box)
            
            # If > 50% of the detected "Figure" is covered by a Stamp...
            # OR > 50% of the Stamp is inside the "Figure" (and sizes are roughly similar)...
            # Then this "Figure" is just the stamp detected twice.
            
            if f_area > 0:
                overlap_ratio = inter / f_area
                if overlap_ratio > 0.5:
                    is_actually_stamp = True
                    break
        
        if not is_actually_stamp:
            filtered_figs.append(f_box)

    # --- 4. GHOST IMAGE VALIDATION ---
    # Now check the remaining figures: Are they real photos or just big text?
    valid_figures = []
    for box in filtered_figs:
        x1, y1, x2, y2 = intersect(box, w, h)
        crop = page_bgr[y1:y2, x1:x2]
        
        # If ghost check passes (returns False -> "It's an image"), we keep it.
        if not is_actually_text_contour(crop, config):
            valid_figures.append(box)
            
    # --- 5. DEFINE PRESERVED REGIONS ---
    # We ONLY preserve Valid Figures.
    # Stamps are intentionally EXCLUDED here so they get binarized with the page.
    preserved_regions = []
    for box in valid_figures:
        preserved_regions.append(box)
    
    # --- 6. CLEAN PAGE ---
    cleaned_bg = clean_page_background(page_bgr, config)
    
    # --- 7. PASTE PRESERVED REGIONS ---
    final_composite = cleaned_bg.copy()
    for box in preserved_regions:
        x1, y1, x2, y2 = intersect(box, w, h)
        if x2 > x1 and y2 > y1:
            final_composite[y1:y2, x1:x2] = page_bgr[y1:y2, x1:x2]
            
    # Debug Annotation
    annotated = page_bgr.copy()
    for box in raw_stamps:
        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0,0,255), 3)
        
    return annotated, final_composite, raw_stamps, (len(raw_stamps) > 0)


def run_full_pipeline(config: PipelineConfig):
    for d in [config.temp_dir, config.stamp_export_dir, config.blank_export_dir, os.path.dirname(config.output_pdf_path)]:
        if d: os.makedirs(d, exist_ok=True)
        
    try:
        doc_model, stamp_model = load_models(config)
        doc = fitz.open(config.input_pdf_path)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    out_doc = fitz.open()
    total = doc.page_count
    
    current_output_idx = 0
    stamp_pages_indices = []
    
    print(f"Processing: {config.input_pdf_path}")
    print("Mode: Binarized Text & Stamps | Preserved Images | Overlap Conflict Fix")
    
    for i in range(total):
        sys.stdout.write(f"\rPage {i+1}/{total}...")
        
        try:
            page_bgr = get_pdf_page_as_bgr(config.input_pdf_path, i+1, config.dpi)
            
            if config.upscale_factor > 1.0:
                h, w = page_bgr.shape[:2]
                page_bgr = cv2.resize(page_bgr, (int(w*config.upscale_factor), int(h*config.upscale_factor)), cv2.INTER_CUBIC)
            
            if is_page_blank(page_bgr, config):
                continue
            
            current_output_idx += 1
            
            annotated, composite, stamp_boxes, had_stamp = process_single_page(
                page_bgr, doc_model, stamp_model, config
            )
            
            if had_stamp:
                stamp_pages_indices.append(current_output_idx)
                try:
                    Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(
                        os.path.join(config.stamp_export_dir, f"page_{i+1}_stamp.jpg"))
                except: pass

            tmp = os.path.join(config.temp_dir, f"temp_{i}.jpg")
            Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)).save(tmp, quality=config.jpg_quality)
            
            with Image.open(tmp) as im:
                pdf_page = out_doc.new_page(width=im.width, height=im.height)
                pdf_page.insert_image(fitz.Rect(0, 0, im.width, im.height), filename=tmp)
            
            if os.path.exists(tmp): os.remove(tmp)
            
        except Exception as e:
            print(f"Error page {i+1}: {e}")
            continue
            
    out_doc.save(config.output_pdf_path, deflate=True, garbage=4)
    out_doc.close()
    
    print(f"\n\nDone. Output: {config.output_pdf_path}")
    print(f"Total Pages: {current_output_idx}")
    print(f"Pages with Stamps (Output Indices): {stamp_pages_indices}")

if __name__ == "__main__":
    cfg = PipelineConfig(
        input_pdf_path=r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\8_VISHWATMA SHRI ADINATH (C1374)_copy.pdf",
        output_pdf_path=r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\8_VISHWATMA SHRI ADINATH (C1374)_FINAL_LOGGED_OVERLAP_FIX.pdf",
        model_doclayout_path=r"D:\Cilans\PDF-RESTORATION\experiments\doclayout_yolo_docstructbench_imgsz1280_2501.pt",
        model_stamp_path=r"D:\Cilans\PDF-RESTORATION\experiments\finetunedyolo11m_896imgsz_50epochs.pt",
        upscale_factor=2.0,
        denoise_h=8.0
    )
    
    if os.path.exists(cfg.input_pdf_path):
        run_full_pipeline(cfg)
    else:
        print("File not found.")
