import os
import sys
import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
from skimage.filters import threshold_sauvola
from ultralytics import YOLO

# Try importing doclayout_yolo, handle error if not installed
try:
    from doclayout_yolo import YOLOv10
except ImportError:
    print("Warning: doclayout_yolo module not found. Ensure it is installed.")
    YOLOv10 = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    Central configuration for the PDF processing pipeline.
    """
    # Paths
    input_pdf_path: str
    output_pdf_path: str
    model_doclayout_path: str
    model_stamp_path: str
    
    # Output Directories
    temp_dir: str = "pipeline_temp"
    stamp_export_dir: str = "stamp_exports"
    
    # PDF Settings
    dpi: int = 96
    upscale_factor: float = 2.0  # 1.0 = Original, 2.0 = Double Resolution
    jpg_quality: int = 95        # 1-100 (Higher quality for production)
    
    # Detection Settings
    image_classes: Tuple[str] = ("figure", "table")
    stamp_class: str = "stamp"
    conf_doclayout: float = 0.5
    conf_stamp: float = 0.25
    
    # Visuals
    color_figure: Tuple[int, int, int] = (0, 255, 0)
    color_stamp: Tuple[int, int, int] = (255, 128, 0)
    box_thickness: int = 3
    
    # Cleaning Parameters
    blank_threshold_ratio: float = 0.005
    darkness_min: int = 10
    sauvola_window: int = 15
    max_dot_area: int = 2

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

def get_pdf_page_as_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
    """
    Renders a specific page of a PDF file into a NumPy BGR image array.
    """
    doc = fitz.open(pdf_path)
    actual_page_idx = page_num - 1
    if actual_page_idx < 0 or actual_page_idx >= doc.page_count:
        raise ValueError(f"Page {page_num} out of range (Total: {doc.page_count})")
    
    page = doc.load_page(actual_page_idx)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert buffer to numpy array
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    # Convert RGB/RGBA to BGR (OpenCV standard)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def resize_for_display(img: np.ndarray, max_height: int = 800) -> np.ndarray:
    """
    Resizes an image maintaining aspect ratio for UI display purposes.
    """
    h, w = img.shape[:2]
    if h <= max_height:
        return img
    scale = max_height / h
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def intersect(box: List[int], w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Clips bounding box coordinates to ensure they stay within image dimensions.
    """
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(x1), w - 1)),
        max(0, min(int(y1), h - 1)),
        max(0, min(int(x2), w)),
        max(0, min(int(y2), h))
    )

# ==============================================================================
# 2. IMAGE PROCESSING & CLEANING FUNCTIONS
# ==============================================================================

def is_page_blank(image_bgr: np.ndarray, config: PipelineConfig) -> bool:
    """
    Detects if a page is effectively blank (empty or just noise).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # If the page is purely black (scanning error), skip logic
    if gray.mean() <= config.darkness_min:
        return False

    try:
        thresh = threshold_sauvola(gray, window_size=config.sauvola_window)
        binary = (gray <= thresh).astype(np.uint8)
    except:
        binary = (cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 11) // 255).astype(np.uint8)

    fg_ratio = binary.sum() / binary.size
    return fg_ratio < config.blank_threshold_ratio

def sauvola_binarize_soft(gray: np.ndarray, window: int, k: float = 0.075) -> np.ndarray:
    """
    Applies soft binarization. Instead of 0/1, it keeps the background white 
    and keeps text gray/black.
    """
    thresh = threshold_sauvola(gray, window_size=window, k=k)
    out = gray.copy()
    out[gray > thresh] = 255
    return out

def remove_tiny_dots(binary_img: np.ndarray, max_area: int) -> np.ndarray:
    """
    Removes small noise specks (salt-and-pepper noise) from the image.
    """
    fg = (binary_img < 180).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    
    cleaned = binary_img.copy()
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            cleaned[labels == i] = 255
    return cleaned

def enhance_text_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to make text pop.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    return clahe.apply(gray)

def blacken_text_core(eq_gray: np.ndarray, hard_thresh=160, soften_margin=40) -> np.ndarray:
    """
    Pushes dark gray text to pure black while keeping edges soft to prevent aliasing.
    """
    out = eq_gray.copy()
    out[eq_gray < hard_thresh] = 0
    margin_mask = (eq_gray >= hard_thresh) & (eq_gray < hard_thresh + soften_margin)
    out[margin_mask] = ((eq_gray[margin_mask] - hard_thresh) * (255 / soften_margin)).astype(np.uint8)
    out[eq_gray >= hard_thresh + soften_margin] = 255
    return out

def crop_black_borders(img: np.ndarray, border_thresh: int = 30) -> np.ndarray:
    """
    Removes black scanning borders from the edges of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = (gray > border_thresh).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    return img

# ==============================================================================
# 3. AI DETECTION FUNCTIONS
# ==============================================================================

def load_models(config: PipelineConfig) -> Tuple[Any, Any]:
    """
    Loads YOLO models. Returns (doclayout_model, stamp_model).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    
    if YOLOv10 is None:
        raise ImportError("DocLayout-YOLO library missing.")
        
    doc_model = YOLOv10(model=config.model_doclayout_path, task="layout")
    stamp_model = YOLO(config.model_stamp_path)
    
    return doc_model, stamp_model

def detect_figures_doclayout(img_bgr: np.ndarray, model: Any, config: PipelineConfig) -> List[List[int]]:
    """
    Detects Layout elements (tables, figures). Returns list of [x1, y1, x2, y2].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.predict(img_bgr, conf=config.conf_doclayout, imgsz=1024, device=device, verbose=False)
    boxes = []
    if len(results) > 0:
        r = results[0]
        for b in r.boxes:
            cls_name = r.names.get(int(b.cls), "").lower()
            if cls_name in config.image_classes:
                boxes.append([int(v) for v in b.xyxy[0].tolist()])
    return boxes

def detect_stamps_yolo(img_bgr: np.ndarray, model: Any, config: PipelineConfig) -> List[List[int]]:
    """
    Detects Stamps. Returns list of [x1, y1, x2, y2].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.predict(img_bgr, conf=config.conf_stamp, imgsz=640, device=device, verbose=False)[0]
    boxes = []
    for b in results.boxes:
        cls_name = results.names[int(b.cls[0])].lower()
        if cls_name == config.stamp_class.lower():
            boxes.append([int(v) for v in b.xyxy[0].tolist()])
    return boxes

# ==============================================================================
# 4. CORE PIPELINE LOGIC
# ==============================================================================

def process_single_page_image(
    page_bgr: np.ndarray, 
    doc_model: Any, 
    stamp_model: Any, 
    config: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray, List[List[int]], bool]:
    """
    Runs the full cleaning pipeline on a single image.
    """
    # 1. Remove borders
    page_bgr = crop_black_borders(page_bgr)
    h, w = page_bgr.shape[:2]
    
    # 2. Detect
    figure_boxes = detect_figures_doclayout(page_bgr, doc_model, config)
    stamp_boxes = detect_stamps_yolo(page_bgr, stamp_model, config)
    had_stamp = len(stamp_boxes) > 0
    
    # 3. Clean Text
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    soft_bin = sauvola_binarize_soft(gray, config.sauvola_window)
    dots_removed = remove_tiny_dots(soft_bin, config.max_dot_area)
    eq = enhance_text_contrast(dots_removed)
    final_gray = blacken_text_core(eq)
    
    # 4. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final_gray = cv2.filter2D(final_gray, -1, kernel)
    
    # 5. Composite (Paste original color figures/stamps back onto clean B&W background)
    cleaned_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
    composite = cleaned_bgr.copy()
    
    for box in figure_boxes + stamp_boxes:
        x1, y1, x2, y2 = intersect(box, w, h)
        composite[y1:y2, x1:x2] = page_bgr[y1:y2, x1:x2]
        
    # 6. Annotation (For debugging/stamps)
    annotated_raw = page_bgr.copy()
    for box in stamp_boxes:
        x1, y1, x2, y2 = intersect(box, w, h)
        cv2.rectangle(annotated_raw, (x1, y1), (x2, y2), config.color_stamp, config.box_thickness)
        
    return annotated_raw, composite, stamp_boxes, had_stamp

def run_full_pdf_processing(config: PipelineConfig) -> List[int]:
    """
    Main orchestration function.
    Returns: List of page numbers (1-based) where stamps were detected.
    """
    # 1. Setup
    os.makedirs(config.temp_dir, exist_ok=True)
    os.makedirs(config.stamp_export_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.output_pdf_path), exist_ok=True)
    
    doc_model, stamp_model = load_models(config)
    
    doc = fitz.open(config.input_pdf_path)
    out_doc = fitz.open()
    total = doc.page_count
    
    detected_stamp_pages = []  # List to track pages with stamps
    
    print(f"Starting processing: {config.input_pdf_path}")
    print(f"Upscale Factor: {config.upscale_factor}x")
    
    # 2. Process Pages
    for i in range(total):
        print(f"Processing page {i+1}/{total}...")
        
        # Load Page
        try:
            page_bgr = get_pdf_page_as_bgr(config.input_pdf_path, i+1, config.dpi)
        except Exception as e:
            print(f"[Error] Failed to load page {i+1}: {e}")
            continue
            
        # Check Blank
        if is_page_blank(page_bgr, config):
            print(f"  -> Skipped (Blank)")
            continue
            
        # Upscale
        if config.upscale_factor != 1.0:
            h, w = page_bgr.shape[:2]
            page_bgr = cv2.resize(page_bgr, (int(w * config.upscale_factor), int(h * config.upscale_factor)), interpolation=cv2.INTER_CUBIC)
            
        # Run Pipeline
        annotated_raw, composite, stamp_boxes, had_stamp = process_single_page_image(
            page_bgr, doc_model, stamp_model, config
        )
        
        # --- Handle Stamps ---
        if had_stamp:
            detected_stamp_pages.append(i + 1)
            print(f"  -> [INFO] Stamp detected on Page {i+1}")
            
            export_path = os.path.join(config.stamp_export_dir, f"page_{i+1}_stamp.jpg")
            Image.fromarray(cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB)).save(
                export_path, "JPEG", quality=80
            )
            
        # Save Composite Temp (JPG Optimized)
        tmp_path = os.path.join(config.temp_dir, f"temp_page_{i}.jpg")
        Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)).save(
            tmp_path, "JPEG", quality=config.jpg_quality, optimize=True
        )
        
        # Insert into Output PDF
        with Image.open(tmp_path) as im:
            pdf_page = out_doc.new_page(width=im.width, height=im.height)
            pdf_page.insert_image(fitz.Rect(0, 0, im.width, im.height), filename=tmp_path)
            
        # Cleanup
        try:
            os.remove(tmp_path)
        except:
            pass

    # 3. Save Final PDF (Deflated)
    print("Saving PDF...")
    out_doc.save(config.output_pdf_path, deflate=True, garbage=4)
    out_doc.close()
    
    # 4. Final Report
    print("\n" + "="*40)
    print("PROCESSING SUMMARY")
    print("="*40)
    print(f"Output File: {config.output_pdf_path}")
    print(f"Total Pages Processed: {total}")
    print(f"Stamps Detected Count: {len(detected_stamp_pages)}")
    if detected_stamp_pages:
        print(f"Pages with Stamps: {detected_stamp_pages}")
    print("="*40)
    
    return detected_stamp_pages

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Define your configuration here
    config = PipelineConfig(
        input_pdf_path = r"INPUT_PDF_PATH",
        output_pdf_path = r"OUTPUT_PDF_PATH",
        model_doclayout_path = r"doclayout_yolo_docstructbench_imgsz1024.pt",
        model_stamp_path = r"finetunedyolo11m_896imgsz_50epochs.pt",
        upscale_factor = 2.0,  
        jpg_quality = 95       
    )
    
    # Run
    if os.path.exists(config.input_pdf_path):
        run_full_pdf_processing(config)
    else:
        print(f"Error: Input file not found at {config.input_pdf_path}")
