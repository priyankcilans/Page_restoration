import os
import sys
import warnings
import logging
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

# ==============================================================================
# 0. ENVIRONMENT & SECURITY PRE-CONFIGURATION
# ==============================================================================
# Fix for PyTorch 2.6+ security breaking YOLO loading.
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
    YOLOv10 = None  # Fallback handled in load_models

try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

# Configure Logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. CONFIGURATION
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
    blank_export_dir: str = "blank_exports"
    
    # PDF Settings
    dpi: int = 96
    upscale_factor: float = 1.0  # 1.0 = Original, 2.0 = Double Resolution
    jpg_quality: int = 100       # 1-100 (Higher quality for production)
    
    # Detection Settings
    image_classes: Tuple[str, ...] = ("figure", "table", "title_image")
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
    
    # Increased Window Size
    sauvola_window: int = 35  
    max_dot_area: int = 2
    
    # Text Check Threshold
    # If a detected 'figure' has more than this many letters, treat it as text.
    min_text_chars_in_figure: int = 15


# ==============================================================================
# 2. UTILITY FUNCTIONS
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
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


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

def merge_overlapping_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """
    Merges overlapping or nested boxes into a single larger box.
    """
    if not boxes:
        return []

    # Sort boxes by left coordinate (x1)
    boxes = sorted(boxes, key=lambda x: x[0])
    
    merged = []
    while boxes:
        current = boxes.pop(0)
        cx1, cy1, cx2, cy2 = current
        
        was_merged = False
        
        i = 0
        while i < len(boxes):
            nx1, ny1, nx2, ny2 = boxes[i]
            
            # Calculate Intersection
            ix1 = max(cx1, nx1)
            iy1 = max(cy1, ny1)
            ix2 = min(cx2, nx2)
            iy2 = min(cy2, ny2)
            
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            
            # If there is ANY overlap, merge them (Union operation)
            if inter_area > 0:
                cx1 = min(cx1, nx1)
                cy1 = min(cy1, ny1)
                cx2 = max(cx2, nx2)
                cy2 = max(cy2, ny2)
                
                boxes.pop(i) # Remove the merged box
                was_merged = True
            else:
                i += 1
        
        # If merged, re-insert to check against others again; else finalize it
        if was_merged:
            boxes.insert(0, [cx1, cy1, cx2, cy2])
        else:
            merged.append([cx1, cy1, cx2, cy2])
            
    return merged


def is_region_text_heavy(crop_bgr: np.ndarray, char_threshold: int = 15) -> bool:
    """
    Checks if a crop contains significant text content.
    Returns True if it looks like text (lots of small connected components).
    """
    if crop_bgr.size == 0:
        return False
        
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Otsu Binarization to separate ink from paper
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        return False
    
    # Find connected components (potential characters)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_chars = 0
    h_img, w_img = crop_bgr.shape[:2]
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter 1: Ignore tiny noise (dots)
        if w < 3 or h < 3:
            continue
            
        # Filter 2: Ignore huge boxes (likely frames or image borders)
        if w > w_img * 0.9 or h > h_img * 0.9:
            continue
            
        # Filter 3: Aspect ratio check (letters are usually 0.2 to 5.0)
        aspect = w / float(h)
        if 0.1 < aspect < 10:
            valid_chars += 1
            
    return valid_chars > char_threshold


# ==============================================================================
# 3. IMAGE PROCESSING & CLEANING
# ==============================================================================

def is_page_blank(image_bgr: np.ndarray, config: PipelineConfig) -> bool:
    """
    Detects if a page is effectively blank (empty or just noise).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    if gray.mean() <= config.darkness_min:
        return False

    try:
        win_size = config.sauvola_window if config.sauvola_window % 2 != 0 else config.sauvola_window + 1
        thresh = threshold_sauvola(gray, window_size=win_size)
        binary = (gray <= thresh).astype(np.uint8)
    except Exception:
        binary = (cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 11) // 255).astype(np.uint8)

    fg_ratio = binary.sum() / binary.size
    return fg_ratio < config.blank_threshold_ratio


def sauvola_binarize_soft(gray: np.ndarray, window: int, k: float = 0.075) -> np.ndarray:
    """
    Applies adaptive Sauvola thresholding while keeping the background white.
    """
    win_size = window if window % 2 != 0 else window + 1
    thresh = threshold_sauvola(gray, window_size=win_size, k=k)
    out = gray.copy()
    out[gray > thresh] = 255
    return out


def fill_text_holes(binary_img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Fills small white holes inside black text characters.
    """
    # Invert image (make text white, background black)
    inverted = cv2.bitwise_not(binary_img)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Apply Morphological Closing (Dilation -> Erosion)
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    
    # Invert back to normal
    return cv2.bitwise_not(closed)


def remove_tiny_dots(binary_img: np.ndarray, max_area: int) -> np.ndarray:
    """
    Removes salt-and-pepper noise/speckles from the image.
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
    Enhances local contrast to make faint text legible.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    return clahe.apply(gray)


def blacken_text_core(eq_gray: np.ndarray, hard_thresh=160, soften_margin=40) -> np.ndarray:
    """
    Darkens the text core while smoothing edges for a natural look.
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
# 4. AI DETECTION & INFERENCE
# ==============================================================================

def load_models(config: PipelineConfig) -> Tuple[Any, Any]:
    """
    Loads the Document Layout and Stamp Detection models with fallback logic.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    
    doc_model = None

    # Logic 1: Try Wrapper
    if YOLOv10 is not None:
        try:
            print("Initializing YOLOv10 from doclayout_yolo...")
            doc_model = YOLOv10(model=config.model_doclayout_path, task="layout")
        except Exception as e:
            print(f"Warning: `YOLOv10` wrapper failed: {e}. Switching to standard YOLO.")

    # Logic 2: Fallback to Ultralytics
    if doc_model is None:
        print("Initializing Standard Ultralytics YOLO...")
        try:
            doc_model = YOLO(config.model_doclayout_path)
        except Exception as e:
            print(f"CRITICAL ERROR loading doclayout model: {e}")
            raise e

    try:
        stamp_model = YOLO(config.model_stamp_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading stamp model: {e}")
        if "weights_only" in str(e):
            print("HINT: PyTorch security blocked this. Ensure TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is set.")
        raise e

    return doc_model, stamp_model


def detect_figures_doclayout(img_bgr: np.ndarray, model: Any, config: PipelineConfig) -> List[List[int]]:
    """
    Detects layout elements (figures, tables) that should be preserved from cleaning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    boxes: List[List[int]] = []

    try:
        results = model.predict(img_bgr, conf=config.conf_doclayout, imgsz=1024, device=device, verbose=False)
    except Exception as e:
        print(f"Warning: layout model prediction failed: {e}")
        return boxes

    if len(results) > 0:
        r = results[0]
        for b in r.boxes:
            try:
                cls_idx = int(b.cls[0]) if hasattr(b.cls, '__getitem__') else int(b.cls)
                cls_name = r.names.get(cls_idx, "").lower()
                if cls_name in config.image_classes:
                    coords = b.xyxy[0].tolist()
                    boxes.append([int(v) for v in coords])
            except Exception:
                continue
    return boxes


def detect_stamps_yolo(img_bgr: np.ndarray, model: Any, config: PipelineConfig) -> List[List[int]]:
    """
    Detects stamps/signatures to be highlighted or preserved.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        results = model.predict(img_bgr, conf=config.conf_stamp, imgsz=640, device=device, verbose=False)[0]
    except Exception as e:
        print(f"Warning: Stamp detection failed: {e}")
        return []

    boxes = []
    for b in results.boxes:
        try:
            cls_idx = int(b.cls[0]) if hasattr(b.cls, '__getitem__') else int(b.cls)
            cls_name = results.names[cls_idx].lower()
            if config.stamp_class.lower() in cls_name: 
                boxes.append([int(v) for v in b.xyxy[0].tolist()])
        except Exception:
            continue
    return boxes


# ==============================================================================
# 5. CORE PIPELINE LOGIC
# ==============================================================================

def process_single_page_image(
    page_bgr: np.ndarray, 
    doc_model: Any, 
    stamp_model: Any, 
    config: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray, List[List[int]], bool]:
    """
    Orchestrates the cleaning and detection for a single image/page.
    """
    # 1. Remove borders
    page_bgr = crop_black_borders(page_bgr)
    h, w = page_bgr.shape[:2]
    
    # 2. Detect
    figure_boxes = detect_figures_doclayout(page_bgr, doc_model, config)
    stamp_boxes = detect_stamps_yolo(page_bgr, stamp_model, config)
    
    # Merge overlapping boxes to prevent cutting (only for figures)
    figure_boxes = merge_overlapping_boxes(figure_boxes)
    
    had_stamp = len(stamp_boxes) > 0
    
    # 3. Clean Text
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step A: Binarize (With increased window size)
    soft_bin = sauvola_binarize_soft(gray, config.sauvola_window)
    
    # Step B: Fill holes inside text BEFORE removing dots
    filled_bin = fill_text_holes(soft_bin, kernel_size=3)
    
    # Step C: Remove noise
    dots_removed = remove_tiny_dots(filled_bin, config.max_dot_area)
    
    eq = enhance_text_contrast(dots_removed)
    final_gray = blacken_text_core(eq)
    
    # 4. Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final_gray = cv2.filter2D(final_gray, -1, kernel)
    
    # 5. Composite (Paste original images back over cleaned background)
    cleaned_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
    composite = cleaned_bgr.copy()
    
    # [CHANGE] STAMP PRESERVATION REMOVED
    # We still detected them above to return 'had_stamp' and 'stamp_boxes'
    # but we DO NOT paste the original raw pixels back onto 'composite'.
    # This results in the stamp being cleaned/thresholded/erased depending on the binarization.

    # B. PASTE FIGURES (Only if they are NOT text)
    # This prevents the "dirty paragraph" bug where text is detected as a figure.
    for box in figure_boxes:
        x1, y1, x2, y2 = intersect(box, w, h)
        if x2 > x1 and y2 > y1:
            # Crop the identified region
            region_crop = page_bgr[y1:y2, x1:x2]
            
            # CHECK: Does this region contain text?
            if is_region_text_heavy(region_crop, config.min_text_chars_in_figure):
                # It contains text -> Treat it as part of the page (cleaned)
                # We SKIP pasting the raw image here.
                # (Optional debug: print("Skipping paste for text-heavy figure"))
                pass
            else:
                # It is a real image/chart -> Paste the raw original
                composite[y1:y2, x1:x2] = region_crop
        
    # 6. Annotation (Draw boxes for debugging/visuals)
    # We still output the debug image in the stamp_exports folder, but NOT in the PDF
    annotated_raw = page_bgr.copy()
    for box in stamp_boxes:
        x1, y1, x2, y2 = intersect(box, w, h)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(annotated_raw, (x1, y1), (x2, y2), config.color_stamp, config.box_thickness)
        
    return annotated_raw, composite, stamp_boxes, had_stamp


def run_full_pdf_processing(config: PipelineConfig) -> Tuple[List[int], List[int]]:
    """
    Main driver. Returns: (detected_stamp_pages, blank_pages)
    """
    # 1. Setup
    os.makedirs(config.temp_dir, exist_ok=True)
    os.makedirs(config.stamp_export_dir, exist_ok=True)
    os.makedirs(config.blank_export_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.output_pdf_path), exist_ok=True)
    
    try:
        doc_model, stamp_model = load_models(config)
        doc = fitz.open(config.input_pdf_path)
    except Exception as e:
        print(f"Setup Aborted: {e}")
        return [], []

    out_doc = fitz.open()
    total = doc.page_count
    detected_stamp_pages = [] 
    blank_pages = [] 
    
    print(f"Starting processing: {config.input_pdf_path}")
    print(f"Upscale Factor: {config.upscale_factor}x")
    
    # 2. Process Pages
    for i in range(total):
        print(f"Processing page {i+1}/{total}...")
        
        # Load & Pre-checks
        try:
            page_bgr = get_pdf_page_as_bgr(config.input_pdf_path, i+1, config.dpi)
            
            # Handle Blank Pages (Export & Skip)
            if is_page_blank(page_bgr, config):
                print(f" -> Page {i+1} is blank. Exporting and Skipping.")
                blank_pages.append(i + 1)
                
                # Export image of blank page
                export_path = os.path.join(config.blank_export_dir, f"page_{i+1}_blank.jpg")
                try:
                    Image.fromarray(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)).save(export_path, "JPEG", quality=80)
                except: pass
                
                continue
        except Exception as e:
            print(f"[Error] Failed to load page {i+1}: {e}")
            continue
            
        # Upscale
        if config.upscale_factor != 1.0:
            h, w = page_bgr.shape[:2]
            page_bgr = cv2.resize(page_bgr, (int(w * config.upscale_factor), int(h * config.upscale_factor)), interpolation=cv2.INTER_CUBIC)
            
        # Run Pipeline
        annotated_raw, composite, stamp_boxes, had_stamp = process_single_page_image(
            page_bgr, doc_model, stamp_model, config
        )
        
        # Handle Stamps (Export visuals if found)
        if had_stamp:
            detected_stamp_pages.append(i + 1)
            print(f"   -> [INFO] Stamp detected on Page {i+1} (Not preserved in PDF)")
            export_path = os.path.join(config.stamp_export_dir, f"page_{i+1}_stamp.jpg")
            try:
                Image.fromarray(cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB)).save(export_path, "JPEG", quality=100)
            except Exception:
                pass
            
        # Save Composite Page to Output PDF
        tmp_path = os.path.join(config.temp_dir, f"temp_page_{i}.jpg")
        try:
            Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)).save(
                tmp_path, "JPEG", quality=config.jpg_quality, optimize=True
            )
            with Image.open(tmp_path) as im:
                pdf_page = out_doc.new_page(width=im.width, height=im.height)
                pdf_page.insert_image(fitz.Rect(0, 0, im.width, im.height), filename=tmp_path)
        except Exception as e:
            print(f"Error saving/inserting page {i+1}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 3. Save Final PDF
    print("Saving PDF...")
    try:
        out_doc.save(config.output_pdf_path, deflate=True, garbage=4)
        print(f"Success! Saved to: {config.output_pdf_path}")
    except Exception as e:
        print(f"Error saving final PDF: {e}")
    finally:
        out_doc.close()
        doc.close()
    
    # 4. FINAL REPORT
    if detected_stamp_pages:
        print("\n" + "="*40)
        print(f"SUMMARY: Stamps detected on {len(detected_stamp_pages)} pages.")
        print(f"Page Indices: {detected_stamp_pages}")
        print("="*40 + "\n")
    else:
        print("\nSUMMARY: No stamps detected.")
    
    return detected_stamp_pages, blank_pages


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    config = PipelineConfig(
      input_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\2_ADINATH STROT (D5733).PDF",
      output_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\data\output\v3\2_ADINATH STROT (D5733)_Stamp_cleaned.pdf",
      model_doclayout_path = r"D:\Cilans\PDF-RESTORATION\experiments\doclayout_yolo_docstructbench_imgsz1280_2501.pt",
      model_stamp_path = r"D:\Cilans\PDF-RESTORATION\experiments\finetunedyolo11m_896imgsz_50epochs.pt",
      upscale_factor = 2.0,
      jpg_quality = 100,
      # Adjust this if your figures contain very sparse text labels
      min_text_chars_in_figure = 15
    )
    
    if os.path.exists(config.input_pdf_path):
        stamps, blanks = run_full_pdf_processing(config)
        # Extra print at the very end to ensure it's the last thing seen
        if stamps:
            print(f"FINAL LIST OF STAMP PAGES: {stamps}")
    else:
        print(f"Error: Input file not found at {config.input_pdf_path}")