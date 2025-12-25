import os
import sys
import warnings
import logging
import multiprocessing
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict

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

# Optional Dependency Imports
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

# Configure Logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    Central configuration. Picklable for multiprocessing.
    """
    # Paths
    input_pdf_path: str
    output_pdf_path: str
    model_doclayout_path: str
    model_stamp_path: str
    
    # Execution
    num_processes: int = 2  # Adjust based on VRAM (2-4 is usually safe for GPUs)
    
    # Output Directories
    temp_dir: str = "pipeline_temp"
    stamp_export_dir: str = "stamp_exports"
    blank_export_dir: str = "blank_exports"
    
    # PDF Settings
    dpi: int = 96
    upscale_factor: float = 2.0
    jpg_quality: int = 100    
    
    # Detection Settings
    image_classes: Tuple[str, ...] = ("figure", "table", "title_image", "image", "picture", "photo")
    stamp_class: str = "stamp"
    conf_doclayout: float = 0.5
    conf_stamp: float = 0.25
    
    # Visuals
    color_figure: Tuple[int, int, int] = (0, 255, 0)
    color_stamp: Tuple[int, int, int] = (255, 128, 0)
    box_thickness: int = 3
    
    # Approaches
    denoise_h: float = 8.0 
    sauvola_window_normal: int = 40  
    max_dot_area: int = 2
    blank_threshold_ratio: float = 0.005
    darkness_min: int = 10
    min_text_chars_in_figure: int = 15
    sauvola_window_faded: int = 25

# ==============================================================================
# 2. UTILITY FUNCTIONS (Pure functions, safe for workers)
# ==============================================================================

def get_pdf_page_as_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
    # Each worker opens the file independently to avoid pickling issues
    doc = fitz.open(pdf_path)
    actual_page_idx = page_num - 1
    if actual_page_idx < 0 or actual_page_idx >= doc.page_count:
        doc.close()
        raise ValueError(f"Page {page_num} out of range")
    
    page = doc.load_page(actual_page_idx)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close() # Close immediately after read
    
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

def intersect(box: List[int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(x1), w - 1)),
        max(0, min(int(y1), h - 1)),
        max(0, min(int(x2), w)),
        max(0, min(int(y2), h))
    )

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
            ix1 = max(cx1, nx1)
            iy1 = max(cy1, ny1)
            ix2 = min(cx2, nx2)
            iy2 = min(cy2, ny2)
            
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            
            if inter_w * inter_h > 0:
                cx1 = min(cx1, nx1)
                cy1 = min(cy1, ny1)
                cx2 = max(cx2, nx2)
                cy2 = max(cy2, ny2)
                boxes.pop(i)
                was_merged = True
            else:
                i += 1
           
        if was_merged:
            boxes.insert(0, [cx1, cy1, cx2, cy2])
        else:
            merged.append([cx1, cy1, cx2, cy2])
    return merged

# def is_region_text_heavy(crop_bgr: np.ndarray, char_threshold: int = 80) -> bool:
#     """
#     Advanced text detection using Horizontal Projection Profiles (HPP).
#     Distinguishes Devanagari text from statues/figures based on 
#     rhythmic line structure and height ratios.
#     """
#     if crop_bgr.size == 0:
#         return False

#     h_img, w_img = crop_bgr.shape[:2]

#     # 1. Color Check: Statues/Photos usually have a sepia or yellowish tint.
#     # Text in these manuscripts is usually pure grayscale.
#     hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
#     avg_saturation = np.mean(hsv[:, :, 1])
#     if avg_saturation > 15:  # If it has color, keep as IMAGE
#         return False

#     # 2. Pre-processing for Structural Analysis
#     gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
#     binary = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY_INV, 25, 11
#     )

#     # 3. Horizontal Projection Profile (HPP)
#     # Sum of ink pixels per row to identify 'lines'
#     hpp = np.sum(binary, axis=1) / 255
    
#     line_heights = []
#     in_line = False
#     start_row = 0
#     row_threshold = w_img * 0.08 
    
#     for i, val in enumerate(hpp):
#         if val > row_threshold:
#             if not in_line:
#                 start_row = i
#                 in_line = True
#         else:
#             if in_line:
#                 line_heights.append(i - start_row)
#                 in_line = False

#     num_lines = len(line_heights)
    
#     # CASE A: Too few lines to be a text block (Statues are often 1-2 large blobs)
#     if num_lines < 3:
#         return False 

#     # CASE B: Massive Blob Check
#     # A text line is usually thin. If a 'line' is > 20% of the box height, it's a figure.
#     max_line_h = max(line_heights) if line_heights else 0
#     if (max_line_h / h_img) > 0.20:
#         return False

#     # CASE C: Consistency Check
#     # Text lines have similar heights. Statues have high variance.
#     if num_lines > 0:
#         std_dev = np.std(line_heights)
#         mean_h = np.mean(line_heights)
#         if std_dev > mean_h * 0.8:
#             return False

#     # 4. Edge Density
#     edges = cv2.Canny(gray, 100, 200)
#     edge_density = np.sum(edges > 0) / float(gray.size)

#     # Text usually has 3+ lines and high edge frequency.
#     return (num_lines >= 3) and (edge_density > 0.06)

def is_region_text_heavy(crop_bgr: np.ndarray, char_threshold: int = 80) -> bool:
    """
    Final optimized logic:
    1. Preserves soft historical portraits (Page 3 fix).
    2. Uses rhythmic variance to kill ghost text blocks (Page 1 & 2 fix).
    """
    if crop_bgr.size < 100: return False
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # --- STEP 1: SHARPNESS PROTECTION (Page 3 Fix) ---
    # According to your logs, Page 3 has Lap=58.9. 
    # Text is usually > 100. We stop immediately if it's soft.
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 65: 
        return False # Definitely a photo or soft figure

    # --- STEP 2: RHYTHMIC ANALYSIS (Page 1 & 2 Fix) ---
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    hpp = np.sum(binary, axis=1) / 255
    
    # Calculate the variance of the horizontal projection.
    # Text lines create high-variance spikes; images/statues are more 'flat'.
    hpp_var = np.var(hpp)
    
    peaks = 0
    in_peak = False
    current_blob = 0
    max_blob = 0
    for val in hpp:
        if val > (w * 0.1):
            if not in_peak: peaks += 1; in_peak = True
            current_blob += 1
        else:
            if in_peak: 
                max_blob = max(max_blob, current_blob)
                current_blob = 0
                in_peak = False
    max_blob = max(max_blob, current_blob)

    # --- STEP 3: FOURIER FREQUENCY ---
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    center_y, center_x = h // 2, w // 2
    v_strip = magnitude_spectrum[max(0,center_y-20):min(h,center_y+20), max(0,center_x-2):min(w,center_x+2)]
    freq_score = np.mean(v_strip) if v_strip.size > 0 else 0

    # --- DIAGNOSTIC LOG ---
    # Check these values for Page 1 and 2 specifically.
    print(f"[TRACE] Lap: {lap_var:.1f} | Peaks: {peaks} | HPP_Var: {hpp_var:.1f} | MaxBlob: {max_blob/h:.2f}")

    # --- FINAL DECISION ---
    # If the region is sharp (Lap > 65) AND has high rhythmic peaks or high HPP variance, it's text.
    if peaks >= 5 or hpp_var > 400 or freq_score > 165:
        # One last check: if it's a solid massive blob (Statue), it's not text.
        if (max_blob / h) > 0.22:
            return False
        return True

    return False

# ==============================================================================
# 3. ADAPTIVE LOGIC
# ==============================================================================

def classify_text_quality(gray: np.ndarray) -> Tuple[str, dict]:
    # 1. Global measures
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    contrast_range = max_val - min_val
    mean_intensity = float(gray.mean())

    # 2. Histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_px = float(gray.size)
    very_dark_pct = float(np.sum(hist[0:40]) / total_px)
    dark_pct      = float(np.sum(hist[0:80]) / total_px)

    # 3. Sharpness
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 4. Sauvola
    try:
        sauvola  = threshold_sauvola(gray, window_size=25, k=0.1)
        sauvola_mean = float(np.mean(sauvola))
        sauvola_std  = float(np.std(sauvola))
    except Exception:
        sauvola_mean, sauvola_std = 128.0, 30.0

    faded_score = 0

    # ---------- STRONG NORMAL: needs both ink and sharpness ----------
    if (
        dark_pct      >= 0.015 and     # >= 1.5% dark pixels
        very_dark_pct >= 0.007 and     # >= 0.7% very dark
        lap_var       >= 40.0          # reasonably sharp
    ):
        label = "normal"
    else:
        # ---------- FADED SCORE (aggressive) ----------
        if very_dark_pct <= 0.004:     # <= 0.4% very dark
            faded_score += 1
        if dark_pct <= 0.015:          # <= 1.5% dark
            faded_score += 1
        if mean_intensity >= 205:      # bright page
            faded_score += 1
        if lap_var <= 35.0:            # not sharp
            faded_score += 1
        if sauvola_mean >= 185 and sauvola_std <= 22:
            faded_score += 1

        # Any clearly weak page (>=2 signals) is faded
        label = "faded" if faded_score >= 2 else "normal"

    metrics = {
        "contrast": contrast_range,
        "mean": mean_intensity,
        "vDark_pct": very_dark_pct,
        "dark_pct": dark_pct,
        "lap_var": lap_var,
        "sauvola_mean": sauvola_mean,
        "sauvola_std": sauvola_std,
        "faded_score": faded_score,
    }
    return label, metrics

# ==============================================================================
# 4. IMAGE PROCESSING KERNELS
# ==============================================================================

def is_page_blank(image_bgr: np.ndarray, config: PipelineConfig) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() <= config.darkness_min: return False

    try:
        win_size = config.sauvola_window_normal if config.sauvola_window_normal % 2 != 0 else config.sauvola_window_normal + 1
        thresh = threshold_sauvola(gray, window_size=win_size)
        binary = (gray <= thresh).astype(np.uint8)
    except Exception:
        binary = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11) // 255).astype(np.uint8)

    fg_ratio = binary.sum() / binary.size
    return fg_ratio < config.blank_threshold_ratio

def sauvola_binarize_soft(gray: np.ndarray, window: int) -> np.ndarray:
    win_size = window if window % 2 != 0 else window + 1
    thresh = threshold_sauvola(gray, window_size=win_size, k=0.075)
    out = gray.copy()
    out[gray > thresh] = 255
    return out

def fill_text_holes(binary_img: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(binary_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(closed)

def remove_tiny_dots(binary_img: np.ndarray, max_area: int) -> np.ndarray:
    fg = (binary_img < 180).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    cleaned = binary_img.copy()
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            cleaned[labels == i] = 255
    return cleaned

def enhance_text_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    return clahe.apply(gray)

def blacken_text_core(eq_gray: np.ndarray) -> np.ndarray:
    out = eq_gray.copy()
    out[eq_gray < 160] = 0
    margin_mask = (eq_gray >= 160) & (eq_gray < 200)
    out[margin_mask] = ((eq_gray[margin_mask] - 160) * (255 / 40)).astype(np.uint8)
    out[eq_gray >= 200] = 255
    return out

def crop_black_borders(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = (gray > 30).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    return img

def binarize_for_print(gray: np.ndarray, config: PipelineConfig) -> np.ndarray:
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    eq = clahe.apply(denoised)
    win_size = config.sauvola_window_faded if config.sauvola_window_faded % 2 != 0 else config.sauvola_window_faded + 1
    thr = threshold_sauvola(eq, window_size=win_size, k=0.08)
    sauvola_mask = (eq <= thr + 3).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sauvola_mask, connectivity=8)
    clean_mask = np.zeros_like(sauvola_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            clean_mask[labels == i] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    out = np.full_like(gray, 255, dtype=np.uint8)
    out[clean_mask == 1] = 0
    return out

# ==============================================================================
# 5. MULTIPROCESSING WORKER SETUP
# ==============================================================================

# Global variable for models INSIDE worker process
worker_models = {}

def init_worker(config_pickle):
    """
    Initializer for the multiprocessing pool.
    Loads models once per process.
    """
    global worker_models
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {os.getpid()}] Initializing models on {device}...")
    
    try:
        # Load Doc Layout Model
        if YOLOv10 is not None:
            try:
                doc_model = YOLOv10(model=config_pickle.model_doclayout_path, task="layout")
            except Exception:
                doc_model = YOLO(config_pickle.model_doclayout_path)
        else:
            doc_model = YOLO(config_pickle.model_doclayout_path)
            
        # Load Stamp Model
        stamp_model = YOLO(config_pickle.model_stamp_path)
        
        worker_models['doc'] = doc_model
        worker_models['stamp'] = stamp_model
        worker_models['config'] = config_pickle
        
    except Exception as e:
        print(f"[Worker {os.getpid()}] Model Load Error: {e}")
        sys.exit(1)

def process_page_wrapper(args):
    """
    Wrapper function called by Pool.map
    """
    page_num, config = args
    return process_page_task(page_num, config)

def process_page_task(page_num: int, config: PipelineConfig) -> Dict[str, Any]:
    """
    Main logic executed by a worker process for a single page.
    Updated to report confidence scores and handle Page 3 vs. Page 1/2 conflicts.
    """
    global worker_models
    doc_model = worker_models.get('doc')
    stamp_model = worker_models.get('stamp')

    result: Dict[str, Any] = {
        'page_num': page_num,
        'temp_path': None,
        'stamp_path': None,
        'blank_path': None,
        'is_blank': False,
        'had_stamp': False,
        'approach': 'none',
        'quality_metrics': None,
        'error': None
    }

    try:
        # 1. Load Image
        page_bgr = get_pdf_page_as_bgr(config.input_pdf_path, page_num, config.dpi)

        # 2. Blank check
        if is_page_blank(page_bgr, config):
            result['is_blank'] = True
            blank_filename = f"page_{page_num}_blank.jpg"
            blank_path = os.path.join(config.blank_export_dir, blank_filename)
            cv2.imwrite(blank_path, page_bgr)
            result['blank_path'] = blank_path
            return result

        # 3. Upscale
        if config.upscale_factor != 1.0:
            h0, w0 = page_bgr.shape[:2]
            page_bgr = cv2.resize(
                page_bgr,
                (int(w0 * config.upscale_factor), int(h0 * config.upscale_factor)),
                interpolation=cv2.INTER_CUBIC,
            )

        # 4. Crop borders
        page_bgr = crop_black_borders(page_bgr)
        h, w = page_bgr.shape[:2]

        # 5. Layout Detection with Confidence Filtering
        fig_boxes: List[List[int]] = []
        # Raised to 0.70 to help ignore high-confidence ghost text on Pages 1 & 2
        LAYOUT_CONF_THRESHOLD = 0.75 

        try:
            # Use 1280 resolution to help the model distinguish text lines from solid figures
            res_doc = doc_model.predict(page_bgr, conf=0.25, imgsz=1280, verbose=False)
            
            if res_doc:
                r = res_doc[0]
                for b in r.boxes:
                    cls_idx = int(b.cls[0])
                    cls_name = r.names.get(cls_idx, "").lower()
                    conf_score = float(b.conf[0])
                    coords = [int(v) for v in b.xyxy[0].tolist()]

                    if cls_name in config.image_classes:
                        # REPORTING: Print every candidate found for debugging
                        print(f"[INFO] Page {page_num}: Detected {cls_name} at {coords} | Conf: {conf_score:.4f}")
                        
                        if conf_score >= LAYOUT_CONF_THRESHOLD:
                            fig_boxes.append(coords)
                        else:
                            print(f"[DEBUG] Page {page_num}: Low confidence {cls_name} rejected.")
        except Exception as e:
            print(f"[ERROR] Layout detection failed on Page {page_num}: {e}")

        # 6. Stamp Detection
        stamp_boxes: List[List[int]] = []
        try:
            res_stamp = stamp_model.predict(page_bgr, conf=config.conf_stamp, imgsz=896, verbose=False)[0]
            for b in res_stamp.boxes:
                cls_name = res_stamp.names[int(b.cls[0])].lower()
                if config.stamp_class.lower() in cls_name:
                    stamp_boxes.append([int(v) for v in b.xyxy[0].tolist()])
        except Exception:
            pass

        fig_boxes = merge_overlapping_boxes(fig_boxes)
        result['had_stamp'] = len(stamp_boxes) > 0

        # Overlap Filtering: Remove figures that are actually stamps
        def boxes_overlap(a, b, iou_thresh: float = 0.2) -> bool:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0: return False
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return (inter / float(area_a + area_b - inter + 1e-6)) >= iou_thresh

        filtered_figs = [f for f in fig_boxes if not any(boxes_overlap(f, s) for s in stamp_boxes)]
        fig_boxes = filtered_figs

        # 7. Adaptive processing (Normal vs. Faded)
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        approach, metrics = classify_text_quality(gray)
        result['approach'] = approach
        result['quality_metrics'] = metrics

        if approach == "faded":
            final_gray = binarize_for_print(gray, config)
        else:
            denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=config.denoise_h, templateWindowSize=7, searchWindowSize=21)
            soft_bin = sauvola_binarize_soft(denoised_gray, config.sauvola_window_normal)
            filled_bin = fill_text_holes(soft_bin)
            dots_removed = remove_tiny_dots(filled_bin, config.max_dot_area)
            eq = enhance_text_contrast(dots_removed)
            final_gray = blacken_text_core(eq)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            final_gray = cv2.filter2D(final_gray, -1, kernel)

        # 8. Composite Assembly
        cleaned_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
        composite = cleaned_bgr.copy()

        # PRESERVE FIGURES: Verify with frequency and sharpness analysis
        for box in fig_boxes:
            x1, y1, x2, y2 = intersect(box, w, h)
            if x2 <= x1 or y2 <= y1: continue

            region = page_bgr[y1:y2, x1:x2]

            # The logic that protects the Monk portrait while killing Page 1/2 text
            if is_region_text_heavy(region, char_threshold=config.min_text_chars_in_figure):
                print(f"[DEBUG] Page {page_num}: fig {box} rejected as TEXT (Ghost)")
                continue

            print(f"[DEBUG] Page {page_num}: fig {box} kept as IMAGE")
            composite[y1:y2, x1:x2] = region

        # Annotated RAW for stamp debug
        annotated_raw = page_bgr.copy()
        for box in stamp_boxes:
            x1, y1, x2, y2 = intersect(box, w, h)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(annotated_raw, (x1, y1), (x2, y2), config.color_stamp, config.box_thickness)

        # 9. Final Output
        temp_filename = f"temp_proc_{page_num}.jpg"
        temp_path = os.path.join(config.temp_dir, temp_filename)
        Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)).save(
            temp_path, "JPEG", quality=config.jpg_quality, optimize=True
        )
        result['temp_path'] = temp_path

        if result['had_stamp']:
            stamp_filename = f"page_{page_num}_stamp.jpg"
            stamp_path = os.path.join(config.stamp_export_dir, stamp_filename)
            Image.fromarray(cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB)).save(stamp_path, "JPEG", quality=100)
            result['stamp_path'] = stamp_path

    except Exception as e:
        result['error'] = str(e)
        traceback.print_exc()

    return result

# ==============================================================================
# 6. MAIN ORCHESTRATOR
# ==============================================================================

def run_multiprocess_pipeline(config: PipelineConfig):
    # 1. Setup Dirs
    for d in [config.temp_dir, config.stamp_export_dir, config.blank_export_dir, os.path.dirname(config.output_pdf_path)]:
        if d: os.makedirs(d, exist_ok=True)

    # 2. Get Page Count
    try:
        doc_pre = fitz.open(config.input_pdf_path)
        total_pages = doc_pre.page_count
        doc_pre.close()
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    print(f"Starting Multiprocess Pipeline on {total_pages} pages")
    print(f"Workers: {config.num_processes} | Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # 3. Prepare Tasks
    # Create list of (page_num, config) tuples
    tasks = [(i + 1, config) for i in range(total_pages)]

    # 4. Run Pool
    results = []
    # set_start_method('spawn') is default on Windows, but good to be aware of.
    
    with multiprocessing.Pool(processes=config.num_processes, initializer=init_worker, initargs=(config,)) as pool:
        # map_async + tqdm could be added for a progress bar, using simple map for clarity
        for i, res in enumerate(pool.imap(process_page_wrapper, tasks)):
            print(f"[{i+1}/{total_pages}] Processed Page {res['page_num']} -> {res['approach']}")
            if res['error']:
                print(f"   Error on page {res['page_num']}: {res['error']}")
                
            m = res.get("quality_metrics") or {}
            print(
                f"   Metrics: "
                f"C={m.get('contrast',0):.1f}, "
                f"mean={m.get('mean',0):.1f}, "
                f"vDark={m.get('vDark_pct',0)*100:.2f}%, "
                f"dark={m.get('dark_pct',0)*100:.2f}%, "
                f"Lap={m.get('lap_var',0):.1f}, "
                f"Smean={m.get('sauvola_mean',0):.1f}, "
                f"Sstd={m.get('sauvola_std',0):.1f}, "
                f"score={m.get('faded_score',0)}"
            )
            results.append(res)

    # 5. Reassemble PDF
    print("\nReassembling PDF...")
    results.sort(key=lambda x: x['page_num']) # Ensure order

    out_doc = fitz.open()
    
    stats = {'normal': 0, 'faded': 0, 'blank': 0, 'stamp': 0}

    for res in results:
        if res['is_blank']:
            stats['blank'] += 1
            continue
        
        if res['had_stamp']: stats['stamp'] += 1
        if res['approach'] in stats: stats[res['approach']] += 1

        if res['temp_path'] and os.path.exists(res['temp_path']):
            try:
                with Image.open(res['temp_path']) as im:
                    pdf_page = out_doc.new_page(width=im.width, height=im.height)
                    pdf_page.insert_image(fitz.Rect(0, 0, im.width, im.height), filename=res['temp_path'])
                
                # Cleanup temp file immediately after adding to PDF
                os.remove(res['temp_path'])
            except Exception as e:
                print(f"Failed to merge page {res['page_num']}: {e}")

    # 6. Save
    out_doc.save(config.output_pdf_path, deflate=True, garbage=4)
    out_doc.close()
    
    print("\n=== FINAL STATISTICS ===")
    print(f"Total Pages: {total_pages}")
    print(f"Blanks: {stats['blank']}")
    print(f"Stamps Detected: {stats['stamp']}")
    print(f"Approaches: Normal={stats['normal']}, Faded={stats['faded']}")
    
    print(f"Saved to: {config.output_pdf_path}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    
    config = PipelineConfig(
        input_pdf_path=r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\8_VISHWATMA SHRI ADINATH (C1374)_copy_2.pdf",
        output_pdf_path=r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\8_VISHWATMA SHRI ADINATH (C1374)_ADAPTIVE_MP_2.pdf",
        
        # Models
        model_doclayout_path=r"D:\Cilans\PDF-RESTORATION\experiments\doclayout_yolo_docstructbench_imgsz1280_2501.pt",
        model_stamp_path=r"D:\Cilans\PDF-RESTORATION\experiments\finetunedyolo11m_896imgsz_50epochs.pt",

        num_processes=2, 
        
        upscale_factor=2.0,
        jpg_quality=100,
        denoise_h=10.0, 
        sauvola_window_normal=40,
        min_text_chars_in_figure=80,
        sauvola_window_faded=25
    )
    
    if os.path.exists(config.input_pdf_path):
        run_multiprocess_pipeline(config)
    else:
        print(f"Error: Input file not found: {config.input_pdf_path}")