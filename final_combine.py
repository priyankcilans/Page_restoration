import os
import sys
import warnings
import logging
import traceback
import argparse
from datetime import datetime
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

# ==============================================================================
# 1. LOGGING & CONFIGURATION
# ==============================================================================

def setup_logging():
    """
    Sets up a standardized logging format with date and time.
    Configures console output for real-time monitoring.
    """
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
    
    # Silence verbose third-party logs
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    
    return logging.getLogger("PDFRestorationPipeline")

logger = setup_logging()

@dataclass
class PipelineConfig:
    """
    Central configuration for the PDF processing pipeline.
    Encapsulates paths, detection thresholds, and image processing constants.
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
    upscale_factor: float = 2.0
    jpg_quality: int = 100    
    
    # Detection Settings
    image_classes: Tuple[str, ...] = ("figure", "table", "title_image", "image", "picture", "photo")
    stamp_class: str = "stamp"
    conf_doclayout_min: float = 0.25
    conf_stamp: float = 0.25
    layout_conf_threshold: float = 0.75  # Threshold to distinguish layout elements
    
    # Visuals
    color_figure: Tuple[int, int, int] = (0, 255, 0)
    color_stamp: Tuple[int, int, int] = (255, 128, 0)
    box_thickness: int = 3
    
    # Logic Parameters
    denoise_h: float = 10.0 
    sauvola_window_normal: int = 40  
    max_dot_area: int = 2
    blank_threshold_ratio: float = 0.005
    darkness_min: int = 10
    min_text_chars_in_figure: int = 80
    sauvola_window_faded: int = 25

# ==============================================================================
# 2. IMAGE PROCESSING KERNELS & UTILITIES
# ==============================================================================

class ImageUtils:
    """Collection of static methods for core image processing logic."""

    @staticmethod
    def get_pdf_page_as_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
        """Loads a PDF page and renders it as a BGR OpenCV image."""
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
        doc.close()
        
        if pix.n == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def intersect(box: List[int], w: int, h: int) -> Tuple[int, int, int, int]:
        """Ensures bounding boxes are within the image frame."""
        x1, y1, x2, y2 = box
        return (max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1)),
                max(0, min(int(x2), w)), max(0, min(int(y2), h)))

    @staticmethod
    def merge_overlapping_boxes(boxes: List[List[int]]) -> List[List[int]]:
        """Consolidates overlapping rectangles into single bounding regions."""
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
                ix1, iy1, ix2, iy2 = max(cx1, nx1), max(cy1, ny1), min(cx2, nx2), min(cy2, ny2)
                if max(0, ix2 - ix1) * max(0, iy2 - iy1) > 0:
                    cx1, cy1, cx2, cy2 = min(cx1, nx1), min(cy1, ny1), max(cx2, nx2), max(cy2, ny2)
                    boxes.pop(i)
                    was_merged = True
                else: i += 1
            if was_merged: boxes.insert(0, [cx1, cy1, cx2, cy2])
            else: merged.append([cx1, cy1, cx2, cy2])
        return merged

    @staticmethod
    def is_region_text_heavy(crop_bgr: np.ndarray) -> bool:
        """
        Advanced logic to distinguish text-heavy blocks from images/figures.
        Uses Laplacian variance, Rhythmic peak analysis (HPP), and Fourier frequency.
        """
        if crop_bgr.size < 100: return False
        h, w = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Sharpness Check
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 65: return False

        # Structural analysis via Horizontal Projection Profile
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 10)
        hpp = np.sum(binary, axis=1) / 255
        hpp_var = np.var(hpp)
        
        peaks, in_peak, current_blob, max_blob = 0, False, 0, 0
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

        # Fourier transform for rhythmic frequency (Text has high horizontal frequency)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        cy, cx = h // 2, w // 2
        v_strip = magnitude[max(0,cy-20):min(h,cy+20), max(0,cx-2):min(w,cx+2)]
        freq_score = np.mean(v_strip) if v_strip.size > 0 else 0

        # Heuristic decision
        if peaks >= 5 or hpp_var > 400 or freq_score > 165:
            if (max_blob / h) <= 0.22:
                return True
        return False

    @staticmethod
    def is_page_blank(image_bgr: np.ndarray, config: PipelineConfig) -> bool:
        """Determines if the page contains significant foreground content."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if gray.mean() <= config.darkness_min: return False
        try:
            win = config.sauvola_window_normal if config.sauvola_window_normal % 2 != 0 else config.sauvola_window_normal + 1
            thresh = threshold_sauvola(gray, window_size=win)
            binary = (gray <= thresh).astype(np.uint8)
        except:
            binary = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11) // 255).astype(np.uint8)
        return (binary.sum() / binary.size) < config.blank_threshold_ratio

    @staticmethod
    def crop_black_borders(img: np.ndarray) -> np.ndarray:
        """Crops dark scanning borders by finding the bounding box of non-dark pixels."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        mask = (gray > 30).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img

# ==============================================================================
# 3. TEXT ENHANCEMENT ENGINE
# ==============================================================================

class EnhancementEngine:
    """Handles binarization, denoising, and contrast adjustment for text regions."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def classify_page(self, gray: np.ndarray) -> Tuple[str, dict]:
        """Analyzes page quality and assigns an enhancement approach."""
        min_v, max_v, _, _ = cv2.minMaxLoc(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_px = float(gray.size)
        vdark_pct = float(np.sum(hist[0:40]) / total_px)
        dark_pct = float(np.sum(hist[0:80]) / total_px)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_i = float(gray.mean())

        try:
            sauvola = threshold_sauvola(gray, window_size=25, k=0.1)
            s_mean, s_std = float(np.mean(sauvola)), float(np.std(sauvola))
        except:
            s_mean, s_std = 128.0, 30.0

        f_score = 0
        if vdark_pct <= 0.004: f_score += 1
        if dark_pct <= 0.015: f_score += 1
        if mean_i >= 205: f_score += 1
        if lap_var <= 35.0: f_score += 1
        if s_mean >= 185 and s_std <= 22: f_score += 1

        approach = "faded" if f_score >= 2 else "normal"
        metrics = {
            "contrast": float(max_v - min_v),
            "mean": mean_i,
            "vDark_pct": vdark_pct,
            "dark_pct": dark_pct,
            "lap_var": lap_var,
            "f_score": f_score
        }
        return approach, metrics

    def binarize_normal(self, gray: np.ndarray) -> np.ndarray:
        """Enhances high-quality text using Sauvola and Non-Local Means Denoising."""
        denoised = cv2.fastNlMeansDenoising(gray, None, h=self.cfg.denoise_h, templateWindowSize=7, searchWindowSize=21)
        win = self.cfg.sauvola_window_normal if self.cfg.sauvola_window_normal % 2 != 0 else self.cfg.sauvola_window_normal + 1
        thr = threshold_sauvola(denoised, window_size=win, k=0.075)
        
        soft_bin = denoised.copy()
        soft_bin[denoised > thr] = 255
        
        # Component filtering for noise
        fg = (soft_bin < 180).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] <= self.cfg.max_dot_area:
                soft_bin[labels == i] = 255
                
        # Final blackening
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        eq = clahe.apply(soft_bin)
        out = eq.copy()
        out[eq < 160] = 0
        out[eq >= 200] = 255
        
        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(out, -1, kernel)

    def binarize_faded(self, gray: np.ndarray) -> np.ndarray:
        """Aggressive binarization for faded or low-contrast text."""
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        eq = clahe.apply(denoised)
        win = self.cfg.sauvola_window_faded if self.cfg.sauvola_window_faded % 2 != 0 else self.cfg.sauvola_window_faded + 1
        thr = threshold_sauvola(eq, window_size=win, k=0.08)
        
        mask = (eq <= thr + 3).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= 8:
                clean[labels == i] = 1
                
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        out = np.full_like(gray, 255)
        out[clean == 1] = 0
        return out

# ==============================================================================
# 4. MAIN PIPELINE ARCHITECTURE
# ==============================================================================

class PDFRestorationPipeline:
    """The central orchestrator for the PDF restoration process."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.engine = EnhancementEngine(config)
        self.models = {}

    def _setup_workspace(self):
        """Initializes model instances and output directories."""
        for path in [self.cfg.temp_dir, self.cfg.stamp_export_dir, 
                     self.cfg.blank_export_dir, os.path.dirname(self.cfg.output_pdf_path)]:
            if path: os.makedirs(path, exist_ok=True)
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Workspace initialized. Target Device: {device.upper()}")
        
        try:
            # Layout Model
            if YOLOv10 is not None:
                self.models['doc'] = YOLOv10(self.cfg.model_doclayout_path)
            else:
                self.models['doc'] = YOLO(self.cfg.model_doclayout_path)
            
            # Stamp Model
            self.models['stamp'] = YOLO(self.cfg.model_stamp_path)
            logger.info("Deep learning models loaded successfully.")
        except Exception as e:
            logger.error(f"Critical Error loading models: {e}")
            sys.exit(1)
        
        logger.info("="*70)

    def process_page(self, page_num: int) -> Dict[str, Any]:
        """Executes the full pipeline for a single PDF page."""
        res = {'page_num': page_num, 'temp_path': None, 'is_blank': False, 'error': None}
        
        try:
            # 1. RENDER & PRE-PROCESS
            page_bgr = ImageUtils.get_pdf_page_as_bgr(self.cfg.input_pdf_path, page_num, self.cfg.dpi)
            
            if self.cfg.upscale_factor != 1.0:
                h0, w0 = page_bgr.shape[:2]
                page_bgr = cv2.resize(page_bgr, (int(w0 * self.cfg.upscale_factor), 
                                                int(h0 * self.cfg.upscale_factor)), 
                                     interpolation=cv2.INTER_CUBIC)
            
            page_bgr = ImageUtils.crop_black_borders(page_bgr)
            if ImageUtils.is_page_blank(page_bgr, self.cfg):
                res['is_blank'] = True
                return res

            h, w = page_bgr.shape[:2]

            # 2. DETECTION (Layout & Stamps)
            fig_boxes = []
            det_doc = self.models['doc'].predict(page_bgr, conf=self.cfg.conf_doclayout_min, imgsz=1280, verbose=False)
            if det_doc:
                for b in det_doc[0].boxes:
                    cls = det_doc[0].names.get(int(b.cls[0]), "").lower()
                    conf = float(b.conf[0])
                    coords = [int(v) for v in b.xyxy[0].tolist()]
                    
                    if cls in self.cfg.image_classes:
                        status = "REJECTED (Low Conf)"
                        if conf >= self.cfg.layout_conf_threshold:
                            region = page_bgr[coords[1]:coords[3], coords[0]:coords[2]]
                            if not ImageUtils.is_region_text_heavy(region):
                                fig_boxes.append(coords)
                                status = "KEPT"
                            else: status = "REJECTED (Text Block)"
                        
                        logger.info(f"Page {page_num}: Found {cls} | Conf: {conf:.4f} | Decision: {status}")

            stamp_boxes = []
            det_stamp = self.models['stamp'].predict(page_bgr, conf=self.cfg.conf_stamp, imgsz=896, verbose=False)
            if det_stamp:
                for b in det_stamp[0].boxes:
                    if self.cfg.stamp_class.lower() in det_stamp[0].names[int(b.cls[0])].lower():
                        stamp_boxes.append([int(v) for v in b.xyxy[0].tolist()])
            
            fig_boxes = ImageUtils.merge_overlapping_boxes(fig_boxes)

            # 3. TEXT ENHANCEMENT
            gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
            approach, m = self.engine.classify_page(gray)
            
            logger.info(f"Page {page_num}: Process: {approach.upper()} | Metrics: "
                        f"Contrast={m['contrast']:.1f}, Mean={m['mean']:.1f}, "
                        f"vDark={m['vDark_pct']*100:.2f}%, Lap={m['lap_var']:.1f}")

            enhanced_gray = self.engine.binarize_faded(gray) if approach == "faded" else self.engine.binarize_normal(gray)
            
            # 4. COMPOSITE ASSEMBLY
            # Replace enhanced text background with original figures
            final_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            for box in fig_boxes:
                x1, y1, x2, y2 = ImageUtils.intersect(box, w, h)
                final_bgr[y1:y2, x1:x2] = page_bgr[y1:y2, x1:x2]

            # 5. PERSISTENCE
            temp_path = os.path.join(self.cfg.temp_dir, f"proc_p{page_num}.jpg")
            Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)).save(
                temp_path, "JPEG", quality=self.cfg.jpg_quality, optimize=True
            )
            res['temp_path'] = temp_path

        except Exception as e:
            res['error'] = str(e)
            logger.error(f"Page {page_num}: Processing failed: {e}")
            traceback.print_exc()

        return res

    def run(self):
        """Orchestrates the sequential processing and reassembly of the PDF."""
        start_time = datetime.now()
        logger.info("="*70)
        logger.info(f"PDF RESTORATION PIPELINE")
        logger.info("-"*70)
        logger.info(f"Input: {self.cfg.input_pdf_path}")

        self._setup_workspace()
        
        doc_in = fitz.open(self.cfg.input_pdf_path)
        total_pages = doc_in.page_count
        doc_in.close()

        out_doc = fitz.open()
        processed_count = 0
        
        for i in range(1, total_pages + 1):
            res = self.process_page(i)
            
            if res['is_blank']:
                logger.info(f"Page {i}: Blank page skipped.")
                continue
                
            if res['temp_path'] and os.path.exists(res['temp_path']):
                with Image.open(res['temp_path']) as im:
                    pdf_page = out_doc.new_page(width=im.width, height=im.height)
                    pdf_page.insert_image(fitz.Rect(0, 0, im.width, im.height), filename=res['temp_path'])
                os.remove(res['temp_path'])
                processed_count += 1
            
            logger.info(f"Status: [{i}/{total_pages}] successfully integrated into output.")

        # FINAL SAVE
        logger.info("Compressing and saving output file...")
        out_doc.save(self.cfg.output_pdf_path, deflate=True, garbage=4)
        out_doc.close()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("="*70)
        logger.info(f"PROJECT SUMMARY")
        logger.info("-"*70)
        logger.info(f"Total Duration: {duration}")
        logger.info(f"Pages Restored: {processed_count}/{total_pages}")
        logger.info(f"Output Path:    {self.cfg.output_pdf_path}")
        logger.info("="*70)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Restoration Pipeline - Sequential Version")
    parser.add_argument("--input", type=str, required=True, help="Full path to the source PDF file.")
    parser.add_argument("--output", type=str, required=True, help="Full path where the restored PDF will be saved.")
    args = parser.parse_args()

    # Pre-defined model paths for the environment
    MODEL_LAYOUT = r"D:\Cilans\PDF-RESTORATION\experiments\doclayout_yolo_docstructbench_imgsz1280_2501.pt"
    MODEL_STAMP = r"D:\Cilans\PDF-RESTORATION\experiments\finetunedyolo11m_896imgsz_50epochs.pt"

    cfg = PipelineConfig(
        input_pdf_path=args.input,
        output_pdf_path=args.output,
        model_doclayout_path=MODEL_LAYOUT,
        model_stamp_path=MODEL_STAMP,
        upscale_factor=2.0,
        jpg_quality=100
    )
    
    if os.path.exists(cfg.input_pdf_path):
        pipeline = PDFRestorationPipeline(cfg)
        pipeline.run()
    else:
        logger.error(f"CRITICAL: Input file not found at {cfg.input_pdf_path}")