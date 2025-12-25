import os
import sys
import time
import argparse

# ==============================================================================
# 0. CRITICAL ANTI-FREEZE CONFIGURATION (MUST BE FIRST)
# ==============================================================================
print("Initializing Environment...")

# 1. Disable MKLDNN (CRITICAL FIX FOR WINDOWS HANGS)
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

# 2. Allow multiple OpenMP libraries & Force Single Thread
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 3. Disable Paddle's parallel workers
os.environ["FLAGS_use_parallel_work"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

import shutil
import logging
import traceback
import warnings
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

# Suppress warnings
warnings.filterwarnings("ignore")

# Check Imports
try:
    from paddleocr import LayoutDetection
except ImportError:
    print("CRITICAL ERROR: paddleocr not installed. Run: pip install paddlepaddle paddleocr")
    sys.exit(1)

try:
    from skimage.filters import threshold_sauvola
except ImportError:
    print("CRITICAL ERROR: scikit-image not installed. Run: pip install scikit-image")
    sys.exit(1)

# ==============================================================================
# 1. LOGGING & CONFIGURATION
# ==============================================================================

def setup_logging():
    """
    Configures a strictly formatted logger. 
    Uses propagate=False to prevent double logging or leakage to root handlers.
    """
    # 1. Define Format (No filename, no line number)
    log_fmt = '%(asctime)s | %(levelname)s | %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_fmt, datefmt=date_fmt)

    # 2. Configure Specific Logger
    logger = logging.getLogger("PDFRestoration")
    logger.setLevel(logging.INFO)
    
    # CRITICAL: Stop logs from bubbling up to the root logger (which has the noisy handler)
    logger.propagate = False 

    # 3. Clean existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. Add clean StreamHandler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 5. Silence third-party noise
    logging.getLogger("ppocr").setLevel(logging.ERROR) 
    logging.getLogger("paddle").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    
    return logger

logger = setup_logging()

@dataclass
class PipelineConfig:
    # Paths
    input_pdf_path: str
    output_pdf_path: str
    model_paddle_dir: str
    
    # Output Directories
    temp_dir: str = "pipeline_temp"
    
    # PDF Settings
    dpi: int = 200        
    upscale_factor: float = 1.0 
    jpg_quality: int = 100    
    
    # Detection Settings 
    conf_threshold: float = 0.20
    
    # Logic Parameters
    denoise_h: float = 10.0 
    sauvola_window_normal: int = 40  
    max_dot_area: int = 2
    blank_threshold_ratio: float = 0.005
    darkness_min: int = 10
    sauvola_window_faded: int = 25

# ==============================================================================
# 2. IMAGE UTILS
# ==============================================================================

class ImageUtils:
    @staticmethod
    def get_pdf_page_as_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
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
        
        if pix.n == 4: return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3: return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def intersect(box: List[int], w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1)),
                max(0, min(int(x2), w)), max(0, min(int(y2), h)))

    @staticmethod
    def is_page_blank(image_bgr: np.ndarray, config: PipelineConfig) -> bool:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if gray.mean() <= config.darkness_min: return False
        
        try:
            win = config.sauvola_window_normal if config.sauvola_window_normal % 2 != 0 else config.sauvola_window_normal + 1
            thresh = threshold_sauvola(gray, window_size=win)
            binary = (gray <= thresh).astype(np.uint8)
        except:
            binary = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11) // 255).astype(np.uint8)
            
        density = binary.sum() / binary.size
        return density < config.blank_threshold_ratio

    @staticmethod
    def crop_black_borders(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        mask = (gray > 30).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w]
        return img

# ==============================================================================
# 3. TEXT ENHANCEMENT
# ==============================================================================

class EnhancementEngine:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def classify_page(self, gray: np.ndarray) -> Tuple[str, Dict[str, float]]:
        min_v, max_v, _, _ = cv2.minMaxLoc(gray)
        contrast = float(max_v - min_v)
        mean_i = float(gray.mean())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_px = float(gray.size)
        v_dark_sum = float(np.sum(hist[0:40])) 
        v_dark_pct = (v_dark_sum / total_px) * 100

        if lap_var < 50 or mean_i > 210:
            approach = "faded"
        else:
            approach = "normal"
            
        metrics = {
            "Contrast": contrast,
            "Mean": mean_i,
            "vDark": v_dark_pct,
            "Lap": lap_var
        }
        return approach, metrics

    def binarize_normal(self, gray: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoising(gray, None, h=self.cfg.denoise_h, templateWindowSize=7, searchWindowSize=21)
        win = self.cfg.sauvola_window_normal if self.cfg.sauvola_window_normal % 2 != 0 else self.cfg.sauvola_window_normal + 1
        thr = threshold_sauvola(denoised, window_size=win, k=0.075)
        soft_bin = denoised.copy()
        soft_bin[denoised > thr] = 255
        
        fg = (soft_bin < 180).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] <= self.cfg.max_dot_area:
                soft_bin[labels == i] = 255
                
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        eq = clahe.apply(soft_bin)
        out = eq.copy()
        out[eq < 160] = 0
        out[eq >= 200] = 255
        return out

    def binarize_faded(self, gray: np.ndarray) -> np.ndarray:
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
# 4. MAIN PIPELINE
# ==============================================================================

class PDFRestorationPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.engine = EnhancementEngine(config)
        self.model = None

    def _setup_workspace(self):
        if os.path.exists(self.cfg.temp_dir):
            shutil.rmtree(self.cfg.temp_dir)
        os.makedirs(self.cfg.temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.output_pdf_path), exist_ok=True)
        
        logger.info(f"Workspace initialized. Target Device: CPU") 

        # Validate Model Path
        if not os.path.exists(self.cfg.model_paddle_dir):
            raise FileNotFoundError(f"Paddle Model directory not found: {self.cfg.model_paddle_dir}")

        try:
            logger.info(f"Loading Paddle model from: {self.cfg.model_paddle_dir}")
            
            # Initialization
            self.model = LayoutDetection(
                model_dir=self.cfg.model_paddle_dir,
                model_name=None, 
                device="cpu"
            )
            logger.info("PaddleOCR LayoutDetection Model loaded successfully.")
        except Exception as e:
            logger.error(f"Critical Error loading Paddle model: {e}")
            raise e
        
        logger.info("="*70)

    def _parse_paddle_result(self, result):
        """Parses the result from LayoutDetection safely."""
        parsed_items = []
        
        # 1. Try Object Attributes (Standard Wrapper)
        if hasattr(result, 'boxes') and hasattr(result, 'scores') and hasattr(result, 'label_ids'):
            boxes = getattr(result, 'boxes', None)
            scores = getattr(result, 'scores', None)
            ids = getattr(result, 'label_ids', None)
            
            if boxes is not None and scores is not None and ids is not None and len(boxes) > 0:
                for box, score, label_id in zip(boxes, scores, ids):
                    try:
                        lid = int(float(label_id))
                        parsed_items.append((box, score, lid))
                    except:
                        continue
                return parsed_items

        # 2. Try Dictionary Keys
        if isinstance(result, dict):
            # Case A: Separate keys exist
            if 'boxes' in result and 'scores' in result and 'label_ids' in result:
                boxes = result['boxes']
                scores = result['scores']
                ids = result['label_ids']
                # Ensure they are lists
                if isinstance(boxes, list) and isinstance(scores, list) and isinstance(ids, list):
                    for box, score, label_id in zip(boxes, scores, ids):
                        try:
                            lid = int(float(label_id))
                            parsed_items.append((box, score, lid))
                        except:
                            continue
                    return parsed_items
            
            # Case B: Nested List of Dictionaries (Your Specific JSON Format)
            # { "boxes": [ {"cls_id": 1, "coordinate": [...], "score": ...}, ... ] }
            if 'boxes' in result:
                boxes_data = result['boxes']
                if isinstance(boxes_data, list) and len(boxes_data) > 0:
                    # Check if the contents are dicts
                    if isinstance(boxes_data[0], dict):
                        for item in boxes_data:
                            box = item.get('coordinate') or item.get('bbox')
                            score = item.get('score')
                            label_id = item.get('cls_id') or item.get('label')
                            
                            if box is not None and score is not None and label_id is not None:
                                try:
                                    lid = int(float(label_id))
                                    parsed_items.append((box, score, lid))
                                except:
                                    continue
                        return parsed_items

        return parsed_items

    def process_page(self, page_num: int) -> Dict[str, Any]:
        res = {'page_num': page_num, 'temp_path': None, 'is_blank': False, 'error': None, 'has_seal': False}
        
        try:
            # 1. RENDER PAGE
            page_bgr = ImageUtils.get_pdf_page_as_bgr(self.cfg.input_pdf_path, page_num, self.cfg.dpi)
            
            if self.cfg.upscale_factor != 1.0:
                h0, w0 = page_bgr.shape[:2]
                page_bgr = cv2.resize(page_bgr, (int(w0 * self.cfg.upscale_factor), int(h0 * self.cfg.upscale_factor)), interpolation=cv2.INTER_CUBIC)
            
            page_bgr = ImageUtils.crop_black_borders(page_bgr)
            h, w = page_bgr.shape[:2]

            # 2. BLANK CHECK
            if ImageUtils.is_page_blank(page_bgr, self.cfg):
                res['is_blank'] = True
                return res

            # 3. SAVE TEMP IMAGE
            temp_input_path = os.path.abspath(os.path.join(self.cfg.temp_dir, f"input_p{page_num}.png"))
            cv2.imwrite(temp_input_path, page_bgr)

            # 4. DETECTION
            images_to_preserve = []
            
            try:
                output = self.model.predict([temp_input_path], batch_size=1, layout_nms=True)
                res_list = list(output)
                
                if res_list:
                    result_obj = res_list[0]
                    items = self._parse_paddle_result(result_obj)
                    
                    if not items:
                        logger.info(f"Page {page_num}: No layout elements detected.")

                    for box, score, lid in items:
                        # DEBUG: Log all detections for transparency
                        if score < self.cfg.conf_threshold: 
                            continue
                        
                        coords = [int(v) for v in box]
                        
                        if lid == 1: 
                            images_to_preserve.append(coords)
                            logger.info(f"Page {page_num}: Found figure | Conf: {score:.4f}")
                        elif lid == 15: 
                            logger.info(f"Page {page_num}: Found seal | Conf: {score:.4f}")
                            res['has_seal'] = True
                        
            except Exception as e:
                logger.error(f"Detection error on page {page_num}: {e}")

            # 5. TEXT ENHANCEMENT
            gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
            approach, m = self.engine.classify_page(gray)
            
            # LOG: Process Metrics
            logger.info(f"Page {page_num}: Process: {approach.upper()} | Metrics: "
                        f"Contrast={m['Contrast']:.1f}, Mean={m['Mean']:.1f}, "
                        f"vDark={m['vDark']:.2f}%, Lap={m['Lap']:.1f}")

            enhanced_gray = self.engine.binarize_faded(gray) if approach == "faded" else self.engine.binarize_normal(gray)
            final_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

            # 6. COMPOSITE
            for box in images_to_preserve:
                x1, y1, x2, y2 = ImageUtils.intersect(box, w, h)
                final_bgr[y1:y2, x1:x2] = page_bgr[y1:y2, x1:x2]

            # 7. SAVE OUTPUT
            temp_out_path = os.path.join(self.cfg.temp_dir, f"proc_p{page_num}.jpg")
            Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)).save(
                temp_out_path, "JPEG", quality=self.cfg.jpg_quality, optimize=True
            )
            res['temp_path'] = temp_out_path
            
            if os.path.exists(temp_input_path): os.remove(temp_input_path)

        except Exception as e:
            res['error'] = str(e)
            logger.error(f"Page {page_num} Failed: {e}")
            traceback.print_exc()

        return res

    def run(self):
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
        blank_count = 0
        pages_with_seals = []
        
        for i in range(1, total_pages + 1):
            res = self.process_page(i)
            
            if res['is_blank']:
                logger.info(f"Page {i}: Blank page skipped.")
                blank_count += 1
                continue
            
            if res['error']:
                continue
            
            # Track stamps
            if res.get('has_seal', False):
                pages_with_seals.append(i)
                
            if res['temp_path'] and os.path.exists(res['temp_path']):
                with Image.open(res['temp_path']) as im:
                    pdf_page = out_doc.new_page(width=im.width, height=im.height)
                    pdf_page.insert_image(fitz.Rect(0, 0, im.width, im.height), filename=res['temp_path'])
                
                os.remove(res['temp_path'])
                processed_count += 1
                
                logger.info(f"Status: [{i}/{total_pages}] successfully integrated into output.")

        logger.info("Compressing and saving output file...")
        out_doc.save(self.cfg.output_pdf_path, deflate=True, garbage=4)
        out_doc.close()
        
        if os.path.exists(self.cfg.temp_dir): shutil.rmtree(self.cfg.temp_dir)
        
        duration = datetime.now() - start_time
        logger.info("="*70)
        logger.info(f"PROJECT SUMMARY")
        logger.info("-"*70)
        logger.info(f"Total Duration: {duration}")
        logger.info(f"Pages Restored: {processed_count}/{total_pages}")
        logger.info(f"Output Path:    {self.cfg.output_pdf_path}")
        logger.info(f"Stamp Detected on Pages: {pages_with_seals}")
        logger.info("="*70)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Restoration Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, required=True, help="Path to output PDF file")
    
    args = parser.parse_args()

    try:
        cfg = PipelineConfig(
            input_pdf_path=args.input,
            output_pdf_path=args.output,
            model_paddle_dir=r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\models\model"
        )
        
        if os.path.exists(cfg.input_pdf_path):
            pipeline = PDFRestorationPipeline(cfg)
            pipeline.run()
        else:
            print(f"CRITICAL ERROR: Input file not found: {cfg.input_pdf_path}")
            
    except Exception as e:
        print("\n\n" + "="*50)
        print("FATAL PIPELINE CRASH")
        print("="*50)
        traceback.print_exc()
        print("="*50)
        
    print("\nProcess finished.")