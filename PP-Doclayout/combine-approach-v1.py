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
import datetime
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
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    input_pdf_path: str
    output_pdf_path: str
    model_paddle_dir: str
    
    temp_dir: str = "pipeline_temp"
    dpi: int = 200
    jpg_quality: int = 95
    upscale_factor: float = 1.0
    conf_threshold: float = 0.20
    
    denoise_h: float = 10.0 
    sauvola_window_normal: int = 40  
    max_dot_area: int = 2
    blank_threshold_ratio: float = 0.005
    darkness_min: int = 10
    sauvola_window_faded: int = 25

# ==============================================================================
# IMAGE UTILS
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
        return (max(0, min(int(x1), w)), max(0, min(int(y1), h)), 
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

    @staticmethod
    def is_actually_text_contour(crop_bgr: np.ndarray) -> bool:
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
        
        if num_blobs > 20 and avg_blob_area < 800: return True
        if 3 <= num_blobs <= 30 and fill_ratio < 0.45: return True
        return False

# ==============================================================================
# CLEANING ENGINE (SIMPLIFIED FOR GRAYSCALE)
# ==============================================================================

class EnhancementEngine:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def convert_to_grayscale(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Converts the image to high-quality grayscale with mild denoising.
        No binarization or classification is performed.
        """
        # 1. Convert to Gray
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. Mild Denoising (Optional: keeps text sharp but removes some grain)
        # Using a low 'h' value preserves faint text while removing digital noise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
        
        return denoised

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class PDFRestorationPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.engine = EnhancementEngine(config)
        self.model = None

    def _load_model(self):
        logger.info(f"Loading Paddle model from: {self.cfg.model_paddle_dir}")
        path = os.path.abspath(self.cfg.model_paddle_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path does not exist: {path}")
            
        try:
            self.model = LayoutDetection(model_dir=path, model_name=None, device="cpu")
            logger.info("PaddleOCR LayoutDetection Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Paddle model: {e}")
            sys.exit(1)

    def _parse_paddle_result(self, result):
        parsed_items = []
        boxes_list, scores_list, ids_list = [], [], []

        if isinstance(result, dict) and 'boxes' in result:
             boxes_data = result['boxes']
             if isinstance(boxes_data, list) and len(boxes_data) > 0:
                 if isinstance(boxes_data[0], dict):
                     for item in boxes_data:
                         boxes_list.append(item.get('coordinate') or item.get('bbox'))
                         scores_list.append(item.get('score'))
                         ids_list.append(item.get('cls_id') or item.get('label'))
                 elif isinstance(boxes_data[0], list):
                     boxes_list = result.get('boxes')
                     scores_list = result.get('scores')
                     ids_list = result.get('label_ids')

        elif hasattr(result, 'boxes'):
             boxes_list = getattr(result, 'boxes', [])
             scores_list = getattr(result, 'scores', [])
             ids_list = getattr(result, 'label_ids', [])

        if not boxes_list: return []

        for box, score, label_id in zip(boxes_list, scores_list, ids_list):
            try:
                lid = int(float(label_id))
                parsed_items.append((box, score, lid))
            except: continue
        return parsed_items

    def run(self):
        start_time = datetime.datetime.now()
        
        if not os.path.exists(self.cfg.input_pdf_path):
            logger.error(f"Input file not found: {self.cfg.input_pdf_path}")
            return

        if os.path.exists(self.cfg.temp_dir): shutil.rmtree(self.cfg.temp_dir)
        os.makedirs(self.cfg.temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.output_pdf_path), exist_ok=True)

        logger.info("="*70)
        logger.info(f"PDF RESTORATION PIPELINE")
        logger.info("-"*70)
        logger.info(f"Input: {self.cfg.input_pdf_path}")
        logger.info(f"Workspace initialized. Target Device: CPU")

        self._load_model()
        
        try:
            doc_in = fitz.open(self.cfg.input_pdf_path)
        except Exception as e:
            logger.error(f"Could not open PDF: {e}")
            return

        total_pages = doc_in.page_count
        doc_in.close()

        logger.info("="*70)

        out_doc = fitz.open()
        processed_count = 0
        blank_count = 0
        pages_with_seals_output_idx = []
        current_output_page_idx = 0

        for i in range(total_pages):
            page_num = i + 1
            logger.info(f"Processing Page {page_num}/{total_pages}...")
            
            try:
                original_bgr = ImageUtils.get_pdf_page_as_bgr(self.cfg.input_pdf_path, page_num, self.cfg.dpi)
                
                if self.cfg.upscale_factor != 1.0:
                    h0, w0 = original_bgr.shape[:2]
                    original_bgr = cv2.resize(original_bgr, (int(w0 * self.cfg.upscale_factor), int(h0 * self.cfg.upscale_factor)), interpolation=cv2.INTER_CUBIC)
                
                original_bgr = ImageUtils.crop_black_borders(original_bgr)
                h, w = original_bgr.shape[:2]

                if ImageUtils.is_page_blank(original_bgr, self.cfg):
                    logger.info(f"Page {page_num}: Blank page skipped.")
                    blank_count += 1
                    continue
                
                current_output_page_idx += 1
                temp_input_path = os.path.abspath(os.path.join(self.cfg.temp_dir, f"input_p{page_num}.png"))
                cv2.imwrite(temp_input_path, original_bgr)

                images_to_preserve = []
                
                try:
                    output = self.model.predict([temp_input_path], batch_size=1, layout_nms=True)
                    res_list = list(output)
                    
                    if res_list:
                        items = self._parse_paddle_result(res_list[0])
                        for box, score, lid in items:
                            if score < self.cfg.conf_threshold: continue
                            coords = [int(v) for v in box]
                            
                            if lid == 1:
                                crop = original_bgr[coords[1]:coords[3], coords[0]:coords[2]]
                                if not ImageUtils.is_actually_text_contour(crop):
                                    images_to_preserve.append(coords)
                                    logger.info(f"Page {page_num}: Found figure | Conf: {score:.4f}")
                            elif lid == 15:
                                logger.info(f"Page {page_num}: Found seal | Conf: {score:.4f}")
                                if current_output_page_idx not in pages_with_seals_output_idx:
                                    pages_with_seals_output_idx.append(current_output_page_idx)

                except Exception as e:
                    logger.error(f"Detection failed on page {page_num}: {e}")

                # --- GRAYSCALE CONVERSION (NO CLASSIFICATION/BINARIZATION) ---
                # This replaces the complex engine calls with simple grayscale
                cleaned_gray = self.engine.convert_to_grayscale(original_bgr)
                final_composite = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)

                for box in images_to_preserve:
                    x1, y1, x2, y2 = ImageUtils.intersect(box, w, h)
                    if x2 > x1 and y2 > y1:
                        final_composite[y1:y2, x1:x2] = original_bgr[y1:y2, x1:x2]

                temp_out_path = os.path.join(self.cfg.temp_dir, f"proc_p{page_num}.jpg")
                Image.fromarray(cv2.cvtColor(final_composite, cv2.COLOR_BGR2RGB)).save(
                    temp_out_path, "JPEG", quality=self.cfg.jpg_quality, optimize=True
                )
                
                with Image.open(temp_out_path) as im:
                    pdf_page = out_doc.new_page(width=im.width, height=im.height)
                    pdf_page.insert_image(fitz.Rect(0, 0, im.width, im.height), filename=temp_out_path)
                
                if os.path.exists(temp_input_path): os.remove(temp_input_path)
                if os.path.exists(temp_out_path): os.remove(temp_out_path)

            except Exception as e:
                logger.error(f"Critical error processing page {page_num}: {e}")
                continue

        logger.info("Compressing and saving output file...")
        out_doc.save(self.cfg.output_pdf_path, deflate=True, garbage=4)
        out_doc.close()
        
        if os.path.exists(self.cfg.temp_dir): shutil.rmtree(self.cfg.temp_dir)
        
        duration = datetime.datetime.now() - start_time
        logger.info("="*70)
        logger.info("PROJECT SUMMARY")
        logger.info("-" * 70)
        logger.info(f"Total Duration: {duration}")
        logger.info(f"Pages Restored: {processed_count}/{total_pages}")
        logger.info(f"Output Path:    {self.cfg.output_pdf_path}")
        logger.info(f"Stamp Detected on Pages: {pages_with_seals_output_idx}")
        logger.info("="*70)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Restoration Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, required=True, help="Path to output PDF file")
    parser.add_argument("--model", type=str, default=r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\models\model", help="Path to PaddleOCR model directory")
    parser.add_argument("--dpi", type=int, default=200, help="Processing DPI")
    
    args = parser.parse_args()

    try:
        cfg = PipelineConfig(
            input_pdf_path=args.input,
            output_pdf_path=args.output,
            model_paddle_dir=args.model,
            dpi=args.dpi
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