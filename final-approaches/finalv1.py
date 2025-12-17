import os
import sys
import warnings
import logging
from dataclasses import dataclass
from typing import List, Tuple, Any

# ==============================================================================
# 0. ENVIRONMENT & SECURITY PRE-CONFIGURATION
# ==============================================================================
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["YOLO_VERBOSE"] = "False"

import cv2
import fitz
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

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
  input_pdf_path: str
  output_pdf_path: str
  model_doclayout_path: str
  model_stamp_path: str

  temp_dir: str = "pipeline_temp"
  stamp_export_dir: str = "stamp_exports"

  dpi: int = 96
  upscale_factor: float = 2.0
  jpg_quality: int = 100

  image_classes: Tuple[str, ...] = ("figure", "table", "title_image")
  stamp_class: str = "stamp"
  conf_doclayout: float = 0.5
  conf_stamp: float = 0.25

  color_figure: Tuple[int, int, int] = (0, 255, 0)
  color_stamp: Tuple[int, int, int] = (255, 128, 0)
  box_thickness: int = 3

  blank_threshold_ratio: float = 0.005
  darkness_min: int = 10
  sauvola_window: int = 25
  max_dot_area: int = 1


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def get_pdf_page_as_bgr(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
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
  x1, y1, x2, y2 = box
  return (
    max(0, min(int(x1), w - 1)),
    max(0, min(int(y1), h - 1)),
    max(0, min(int(x2), w)),
    max(0, min(int(y2), h))
  )


# ==============================================================================
# 3. IMAGE PROCESSING & CLEANING
# ==============================================================================

def is_page_blank(image_bgr: np.ndarray, config: PipelineConfig) -> bool:
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


def remove_tiny_dots(binary_img: np.ndarray, max_area: int) -> np.ndarray:
  fg = (binary_img < 180).astype(np.uint8)
  num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

  cleaned = binary_img.copy()
  for i in range(1, num):
    if stats[i, cv2.CC_STAT_AREA] <= max_area:
      cleaned[labels == i] = 255
  return cleaned


def crop_black_borders(img: np.ndarray, border_thresh: int = 30) -> np.ndarray:
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  mask = (gray > border_thresh).astype(np.uint8)
  coords = cv2.findNonZero(mask)

  if coords is not None:
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]
  return img


def needs_strong_binarization(gray: np.ndarray) -> bool:
  min_v, max_v, _, _ = cv2.minMaxLoc(gray)
  return (max_v - min_v) < 80


def binarize_for_print(gray: np.ndarray, config: PipelineConfig) -> np.ndarray:
    # 0) Mild median blur to reduce grain
    denoised = cv2.medianBlur(gray, 3)

    # 1) CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    eq = clahe.apply(denoised)

    # 2) Sauvola – permissive but not crazy
    win_size = config.sauvola_window if config.sauvola_window % 2 != 0 else config.sauvola_window + 1
    thr = threshold_sauvola(eq, window_size=win_size, k=0.08)
    sauvola_mask = (eq <= thr + 3).astype(np.uint8)

    # 3) Remove very small blobs (noise), keep only plausible glyph parts
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sauvola_mask, connectivity=8)
    clean_mask = np.zeros_like(sauvola_mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 8:          # was 6; slightly stricter to drop dots
            clean_mask[labels == i] = 1

    # 4) Stroke repair + light de‑noising
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # one very gentle opening to knock off isolated pixels around strokes
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) Final binary image
    out = np.full_like(gray, 255, dtype=np.uint8)
    out[clean_mask == 1] = 0
    return out


# ==============================================================================
# 4. AI DETECTION & INFERENCE
# ==============================================================================

def load_models(config: PipelineConfig) -> Tuple[Any, Any]:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Loading models on {device}...")

  doc_model = None
  if YOLOv10 is not None:
    try:
      print("Initializing YOLOv10 from doclayout_yolo...")
      doc_model = YOLOv10(model=config.model_doclayout_path, task="layout")
    except Exception as e:
      print(f"Warning: `YOLOv10` wrapper failed: {e}. Switching to standard YOLO.")

  if doc_model is None:
    print("Initializing Standard Ultralytics YOLO...")
    doc_model = YOLO(config.model_doclayout_path)

  stamp_model = YOLO(config.model_stamp_path)
  return doc_model, stamp_model


def detect_figures_doclayout(img_bgr: np.ndarray, model: Any, config: PipelineConfig) -> List[List[int]]:
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

  page_bgr = crop_black_borders(page_bgr)
  h, w = page_bgr.shape[:2]

  figure_boxes = detect_figures_doclayout(page_bgr, doc_model, config)
  stamp_boxes = detect_stamps_yolo(page_bgr, stamp_model, config)
  had_stamp = len(stamp_boxes) > 0

  gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)

  # For now, always use print-quality binarization
  final_gray = binarize_for_print(gray, config)

  cleaned_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
  composite = cleaned_bgr.copy()

  for box in figure_boxes + stamp_boxes:
    x1, y1, x2, y2 = intersect(box, w, h)
    if x2 > x1 and y2 > y1:
      composite[y1:y2, x1:x2] = page_bgr[y1:y2, x1:x2]

  annotated_raw = page_bgr.copy()
  for box in stamp_boxes:
    x1, y1, x2, y2 = intersect(box, w, h)
    if x2 > x1 and y2 > y1:
      cv2.rectangle(annotated_raw, (x1, y1), (x2, y2), config.color_stamp, config.box_thickness)

  return annotated_raw, composite, stamp_boxes, had_stamp


def run_full_pdf_processing(config: PipelineConfig) -> List[int]:
  os.makedirs(config.temp_dir, exist_ok=True)
  os.makedirs(config.stamp_export_dir, exist_ok=True)
  os.makedirs(os.path.dirname(config.output_pdf_path), exist_ok=True)

  doc_model, stamp_model = load_models(config)
  doc = fitz.open(config.input_pdf_path)

  out_doc = fitz.open()
  total = doc.page_count
  detected_stamp_pages = []

  print(f"Starting processing: {config.input_pdf_path}")
  print(f"Upscale Factor: {config.upscale_factor}x")

  for i in range(total):
    print(f"Processing page {i+1}/{total}...")
    try:
      page_bgr = get_pdf_page_as_bgr(config.input_pdf_path, i + 1, config.dpi)
      if is_page_blank(page_bgr, config):
        continue
    except Exception as e:
      print(f"[Error] Failed to load page {i+1}: {e}")
      continue

    if config.upscale_factor != 1.0:
      h, w = page_bgr.shape[:2]
      page_bgr = cv2.resize(
        page_bgr,
        (int(w * config.upscale_factor), int(h * config.upscale_factor)),
        interpolation=cv2.INTER_CUBIC,
      )

    annotated_raw, composite, stamp_boxes, had_stamp = process_single_page_image(
      page_bgr, doc_model, stamp_model, config
    )

    if had_stamp:
      detected_stamp_pages.append(i + 1)
      print(f"  -> [INFO] Stamp detected on Page {i+1}")
      export_path = os.path.join(config.stamp_export_dir, f"page_{i+1}_stamp.jpg")
      try:
        Image.fromarray(cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB)).save(
          export_path, "JPEG", quality=80
        )
      except Exception:
        pass

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

  print("Saving PDF...")
  try:
    out_doc.save(config.output_pdf_path, deflate=True, garbage=4)
    print(f"Success! Saved to: {config.output_pdf_path}")
  except Exception as e:
    print(f"Error saving final PDF: {e}")
  finally:
    out_doc.close()
    doc.close()

  return detected_stamp_pages

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
  config = PipelineConfig(
      input_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\PDF_restoration\pdfs\2_ADINATH STROT (D5733).PDF",
      output_pdf_path = r"D:\Cilans\PDF-RESTORATION\experiments\data\output\v3\2_ADINATH STROT (D5733)_Stamp_cleaned_3.pdf",
      model_doclayout_path = r"D:\Cilans\PDF-RESTORATION\experiments\doclayout_yolo_docstructbench_imgsz1280_2501.pt",
      model_stamp_path = r"D:\Cilans\PDF-RESTORATION\experiments\finetunedyolo11m_896imgsz_50epochs.pt",
      upscale_factor = 2.0,
      jpg_quality = 100
  )

  if os.path.exists(config.input_pdf_path):
    run_full_pdf_processing(config)
  else:
    print(f"Error: Input file not found at {config.input_pdf_path}")