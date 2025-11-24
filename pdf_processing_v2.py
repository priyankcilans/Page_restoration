import fitz
import cv2
import numpy as np
import os
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from typing import List, Tuple
from PIL import Image
from skimage.filters import threshold_sauvola


# --- Configuration ---
INPUT_PDF_PATH = 'D:\\Cilans\\data\\data\\SAMPLE_1.pdf'
OUTPUT_PDF_PATH = 'D:\\Cilans\\data\\experiments\\data\\output\\v0\\SAMPLE_1_CLEANED_detr.pdf' 
DPI = 300

MODEL_NAME = "cmarkea/detr-layout-detection"
CONTENT_CONFIDENCE_THRESHOLD = 0.001 

TEXT_CLASSES = [
    'Text', 'Title', 'List-item', 'Caption', 'Page-header', 'Page-footer', 
    'Section-header', 'Footnote', 'Formula'
]
IMAGE_CLASSES = ['Picture', 'Table', 'Diagram']

BACKGROUND = (255, 255, 255)

# Load DETR model and processor
try:
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetrForObjectDetection.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"DETR Layout Model loaded successfully on {device}.")
except Exception as e:
    print(f"FATAL: Could not load DETR model. Error: {e}")
    exit()

# Utility functions

def get_pdf_page_image(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
    try:
        doc = fitz.open(pdf_path)
        if page_num < 0 or page_num >= doc.page_count:
            return None
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img_array.shape[2] == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    except:
        return None

def is_page_blank(image_bgr, threshold_ratio=0.005):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    foreground_ratio = np.count_nonzero(bw) / bw.size
    return foreground_ratio < threshold_ratio

def detect_layout_blocks(image_bgr, model, processor, confidence_threshold):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    inputs = processor(images=image_pil, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]]).to(model.device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]

    text_boxes = []
    image_boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        try:
            if score.item() < confidence_threshold:
                continue
            box_coords = [round(i.item()) for i in box.tolist()]
            class_label = model.config.id2label[label.item()]
            if class_label.lower() in [c.lower() for c in TEXT_CLASSES]:
                text_boxes.append(box_coords)
            elif class_label.lower() in [c.lower() for c in IMAGE_CLASSES]:
                image_boxes.append(box_coords)
        except:
            continue
    return text_boxes, image_boxes

def sauvola_binarize(image_gray):
    window_size = 15
    thresh = threshold_sauvola(image_gray, window_size=window_size)
    binary = (image_gray > thresh).astype(np.uint8) * 255
    return binary

def remove_tiny_dots(binary_img, max_dot_area=2):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_img == 0).astype(np.uint8), connectivity=8)
    cleaned = binary_img.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_dot_area:
            cleaned[labels == i] = 255
    return cleaned

def clean_page_advanced(original_image_bgr: np.ndarray, image_blocks: list) -> np.ndarray:
    gray = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2GRAY)

    # You can insert or replace this with your denoising model or method
    denoised = gray  # currently no denoising applied

    binarized = sauvola_binarize(denoised)
    cleaned = remove_tiny_dots(binarized, max_dot_area=2)

    h, w = cleaned.shape
    final_output_bgr = np.full((h, w, 3), BACKGROUND, dtype=np.uint8)
    black_pixels = cleaned == 0
    final_output_bgr[black_pixels] = [0, 0, 0]

    h_orig, w_orig, _ = original_image_bgr.shape
    for x_min, y_min, x_max, y_max in image_blocks:
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w_orig, x_max), min(h_orig, y_max)
        roi = original_image_bgr[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            final_output_bgr[y_min:y_max, x_min:x_max] = roi

    return final_output_bgr

def process_pdf_to_clean_pdf(input_pdf_path: str, output_pdf_path: str, dpi: int):
    doc = fitz.open(input_pdf_path)
    num_pages = doc.page_count
    print(f"Total pages to process: {num_pages}")

    new_pdf = fitz.open()

    for i in range(num_pages):
        print(f"\nProcessing page {i+1}/{num_pages}...")

        page_img_bgr = get_pdf_page_image(input_pdf_path, i, dpi)
        if page_img_bgr is None:
            print(f"Error loading page {i+1}, skipping.")
            continue

        if is_page_blank(page_img_bgr):
            print(f"Skipping blank page {i+1}")
            continue

        text_blocks, image_blocks = detect_layout_blocks(
            page_img_bgr, model, processor, confidence_threshold=CONTENT_CONFIDENCE_THRESHOLD)

        cleaned_page = clean_page_advanced(page_img_bgr, image_blocks)
        
        cleaned_rgb = cv2.cvtColor(cleaned_page, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cleaned_rgb)
        img_temp_path = f'temp_cleaned_page_{i}.png'
        pil_img.save(img_temp_path)

        with Image.open(img_temp_path) as img_pil:
            width, height = img_pil.size
            new_page = new_pdf.new_page(width=width, height=height)
            rect = fitz.Rect(0, 0, width, height)
            new_page.insert_image(rect, filename=img_temp_path)
        os.remove(img_temp_path)

    new_pdf.save(output_pdf_path)
    new_pdf.close()
    print(f"\n✅ Cleaned PDF saved at: {output_pdf_path}")

if __name__ == "__main__":
    process_pdf_to_clean_pdf(INPUT_PDF_PATH, OUTPUT_PDF_PATH, DPI)
