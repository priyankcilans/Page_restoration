import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import os
import easyocr
import torch  # Required by EasyOCR

# --- 1. Configuration Constants ---

# !!! SET THESE PATHS !!!
INPUT_PDF = "data/SAMPLE_1.pdf"
OUTPUT_PDF = "data/output/v1/SAMPLE_1_restored.pdf"
BLANK_PAGES_FOLDER = "data/output/blank_pages"

# Classifier thresholds
BLANK_PAGE_STD_DEV_THRESHOLD = 15.0  # Std dev below this is likely blank
SUBSTANTIAL_CONTENT_THRESHOLD = 0.005 # Ink-to-page ratio to be "CONTENT"

# --- 2. Helper Function: PDF to Image ---

def pdf_to_image(doc, page_num, dpi=300):
    """Fetches and renders a SINGLE page from a PDF doc as a NumPy array."""
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        
        if pix.alpha:
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        else:
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img_bgr = img_data
            
        return img_bgr
        
    except Exception as e:
        print(f"    - ERROR loading page {page_num + 1}: {e}")
        return None

# --- 3. Helper Function: The V11 "Final" Cleaner ---

def clean_page_v10_easyocr(image_bgr, reader):
    """
    V10 (Fixed): EasyOCR + Smart Filtering.
    - Keeps small text blocks (titles/page numbers).
    """
    
    results = reader.readtext(
        image_bgr, 
        low_text=0.4 # Use the sensitive setting
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    if not results:
        final_clean_page = np.full(image_bgr.shape, 255, dtype=np.uint8)
        dummy_inv = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        return gray, gray, dummy_inv, final_clean_page
    
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl_simple = (int(min(tl[0], bl[0])), int(min(tl[1], tr[1])))
        br_simple = (int(max(tr[0], br[0])), int(max(bl[1], br[1])))
        cv2.rectangle(mask, tl_simple, br_simple, 255, -1)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 15))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    main_rois = []
    min_area = 50 # <-- FIX: Keeps small text
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            main_rois.append((x, y, w, h))

    final_clean_page = np.full(image_bgr.shape, 255, dtype=np.uint8)
    
    for x, y, w, h in main_rois:
        y_start, y_end = max(0, y-1), min(image_bgr.shape[0], y+h+1)
        x_start, x_end = max(0, x-1), min(image_bgr.shape[1], x+w+1)
        
        noisy_roi = image_bgr[y_start:y_end, x_start:x_end]
        if noisy_roi.size == 0: continue
            
        noisy_roi_gray = cv2.cvtColor(noisy_roi, cv2.COLOR_BGR2GRAY)
        _, cleaned_roi = cv2.threshold(noisy_roi_gray, 0, 255, 
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        cleaned_roi_bgr = cv2.cvtColor(cleaned_roi, cv2.COLOR_GRAY2BGR)
        final_clean_page[y_start:y_end, x_start:x_end] = cleaned_roi_bgr

    dummy_inv = cv2.bitwise_not(cv2.cvtColor(final_clean_page, cv2.COLOR_BGR2GRAY))
    return gray, gray, dummy_inv, final_clean_page

# --- 4. Helper Function: Page Classifier (Updated for V10) ---
def classify_page(image_bgr, std_threshold, content_threshold, reader):
    """
    Robust blank-page classifier (V10 - Fixed).
    """
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    mean_val = np.mean(gray)
    
    if std_dev < std_threshold and mean_val > 220:
        cleaned_page = np.full(image_bgr.shape, 255, dtype=np.uint8)
        return "BLANK", std_dev, 0.0, cleaned_page

    _, _, _, cleaned_page = clean_page_v10_easyocr(image_bgr, reader)

    cleaned_gray = cv2.cvtColor(cleaned_page, cv2.COLOR_BGR2GRAY)
    total_pixels = cleaned_gray.size
    black_pixels = np.sum(cleaned_gray == 0)
    black_pixel_ratio = black_pixels / total_pixels

    if black_pixel_ratio < 0.0001: 
        classification = "BLANK"
    elif black_pixel_ratio < content_threshold:
        classification = "METADATA"
    else:
        classification = "CONTENT"

    return classification, std_dev, black_pixel_ratio, cleaned_page

# --- 4. Helper Function: Page Classifier (Updated for V11) ---

def classify_page(image_bgr, std_threshold, content_threshold, reader):
    """
    Robust blank-page classifier (V11).
    - Uses std dev for a quick "BLANK" check.
    - Uses V11 cleaning to get an accurate ink ratio (titles are preserved).
    - Returns the classification AND the cleaned image.
    """
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    mean_val = np.mean(gray)
    
    if std_dev < std_threshold and mean_val > 220:
        cleaned_page = np.full(image_bgr.shape, 255, dtype=np.uint8)
        return "BLANK", std_dev, 0.0, cleaned_page

    # Call the v11 function that *keeps* titles
    _, _, _, cleaned_page = clean_page_v10_easyocr(image_bgr, reader)

    cleaned_gray = cv2.cvtColor(cleaned_page, cv2.COLOR_BGR2GRAY)
    total_pixels = cleaned_gray.size
    black_pixels = np.sum(cleaned_gray == 0)
    black_pixel_ratio = black_pixels / total_pixels

    # This logic correctly classifies pages with *only* a title
    if black_pixel_ratio < 0.0001: 
        classification = "BLANK"
    elif black_pixel_ratio < content_threshold:
        classification = "METADATA" # A title page will now fall here
    else:
        classification = "CONTENT"

    return classification, std_dev, black_pixel_ratio, cleaned_page

# --- 5. Main Processing Script ---

def main():
    print(f"Starting restoration process for: {INPUT_PDF}")

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    os.makedirs(BLANK_PAGES_FOLDER, exist_ok=True)
    print(f"Output PDF will be saved to: {OUTPUT_PDF}")
    print(f"Rejected pages will be saved to: {BLANK_PAGES_FOLDER}")

    print("Loading EasyOCR model... (This may take a moment)")
    try:
        reader = easyocr.Reader(['en'], gpu=True) 
        if 'cuda' in str(reader.device):
            print(f"--- GPU IS ACTIVE. EasyOCR model loaded on: {reader.device} ---")
        else:
            print(f"--- WARNING: GPU NOT FOUND. EasyOCR model loaded on CPU ({reader.device}). ---")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load EasyOCR model. {e}")
        exit()

    content_pil_images = []
    rejected_page_data = []

    doc = None
    try:
        doc = fitz.open(INPUT_PDF)
        print(f"Processing {len(doc)} pages...")

        for page_num in range(len(doc)):
            print(f"\n--- Processing Page {page_num + 1} of {len(doc)} ---")
            image_bgr = pdf_to_image(doc, page_num)
            
            if image_bgr is None:
                continue
                
            classification, std_dev, black_ratio, cleaned_array = classify_page(
                image_bgr,
                BLANK_PAGE_STD_DEV_THRESHOLD,
                SUBSTANTIAL_CONTENT_THRESHOLD,
                reader
            )
            
            print(f"  - Page {page_num + 1:03d}: {classification} (Std: {std_dev:.2f}, Ink: {black_ratio:.5f})")
            
            # This logic keeps BOTH "CONTENT" and "METADATA" pages
            if classification != "BLANK":
                print(f"  - Adding to final PDF.")
                pil_img = Image.fromarray(cv2.cvtColor(cleaned_array, cv2.COLOR_BGR2RGB))
                content_pil_images.append(pil_img)
            else:
                print(f"  - Skipping (BLANK).")
                rejected_page_data.append((page_num, image_bgr, classification))

        doc.close()

        # 6. Save the final content PDF
        if not content_pil_images:
            print("\nNo content pages were found. Output PDF will not be created.")
        else:
            print(f"\nSaving {len(content_pil_images)} content pages to {OUTPUT_PDF}...")
            content_pil_images[0].save(
                OUTPUT_PDF,
                save_all=True,
                append_images=content_pil_images[1:],
                resolution=300.0
            )
            print("Restored PDF saved successfully.")

        # 7. Save the rejected images
        if not rejected_page_data:
            print("No rejected (BLANK) pages to save.")
        else:
            print(f"\nSaving {len(rejected_page_data)} rejected pages to {BLANK_PAGES_FOLDER}...")
            for page_num, image_bgr, classification in rejected_page_data:
                filename = f"page_{page_num + 1:04d}_{classification}.png"
                output_path = os.path.join(BLANK_PAGES_FOLDER, filename)
                cv2.imwrite(output_path, image_bgr)
            print(f"Rejected images saved successfully.")

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        if doc:
            doc.close()

    print("\n--- Process Finished ---")

# This makes the script runnable
if __name__ == "__main__":
    main()