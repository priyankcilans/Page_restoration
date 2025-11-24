# Page_restoration
**1. Configuration (PipelineConfig) :**  
Centralizes control of input/output paths, model parameters, visual settings, cleaning and detection parameters, and export preferences.

**Key parameters:**

* input_pdf_path — Path to input PDF

* output_pdf_path — Path for cleaned output PDF

* model_doclayout_path, model_stamp_path — Paths to YOLO model files

* dpi, upscale_factor — Controls PDF rendering and page resolution

* stamp_export_dir, temp_dir — Temporary storage for intermediate/output files

**2. PDF Loading and Rendering :**
PDF is loaded with PyMuPDF (fitz). Pages are rendered to images at set DPI and optionally upscaled (e.g., 2x for higher resolution).
```
python 
page_bgr = get_pdf_page_as_bgr(pdf_path, page_num, dpi)  
# Optionally upscale using cv2.resize with INTER_CUBIC
```
**3. Image Preprocessing & Cleaning :**  
Each page undergoes several cleaning steps:

* Black border removal: Removes scan artifacts from page edges

* Soft binarization (Sauvola): Separates text from background while retaining grayscale/anti-aliased edges

* Tiny dot removal: Cleans salt-and-pepper noise using connected component analysis

* Contrast enhancement (CLAHE): Locally boosts contrast for weak/faded text

* Smart binarization: Pushes dark/gray text to pure black, blends soft edges for readability

* Sharpening: Enhances fine details with a convolution kernel

* Stamp/Figure preservation: Detected regions are pasted back in original color

**4. AI Layout and Stamp Detection :**
* Layout YOLO (doclayout_yolo): Locates figures, tables, titles, etc.

* Stamp YOLO: Detects and marks stamp regions

* Bounding boxes for figures and stamps are composited onto the cleaned image, ensuring these elements retain original color and structure.

**5. Blank Page Removal :**
* Pages detected as blank (by threshold ratio of foreground pixels) are automatically skipped.

**6. Export Workflow :**
* Each processed page is saved temporarily (PNG/JPG).

* Pages with detected stamps also have an annotated version exported for review.

* Cleaned pages are embedded as images into the output PDF using PyMuPDF (out_doc.new_page and insert_image).

**7. Saving and Cleanup :**
* The final PDF is saved with optional compression (deflate=True).

* All temporary/intermediate images are cleaned up.
