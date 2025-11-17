# Page_restoration
**1. Input PDF Page Extraction**  
* Load each PDF page with PyMuPDF (fitz), rendering at user-specified DPI for resolution control.  
* Convert page pixmap bytes to numpy array and rearrange color channels for OpenCV processing.  

  
**2. Blank Page Detection**
* Convert the page image to grayscale.  
* Apply binary threshold inversion to create a mask of possible foreground pixels.  
* Calculate foreground pixel ratio vs. total pixels.  
* Skip pages where the ratio is below a threshold (default 0.5%), assuming page is blank.  

**3. Document Layout Detection (with DETR model)**
* Use Hugging Face's DETR architecture for layout detection, loaded via transformers for pretrained model weights. (cmarkea/detr-layout-detection)  
* Convert OpenCV image to PIL Image then prepare input tensor for DETR processor.  
* Obtain bounding boxes and scores for layout elements like text blocks and image blocks.  
* Filter boxes based on confidence and class (text vs. image) using user-defined confidence cutoff.  
* Separate detected layout blocks into:  
   * **Text blocks:** Likely text or captions, for later targeted processing.  
   * **Image blocks:** Pictures, tables, or diagrams preserved as-is.  

**4. Page Image Cleaning Procedure**  
* Convert the original page image to grayscale.  
* Adaptive Binarization using Sauvola thresholding:  
* Sauvola method adapts threshold locally based on mean and variance within a window.  
* Superior at preserving faint and small edges compared to global or simpler adaptive methods.  

* **Noise removal:**  
  * Identify connected components in the binary image representing black pixels.  
  * Remove isolated small blobs with pixel area less than or equal to 2 pixels as noise.  
  
* **Region Preservation:**  
  * Restore regions identified as image blocks from original colored page image to the processed output to preserve picture quality.  

5. Output Assembly and Saving  
* Convert the cleaned binary image to a 3-channel color image for PDF embedding.  
* Save cleaned page temporarily as PNG.  
* Use PyMuPDF to create a new PDF page with dimensions matching the cleaned image.  
* Embed the cleaned PNG image into PDF page.  
* Repeat for all processed pages (skipping blanks).  
* Save the entire cleaned set as a final PDF output.  

