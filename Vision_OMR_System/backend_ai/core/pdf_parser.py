"""
pdf_parser.py
=============
Utility to parse uploaded PDF documents and extract pages as OpenCV image matrices.
"""

import io
import fitz  # PyMuPDF
import cv2
import numpy as np

def extract_pages_from_pdf(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 50) -> list[np.ndarray]:
    """
    Parse a PDF document in memory using PyMuPDF (fitz).
    Returns a list of OpenCV BGR matrices, one per page.
    
    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file bytes.
    dpi : int
        The DPI at which to render the PDF pages. 200 is generally good for OMR.
    max_pages : int
        Security limit to prevent massive PDFs from crashing the server.
        
    Returns
    -------
    list[np.ndarray]
        List of BGR image matrices.
    """
    images = []
    
    # Open the PDF directly from memory
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = min(len(doc), max_pages)
        
        for i in range(page_count):
            page = doc[i]
            
            # Render page to a pixmap at the specified DPI
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            
            # Convert the pixmap raw bytes into a numpy array (RGB)
            # PyMuPDF pixmaps are typically in RGB format when alpha=False
            img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            
            # Convert RGB to BGR for OpenCV
            if pix.n == 3:
                bgr_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                bgr_img = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
            else:
                # Fallback if something weird happens
                bgr_img = img_arr
                
            images.append(bgr_img)
            
    return images
