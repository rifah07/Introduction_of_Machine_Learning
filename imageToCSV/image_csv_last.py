import cv2
import pytesseract
import pandas as pd
import numpy as np

# Set Tesseract path (only needed for Windows)
# Uncomment and update the path if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding (helps with noisy backgrounds)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return processed

def image_to_csv(image_path, output_csv):
    """Extract Bangla and English text from an image and save to CSV"""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return
    
    # Perform OCR using Tesseract with Bangla and English support
    extracted_text = pytesseract.image_to_string(processed_image, lang="ben+eng")
    
    # Convert text into a structured format (list of lines)
    lines = extracted_text.split("\n")
    
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Save extracted text to a CSV file
    df = pd.DataFrame(lines, columns=["Extracted Text"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")  # utf-8-sig ensures proper Bangla text encoding
    
    print(f"Text extracted and saved to: {output_csv}")

# Example usage
image_to_csv("image.jpeg", "extracted_text.csv")
