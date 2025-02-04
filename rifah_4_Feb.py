#pip install pillow
#pip install pytesseract pillow
#sudo apt update
#sudo apt install tesseract-ocr


import pytesseract
from PIL import Image
import csv

image_path = "image.jpeg"
output_csv = "generated_csv.csv"

#convert image to text using Tesseract
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="ben+eng")  # Specify both Bengali (ben) and English (eng) for better OCR
    return text

#save extracted text to CSV
def save_text_to_csv(text, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        #split text by line and write each line to CSV
        for line in text.splitlines():
            writer.writerow([line])

#main processing
text = extract_text_from_image(image_path)
save_text_to_csv(text, output_csv)

print(f"Text has been extracted and saved to {output_csv}")