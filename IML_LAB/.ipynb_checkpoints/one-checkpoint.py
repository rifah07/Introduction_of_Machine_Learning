import pytesseract
from PIL import Image
import re


image_path = 'Que_paper.jpeg'
image = Image.open(image_path)

text = pytesseract.image_to_string(image)

print("Extracted Text:")
print(text)


numbers = re.findall(r'\d+', text)
print("\nExtracted Numerical Values:")
for num in numbers:
    print(num)