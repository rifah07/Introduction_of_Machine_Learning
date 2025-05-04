import pytesseract
from PIL import Image
import re


image = Image.open('Que_paper.jpeg')

text = pytesseract.image_to_string(image)

print("Extracted Text:")
print(text)


numbers = re.findall(r'\d+', text)
print("\nExtracted Numerical Values:")
for num in numbers:
    print(num)