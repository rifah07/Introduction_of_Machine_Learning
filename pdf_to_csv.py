import PyPDF2
import os

# Path to your PDF file
pdf_path = "js.pdf"

if not os.path.exists(pdf_path):
    print(f"File {pdf_path} does not exist!")
else:
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Save extracted text as CSV
        with open("output.csv", "w") as csv_file:
            csv_file.write(text)
        
        print("Text extracted and saved to output.csv")
        print("First few lines of extracted text:")
        print(text[:100])  # Show first 100 characters
