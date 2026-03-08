import fitz
import json

pdf_path = "readings/statphys.pdf"
doc = fitz.open(pdf_path)

data = []
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()
    data.append({"page": page_num + 1, "text": text})

with open("extracted_text.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Extracted {len(doc)} pages of text.")
