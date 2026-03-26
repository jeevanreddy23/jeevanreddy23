import PyPDF2

pdf_path = r"C:\J files\AS 1726 2017 Geotechnical site investigations.pdf"

try:
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        # Assuming classification rules are generally in the middle pages
        for i in range(min(15, len(reader.pages)), min(35, len(reader.pages))):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
        
        with open("as1726_extract.txt", "w", encoding="utf-8") as out:
            out.write(text)
        print("Success! Extracted PDF to as1726_extract.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
