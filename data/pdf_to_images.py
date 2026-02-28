"""
data/pdf_to_images.py
Convert all PDFs in data/sample_pdf/ to page-by-page PNG images in data/images/.

Naming convention: {pdf_stem}_{page_number}_image.png  (page numbers are 1-indexed)

Skips pages that already exist as images.
"""
import os
import sys
import fitz  # pymupdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR  = os.path.join(BASE_DIR, "data", "sample_pdf")
IMG_DIR  = os.path.join(BASE_DIR, "data", "images")

# Resolution: 150 DPI is a good balance for legibility vs file size.
# The existing images appear to be similar resolution.
DPI = 150
MATRIX = fitz.Matrix(DPI / 72, DPI / 72)


def convert_all():
    os.makedirs(IMG_DIR, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}\n")

    total_new = 0
    total_skipped = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        stem = os.path.splitext(pdf_file)[0]  # e.g. "1700"

        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        new_count = 0
        for page_idx in range(num_pages):
            page_num = page_idx + 1  # 1-indexed
            out_name = f"{stem}_{page_num}_image.png"
            out_path = os.path.join(IMG_DIR, out_name)

            if os.path.exists(out_path):
                total_skipped += 1
                continue

            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=MATRIX)
            pix.save(out_path)
            new_count += 1
            total_new += 1

        doc.close()

        status = f"{new_count} new" if new_count else "all cached"
        print(f"  {pdf_file:25s}  {num_pages:2d} pages  ({status})")

    print(f"\nDone. {total_new} new images saved, {total_skipped} already existed.")


if __name__ == "__main__":
    convert_all()
