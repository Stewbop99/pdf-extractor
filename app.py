import streamlit as st
import os
import sys
import fitz
from pathlib import Path
from extractor import run_extraction



DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.title("PDF Table Extractor")
st.write("Upload a PDF, choose pages, and extract tables into Excel.")

# ------------------------------------------------------------
# PAGE PARSER (must be defined BEFORE widgets use it)
# ------------------------------------------------------------
def parse_pages(pages_str, pdf_path):
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
    except Exception as e:
        st.error(f"Could not open PDF: {e}")
        return None

    pages_str = pages_str.strip().lower()

    if pages_str == "all":
        return list(range(total))

    pages = []
    parts = pages_str.split(",")

    try:
        for part in parts:
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                pages.extend(range(int(a), int(b) + 1))
            else:
                pages.append(int(part))

        return sorted(set(pages))

    except Exception as e:
        st.error(f"Invalid page input: {e}")
        return None


# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    pdf_path = os.path.join(DATA_DIR, uploaded.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded.getvalue())

    st.success(f"Saved PDF to {pdf_path}")

    # PAGE INPUT
    pages_input = st.text_input("Pages to extract", value="all")

    # PARSE PAGES
    pages = parse_pages(pages_input, pdf_path)

    # EXTRACTION BUTTON
    if pages is not None:
        if st.button("Extract"):
            st.info("Running extraction… this may take a while.")

            output_path = run_extraction(pdf_path, pages)

            if output_path and os.path.exists(output_path):
                st.success("Extraction complete!")

                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Excel",
                        f,
                        file_name=os.path.basename(output_path)
                    )
            else:
                st.error("Extraction finished but no output file was produced.")
