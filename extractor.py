import streamlit as st
import os
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# extractor.py
import streamlit as st
import os
import io
import time
import base64
import random
import glob
import csv
import re
from pathlib import Path

import fitz
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from openpyxl.styles import numbers

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
#promt for llm

prompt = """
You are a data-extraction system. Read the table row-by-row and output CSV only.

OUTPUT RULES
- Output CSV only. No commentary, no markdown.
- If a page contains NO table-only text, do not produce a CSV file for that page and do not output any message. Simply move on to the next page.- Use commas as delimiters.
- Any field containing a comma, quote, or newline must be wrapped in double quotes.
- Escape internal quotes by doubling them ("").
- Represent empty cells with "0" unless in a header, where they should be left blank.
- Convert true dots (.) to 0. Do not convert speckles.
- Do NOT invent or infer missing values.
- Do NOT add thousand separators. Example: 2,221 → 2221.
- Do NOT round decimals. Example: 13.0 stays 13.0 not 13.
- Replace commas inside text strings (titles, labels) with spaces.
- Replace any hyphens with %, unless it is a negative number (e.g., -5) or an age range (e.g., 0-2), in which case keep the hyphen.


HEADER RULES
- Expand merged headers into fully qualified names.
- If a header spans multiple columns, prepend its text to each subcolumn.
- If a header spans multiple rows, propagate downward.
- Convert multi-line headers to a single line.
- First column is the row header; remove commas, asterisks, ellipses from it.
- Remove any leading repeated quote patterns such as `"" "" ""` from row labels.
- Normalize labels such as `" " " 2 " Families` to `2 Families`.
- Exclude commas (,), asterisks (*) and (...) from the row headers (first column)
AGE FIELD RULE
- Age values must be output exactly as printed (e.g., "0-2", "3, 4", "16 and over").
- Do NOT normalise, reformat, or interpret age values.

TABLE RULES
- Preserve column order exactly as in the source.
- Read cells strictly left-to-right, top-to-bottom.
- Do not infer structure beyond what is visible.

EXAMPLES

Example 1: Handling commas inside text
Input cell: "Hotels, Boarding and Lodging Houses"
Output: "Hotels Boarding and Lodging Houses"

Example 2: Handling dots
Input cell: "."
Output: 0

Example 3 Escaping quotes
Input cell: He said "Yes"
Output: "He said "Yes"

Example 4: Thousand separators
Input cell: 2,221
Output: 2221

Example 5: Empty cell
Input cell: [blank]
Output: 0

Example 6: handling hyphens
Input cell: "-"
Output: %

Example 7: handling hyphens
Input cell: "0-2"
Output: "0-2"

Example 7: handling hyphens
Input cell: "-222"
Output: "-222"

Begin extraction now.
"""

# ============================================================
#  ALL OUTPUT GOES INTO /data EXACTLY AS YOU REQUESTED
# ============================================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
#  PDF → IMAGE (UNCHANGED)
# ============================================================
def pdf_to_image(pdf_path, dpi=200, page_number=0):
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    pix = page.get_pixmap(dpi=dpi)
    raw = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.n == 1:
        img = raw.reshape(pix.height, pix.width)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = raw.reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    doc.close()
    return img

# ============================================================
#  DESKEW (UNCHANGED)
# ============================================================
def deskew_image(gray):
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None:
        return gray

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed

# ============================================================
#  PREPROCESS IMAGE (UNCHANGED)
# ============================================================
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = gray

    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        23,
        4
    )

    kernel = np.ones((1, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    close_kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, close_kernel)

    return processed

# ============================================================
#  GEMINI MODEL (UNCHANGED)
# ============================================================
try:
    model
except NameError:
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.0,
        timeout=220,
        max_retries=0,
    )

# ============================================================
#  RETRY LOGIC (UNCHANGED)
# ============================================================
import httpx

HTTPX_EXC = (
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

try:
    from google.api_core.exceptions import (
        DeadlineExceeded, ServiceUnavailable, InternalServerError, ResourceExhausted,
    )
    GOOGLE_EXC = (DeadlineExceeded, ServiceUnavailable, InternalServerError, ResourceExhausted)
except ImportError:
    GOOGLE_EXC = tuple()

TRANSIENT_EXC = HTTPX_EXC + (ConnectionError, OSError, TimeoutError) + GOOGLE_EXC

def backoff(retry):
    base = 1.5 ** retry
    jitter = random.uniform(0, 0.3)
    return base + jitter

def invoke_with_retry(model, message, max_retries=5, base_delay=5):
    for attempt in range(max_retries):
        try:
            return model.invoke([message])
        except TRANSIENT_EXC as e:
            if attempt == max_retries - 1:
                raise
            delay = backoff(attempt) + base_delay
            time.sleep(delay)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in [
                "disconnect", "timeout", "unavailable", "reset", "aborted",
                "10053", "10054", "503", "500", "429", "deadline",
            ]):
                if attempt == max_retries - 1:
                    raise
                delay = backoff(attempt) + base_delay
                time.sleep(delay)
            else:
                raise

# ============================================================
#  EXTRACT ONE PAGE (UNCHANGED LOGIC — ONLY PATH CHANGED)
# ============================================================
def extract_single_page(pdf_path, page_num, pdf_prefix):
    image = pdf_to_image(pdf_path, dpi=200, page_number=page_num)
    processed = preprocess_image(image)

    rgb_binary = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    pil_image = Image.fromarray(rgb_binary)

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    response = invoke_with_retry(model, message)
    csv_output = response.content

    # Clean markdown fences (UNCHANGED)
    csv_clean = csv_output.strip()
    if csv_clean.startswith("```csv"):
        csv_clean = csv_clean[6:]
    if csv_clean.startswith("```"):
        csv_clean = csv_clean[3:]
    if csv_clean.endswith("```"):
        csv_clean = csv_clean[:-3]
    csv_clean = csv_clean.strip()

    # Skip empty CSVs (UNCHANGED)
    if not csv_clean or "," not in csv_clean:
        return None

    # ============================================================
    #  ORIGINAL SAVE PATH (COMMENTED OUT — DO NOT DELETE)
    # ============================================================
    # original_path = os.path.join("data/no_deskew", f"{pdf_prefix}_page_{page_num+1}.csv")
    # with open(original_path, "w", encoding="utf-8") as f:
    #     f.write(csv_clean)

    # ============================================================
    #  NEW SAVE PATH — EXACT SAME FILENAME, NOW IN /data
    # ============================================================
    new_path = os.path.join(DATA_DIR, f"{pdf_prefix}_page_{page_num+1}.csv")
    with open(new_path, "w", encoding="utf-8") as f:
        f.write(csv_clean)

    return new_path

# ============================================================
#  % → - TRANSFORMATION (UNCHANGED LOGIC — ONLY PATH UPDATED)
# ============================================================
def percent_to_hyphen_pass(pdf_prefix):
    """
    Runs your exact '%' → '-' rewrite on all CSVs for this PDF.
    """
    # ORIGINAL:
    # folder = "data/deskew"
    # csv_files = glob.glob(os.path.join(folder, "*_page_*.csv"))

    folder = DATA_DIR
    csv_files = glob.glob(os.path.join(folder, f"{pdf_prefix}_page_*.csv"))

    for path in csv_files:
        rows = []

        # Read CSV exactly as-is
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                new_row = []
                for cell in row:
                    cell_stripped = cell.strip()

                    # Replace exact "%" with hyphen (UNCHANGED)
                    if cell_stripped == "%":
                        new_cell = "-"
                    else:
                        new_cell = cell

                    new_row.append(new_cell)

                rows.append(new_row)

        # Write back to same file (UNCHANGED)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

# ============================================================
#  EXCEL COMBINER — YOUR LOGIC EXACTLY AS IN THE PROTOTYPE
# ============================================================

def page_num_from_path(p: Path):
    """Extract page number from filename for sorting."""
    m = re.search(r"_page_(\d+)", p.stem)
    return int(m.group(1)) if m else 9999


def read_csv_safe(path):
    """
    Read a CSV and preserve every row, even if column counts are inconsistent.
    Short rows are padded with empty strings; extra columns get auto-named headers.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(c.strip() for c in r)]

    if not rows:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)

    # Build header: use row 0 and pad with extra_N if later rows are wider
    header = rows[0] + [f"extra_{i}" for i in range(max_cols - len(rows[0]))]

    # Pad every data row out to max_cols
    data = [r + [""] * (max_cols - len(r)) for r in rows[1:]]

    return pd.DataFrame(data, columns=header)


# ============================================================
#  NUMERIC DETECTION — YOUR EXACT LOGIC
# ============================================================
def is_numeric_value(v):
    v = v.strip()

    # Hyphens meaning "no data" → treat as text
    if v in ["-", "–", "—", "‑"]:
        return False

    # Dagger-prefixed → text
    if v.startswith(("†", "‡")):
        return False

    # Normalise Unicode minus signs to ASCII
    v_norm = (
        v.replace("−", "-")
         .replace("–", "-")
         .replace("—", "-")
         .replace("‑", "-")
    )

    # Remove commas
    v_clean = v_norm.replace(",", "")

    # Integer?
    if re.fullmatch(r"-?\d+", v_clean):
        return True

    # Decimal?
    if re.fullmatch(r"-?\d+\.\d+", v_clean):
        return True

    return False


# ============================================================
#  COMBINE CSV → EXCEL (UNCHANGED LOGIC — ONLY PATH UPDATED)
# ============================================================
def combine_csv_to_excel(pdf_prefix):
    """
    EXACT behaviour of your working prototype.
    Reads all CSVs for this PDF from /data,
    applies your numeric logic,
    writes combined Excel to /data.
    """

    # ORIGINAL:
    # input_folder = "data/deskew"
    # output_file = Path(input_folder) / f"{pdf_basename}_all_pages.xlsx"

    input_folder = DATA_DIR
    output_file = Path(DATA_DIR) / f"{pdf_prefix}_all_pages.xlsx"

    # Delete old workbook if exists (UNCHANGED)
    if output_file.exists():
        output_file.unlink()

    # Find CSVs (UNCHANGED)
    csv_files = sorted(
        Path(input_folder).glob(f"{pdf_prefix}_page_*.csv"),
        key=page_num_from_path,
    )

    if not csv_files:
        return None

    sheets_written = 0

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for csv_path in csv_files:
            try:
                df = read_csv_safe(csv_path)
                sheet_name = f"Page_{page_num_from_path(csv_path)}"[:31]

                # Write raw data first (UNCHANGED)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Apply numeric conversion + formatting (UNCHANGED)
                ws = writer.book[sheet_name]

                for col_idx, col_name in enumerate(df.columns, start=1):
                    col_values = df[col_name].dropna().astype(str)

                    # If at least one numeric value exists, process column
                    if any(is_numeric_value(v) for v in col_values):

                        for cell in ws.iter_cols(
                            min_col=col_idx, max_col=col_idx,
                            min_row=2, max_row=ws.max_row
                        ):
                            for c in cell:
                                if c.value is None:
                                    continue

                                v = str(c.value).strip()

                                # Hyphens → force text
                                if v in ["-", "–", "—", "‑"]:
                                    c.data_type = "s"
                                    continue

                                # Dagger-prefixed → force text
                                if v.startswith(("†", "‡")):
                                    c.data_type = "s"
                                    continue

                                # Normalise minus signs
                                v_norm = (
                                    v.replace("−", "-")
                                     .replace("–", "-")
                                     .replace("—", "-")
                                     .replace("‑", "-")
                                )

                                # Remove commas
                                v_clean = v_norm.replace(",", "")

                                # Integer
                                if re.fullmatch(r"-?\d+", v_clean):
                                    c.value = int(v_clean)
                                    c.number_format = "#,##0"
                                    continue

                                # Decimal
                                if re.fullmatch(r"-?\d+\.\d+", v_clean):
                                    c.value = float(v_clean)

                                    # Count decimal places in original value
                                    decimal_places = len(v_clean.split(".")[1])

                                    # Build dynamic Excel format
                                    c.number_format = "#,##0." + ("0" * decimal_places)
                                    continue

                sheets_written += 1

            except Exception:
                continue

        # Excel cannot save with zero sheets
        if sheets_written == 0:
            pd.DataFrame({"info": ["No CSVs could be parsed"]}).to_excel(
                writer, sheet_name="Empty", index=False
            )

    return str(output_file)

# ============================================================
#  FINAL PIPELINE — EXACT ORDER OF OPERATIONS PRESERVED
# ============================================================

def run_extraction(pdf_path, pages):
    """
    EXACT behaviour of your working prototype, wrapped for Streamlit.

    ORDER (unchanged):
        1. Extract CSVs for each page
        2. Run % → - pass
        3. Combine CSVs into Excel

    All output goes into /data.
    """

    # Ensure /data exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Extract PDF basename (unchanged)
    pdf_prefix = Path(pdf_path).stem

    extracted_csvs = []

    # ============================================================
    # 1. EXTRACT CSVs (UNCHANGED LOGIC)
    # ============================================================
    for page_num in pages:
        st.write(f"Extracting page {page_num}…")

        try:
            csv_path = extract_single_page(pdf_path, page_num, pdf_prefix)
            if csv_path:
                st.success(f"Page {page_num} extracted → {csv_path}")
                extracted_csvs.append(csv_path)
            else:
                st.error(f"Page {page_num} returned no CSV.")
        except Exception as e:
            st.error(f"Page {page_num} failed: {e}")
            continue

    # ============================================================
    # 2. RUN % → - PASS (UNCHANGED LOGIC)
    # ============================================================
    percent_to_hyphen_pass(pdf_prefix)

    # ============================================================
    # 3. COMBINE CSVs INTO EXCEL (UNCHANGED LOGIC)
    # ============================================================
    excel_path = combine_csv_to_excel(pdf_prefix)

    return excel_path
