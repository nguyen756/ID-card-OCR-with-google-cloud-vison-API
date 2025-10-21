import io, base64, os
import streamlit as st
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv
from pathlib import Path

# Custom Streamlit component for paste-from-clipboard (optional)
try:
    from st_img_pastebutton import paste
except Exception:
    paste = None

from ocr_utils import OCRUtils
from layout_utils import easyocr_pretty
from id_parser import tidy_text, parse_id_fields
from docAI import StudentIdParser, ReceiptParser

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

st.set_page_config(page_title="OCR (EasyOCR / Google Vision)", layout="wide")
st.title("OCR (EasyOCR / Google Vision)")

engine = st.sidebar.radio("Engine", ["EasyOCR", "Google Vision"])
lang_choice = st.sidebar.radio("Language set", ["English + Vietnamese", "English + Japanese"])
lang_set = "vi" if lang_choice == "English + Vietnamese" else "ja"
vision_hints = ["en", "vi"] if lang_set == "vi" else ["en", "ja"]

docai_mode = st.sidebar.radio(
    "Document AI Processor",
    ["None", "Student ID", "Receipt"],
    help="Choose processor to extract fields."
)

mode = st.sidebar.radio("Input", ["Upload file(s)", "Paste image"])

ocr = OCRUtils()

@st.cache_resource(show_spinner=False)
def get_reader(lang_set_: str):
    return ocr.load_reader(lang_set_)

reader = get_reader(lang_set)

def run_easy(np_img):
    res = reader.readtext(np_img)
    raw = "".join([r[1] for r in res]) if res else ""
    pretty = easyocr_pretty(res)
    return raw, pretty

def ocr_image_both(pil_img):
    pil_img = ocr.resize_max(pil_img, max_dim=1600)
    if engine == "Google Vision":
        raw, pretty = ocr.vision_texts(pil_img, lang_hints=vision_hints)
        return raw, pretty
    else:
        return run_easy(np.array(pil_img))

docai_parser = None
if docai_mode == "Student ID":
    try:
        docai_parser = StudentIdParser()
    except Exception as e:
        st.warning(f"Student ID Parser (DocAI) not ready: {e}")
elif docai_mode == "Receipt":
    try:
        docai_parser = ReceiptParser()
    except Exception as e:
        st.warning(f"Receipt Parser (DocAI) not ready: {e}")


def display_docai_results(kv_fields, mode_name, regex_fields={}):
    if kv_fields:
        st.subheader(f"Fields ({mode_name})")
        st.json(kv_fields)
        merged = {**regex_fields, **kv_fields}
        st.subheader("Merged Fields")
        st.json(merged)
    



def process_image(pil_img, label="Image"):
    st.image(pil_img, caption=label, width="stretch")

    raw_text, pretty_text = ocr_image_both(pil_img)

    st.subheader("Text")
    st.text_area("", raw_text or "", height=220)
    regex_fields = {}
    if docai_mode != "Receipt": 
        regex_fields = parse_id_fields(tidy_text(pretty_text or raw_text or ""))
        if regex_fields:
            st.subheader("Fields (Regex Parser)")
            st.json(regex_fields)


    if docai_parser is not None:
        try:
            kv_fields, *rest = docai_parser.extract_with_text(pil_img)
            display_docai_results(kv_fields, docai_mode, regex_fields)
        except Exception as e:
            st.error(f"DocAI error: {e}")

if mode == "Upload file(s)":
    files = st.file_uploader("Upload image(s) or PDF(s)", type=["jpg","jpeg","png","webp","pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            st.subheader(f"{f.name}")
            if f.type == "application/pdf":
                doc = fitz.open(stream=f.read(), filetype="pdf")
                all_docai_fields = {}
                full_text_pages = []
                for i, page in enumerate(doc, 1):
                    st.caption(f"Page {i}")
                    txt = page.get_text().strip()
                    if txt:
                        st.text_area(f"Embedded Text (p{i})", txt, height=160)
                        full_text_pages.append(txt)
                        if docai_parser is not None:
                            try:
                                pix = page.get_pixmap(dpi=200)
                                pil_for_docai = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


                                kv_fields, *rest = docai_parser.extract_with_text(pil_for_docai)
                                if kv_fields: all_docai_fields.update(kv_fields)
                            except Exception as e:
                                st.error(f"DocAI error on page {i}: {e}")
                    else:
                        imgs = page.get_images(full=True)
                        if not imgs:
                            st.warning("No images on page.")
                            continue
                        for j, im in enumerate(imgs, 1):
                            xref = im[0]
                            raw = doc.extract_image(xref)["image"]
                            pil = Image.open(io.BytesIO(raw)).convert("RGB")


                            process_image(pil, label=f"Page {i} - Image {j}")
                           


                combined = "".join(full_text_pages)
                if combined and docai_mode != "Receipt":
                    fields_from_pdf_text = parse_id_fields(tidy_text(combined))
                    if fields_from_pdf_text:
                        st.subheader("Fields (Regex Parser from embedded text)")
                        st.json(fields_from_pdf_text)
                if all_docai_fields:
                    st.subheader("Fields (DocAI merged across pages)")
                    st.json(all_docai_fields)
            else:
                pil = Image.open(f).convert("RGB")
                process_image(pil, label=f.name)
else:


    if paste is not None:
        data = paste(label="📋 Click then Ctrl+V your image", key="paste")
        if data:
            try:
                _, b64 = data.split(",", 1)
                pil = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                process_image(pil, label="Pasted Image")
            except Exception as e:
                st.error(f"Paste decode error: {e}")
    else:
        st.info("Paste component not available; use Upload mode.")
