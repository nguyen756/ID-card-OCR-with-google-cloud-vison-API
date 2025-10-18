from dotenv import load_dotenv
from docAI import DocAIKV
# Custom Streamlit component for paste-from-clipboard
from st_img_pastebutton import paste
from pathlib import Path
# Import OCR utilities and field extractors
# main.py
import io, base64, os
import streamlit as st
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
import os
print("DOC_AI_LOCATION =", os.getenv("DOC_AI_LOCATION"))
print("DOC_AI_PROCESSOR_ID =", os.getenv("DOC_AI_PROCESSOR_ID"))
from ocr_utils import OCRUtils
from layout_utils import easyocr_pretty
from id_parser import tidy_text, parse_id_fields

st.set_page_config(page_title="OCR (EasyOCR / Google Vision)", layout="wide")
st.title("📄 OCR (EasyOCR / Google Vision)")

# OCR engine and language choices
engine = st.sidebar.radio("Engine", ["EasyOCR", "Google Vision"])
lang_choice = st.sidebar.radio("Language set", ["English + Vietnamese", "English + Japanese"])
lang_set = "vi" if lang_choice == "English + Vietnamese" else "ja"
vision_hints = ["en", "vi"] if lang_set == "vi" else ["en", "ja"]
docai_kv = None
use_docai = st.sidebar.checkbox("Extract fields with Document AI (KV/Tables)")
if use_docai:
    try:
        docai_kv = DocAIKV()   # uses env vars DOC_AI_LOCATION, DOC_AI_PROCESSOR_ID
    except Exception as e:
        st.warning(f"Document AI not available: {e}")
        docai_kv = None
ocr = OCRUtils()

@st.cache_resource(show_spinner=False)
def get_reader(lang_set_: str):
    return ocr.load_reader(lang_set_)

reader = get_reader(lang_set)

def run_easy(np_img):
    res = reader.readtext(np_img)
    return "\n".join([r[1] for r in res]) if res else "", easyocr_pretty(res)

def ocr_image(pil_img):
    pil_img = ocr.resize_max(pil_img, max_dim=1600)
    if engine == "Google Vision":
        try:
            return ocr.vision_ocr(pil_img, lang_hints=vision_hints)
        except Exception as e:
            st.error(f"Vision error: {e}")
            return ""
    else:
        _, pretty = run_easy(np.array(pil_img))
        return pretty

mode = st.sidebar.radio("Input", ["Upload file(s)", "Paste image"])
if mode == "Upload file(s)":
    files = st.file_uploader("Upload image(s) or PDF(s)", type=["jpg","jpeg","png","webp","pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            st.subheader(f.name)
            if f.type == "application/pdf":
                doc = fitz.open(stream=f.read(), filetype="pdf")
                full_text = []
                for i, page in enumerate(doc, 1):
                    st.caption(f"Page {i}")
                    txt = page.get_text().strip()
                    if not txt:
                        imgs = page.get_images(full=True)
                        if not imgs:
                            st.warning("No images on page.")
                            continue
                        for j, im in enumerate(imgs, 1):
                            xref = im[0]
                            raw = doc.extract_image(xref)["image"]
                            pil = Image.open(io.BytesIO(raw)).convert("RGB")
                            st.image(pil, caption=f"Page {i} - Image {j}", width="stretch")
                            page_text = ocr_image(pil)
                            full_text.append(page_text)
                            st.text_area(f"OCR Text (p{i}-img{j})", page_text, height=140)
                    else:
                        st.text_area(f"Embedded Text (p{i})", txt, height=140)
                        full_text.append(txt)
                combined = tidy_text("\n\n".join(full_text))
                if combined:
                    st.download_button("Download text", data=combined, file_name="pdf_ocr.txt", mime="text/plain")
                    if use_docai and docai_kv is not None:
                        try:
                            kv_fields, tables = docai_kv.extract(pil)
                            if kv_fields:
                                st.subheader("Fields (Document AI)")
                                st.json(kv_fields)
                                merged = {**fields, **kv_fields} if fields else kv_fields
                                st.subheader("Merged Fields")
                                st.json(merged)
                            if tables:
                                st.subheader("Tables (Document AI)")
                                for i, t in enumerate(tables, 1):
                                    st.caption(f"Table {i}")
                                    if t.get("headers"):
                                        st.write("| " + " | ".join(t["headers"]) + " |")
                                        st.write("| " + " | ".join(["---"] * len(t["headers"])) + " |")
                                    for row in t.get("rows", []):
                                        st.write("| " + " | ".join(row) + " |")
                        except Exception as e:
                            st.error(f"DocAI error: {e}")
            else:
                pil = Image.open(f).convert("RGB")
                st.image(pil, caption=f.name, width="stretch")
                if st.button(f"Extract: {f.name}"):
                    text = ocr_image(pil)
                    cleaned = tidy_text(text)
                    fields = parse_id_fields(cleaned)
                    st.text_area("Text", cleaned, height=220)
                    if fields:
                        st.json(fields)
                    st.download_button("Download text", data=cleaned, file_name=f"{os.path.splitext(f.name)[0]}_ocr.txt", mime="text/plain")
                    if use_docai and docai_kv is not None:
                        try:
                            kv_fields, tables = docai_kv.extract(pil)
                            if kv_fields:
                                st.subheader("Fields (Document AI)")
                                st.json(kv_fields)
                                merged = {**fields, **kv_fields} if fields else kv_fields
                                st.subheader("Merged Fields")
                                st.json(merged)
                            if tables:
                                st.subheader("Tables (Document AI)")
                                for i, t in enumerate(tables, 1):
                                    st.caption(f"Table {i}")
                                    if t.get("headers"):
                                        st.write("| " + " | ".join(t["headers"]) + " |")
                                        st.write("| " + " | ".join(["---"] * len(t["headers"])) + " |")
                                    for row in t.get("rows", []):
                                        st.write("| " + " | ".join(row) + " |")
                        except Exception as e:
                            st.error(f"DocAI error: {e}")

else:
    from st_img_pastebutton import paste
    data = paste(label="📋 Click then Ctrl+V your image", key="paste")
    if data:
        _, b64 = data.split(",", 1)
        pil = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        st.image(pil, caption="Pasted Image", width="auto")
        if st.button("Extract"):
            text = ocr_image(pil)
            cleaned = tidy_text(text)
            fields = parse_id_fields(cleaned)
            st.text_area("Text", cleaned, height=220)
            if fields:
                st.json(fields)
            st.download_button("Download text", data=cleaned, file_name="pasted_ocr.txt", mime="text/plain")
            if use_docai and docai_kv is not None:
                try:
                    kv_fields, tables = docai_kv.extract(pil)
                    if kv_fields:
                        st.subheader("Fields (Document AI)")
                        st.json(kv_fields)
                        merged = {**fields, **kv_fields} if fields else kv_fields
                        st.subheader("Merged Fields")
                        st.json(merged)
                    if tables:
                        st.subheader("Tables (Document AI)")
                        for i, t in enumerate(tables, 1):
                            st.caption(f"Table {i}")
                            if t.get("headers"):
                                st.write("| " + " | ".join(t["headers"]) + " |")
                                st.write("| " + " | ".join(["---"] * len(t["headers"])) + " |")
                            for row in t.get("rows", []):
                                st.write("| " + " | ".join(row) + " |")
                except Exception as e:
                    st.error(f"DocAI error: {e}")
    else:
        st.info("Click above box and paste an image.")
