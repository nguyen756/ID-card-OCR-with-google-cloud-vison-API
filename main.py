# main.py
import io, base64, os
import streamlit as st
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from dotenv import load_dotenv
from pathlib import Path

# Custom Streamlit component for paste-from-clipboard
from st_img_pastebutton import paste

# Tải .env
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# --- THAY ĐỔI IMPORT ---
# Import các parser mới từ file docAI.py (hoặc docAI_ID.py tùy bạn đặt tên)
from docAI import StudentIdParser, ReceiptParser 
# --------------------

# Import OCR utilities and field extractors
from ocr_utils import OCRUtils
from layout_utils import easyocr_pretty
from id_parser import tidy_text, parse_id_fields # Đây là parser dựa trên regex

# Debug
print("DOC_AI_LOCATION =", os.getenv("DOC_AI_LOCATION"))
print("DOC_AI_PROCESSOR_ID =", os.getenv("DOC_AI_PROCESSOR_ID"))
print("RECEIPT_PROCESSOR_ID =", os.getenv("RECEIPT_PROCESSOR_ID")) # Kiểm tra env var mới

st.set_page_config(page_title="OCR (EasyOCR / Google Vision)", layout="wide")
st.title("📄 OCR (EasyOCR / Google Vision)")

# OCR engine and language choices
engine = st.sidebar.radio("Engine", ["EasyOCR", "Google Vision"])
lang_choice = st.sidebar.radio("Language set", ["English + Vietnamese", "English + Japanese"])
lang_set = "vi" if lang_choice == "English + Vietnamese" else "ja"
vision_hints = ["en", "vi"] if lang_set == "vi" else ["en", "ja"]

# --- THAY ĐỔI LOGIC CHỌN PARSER ---
docai_mode = st.sidebar.radio(
    "Document AI Processor",
    ["None", "Student ID", "Receipt"],
    help="Chọn processor để trích xuất trường. 'Student ID' dùng DOC_AI_PROCESSOR_ID, 'Receipt' dùng RECEIPT_PROCESSOR_ID."
)

docai_parser = None
if docai_mode == "Student ID":
    try:
        # Lớp này dùng DOC_AI_PROCESSOR_ID
        docai_parser = StudentIdParser() 
    except Exception as e:
        st.warning(f"Student ID Parser (DocAI) không sẵn sàng: {e}")
elif docai_mode == "Receipt":
    try:
        # Lớp này dùng RECEIPT_PROCESSOR_ID
        docai_parser = ReceiptParser() 
    except Exception as e:
        st.warning(f"Receipt Parser (DocAI) không sẵn sàng: {e}")
# ----------------------------------

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

def display_docai_results(kv_fields, tables, mode_name, regex_fields={}):
    """Hàm tiện ích để hiển thị kết quả từ DocAI và merge."""
    if kv_fields:
        st.subheader(f"Fields ({mode_name})")
        st.json(kv_fields)
        
        # Merge regex fields (nếu có) với DocAI fields
        merged = {**regex_fields, **kv_fields}
        st.subheader("Merged Fields")
        st.json(merged)
        
    if tables:
        st.subheader(f"Tables ({mode_name})")
        for i, t in enumerate(tables, 1):
            st.caption(f"Table {i}")
            if t.get("headers"):
                st.write("| " + " | ".join(t["headers"]) + " |")
                st.write("| " + " | ".join(["---"] * len(t["headers"])) + " |")
            for row in t.get("rows", []):
                st.write("| " + " | ".join(row) + " |")

mode = st.sidebar.radio("Input", ["Upload file(s)", "Paste image"])

if mode == "Upload file(s)":
    files = st.file_uploader("Upload image(s) or PDF(s)", type=["jpg","jpeg","png","webp","pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            st.subheader(f.name)
            if f.type == "application/pdf":
                doc = fitz.open(stream=f.read(), filetype="pdf")
                full_text = []
                
                # --- SỬA LỖI LOGIC PDF ---
                # Các biến để tổng hợp kết quả từ tất cả các trang
                all_docai_fields = {}
                all_docai_tables = []
                
                for i, page in enumerate(doc, 1):
                    st.caption(f"Page {i}")
                    txt = page.get_text().strip()
                    page_pil_images = [] # Lưu các ảnh PIL từ trang này

                    if not txt:
                        imgs = page.get_images(full=True)
                        if not imgs:
                            st.warning("No images on page.")
                            continue
                        for j, im in enumerate(imgs, 1):
                            xref = im[0]
                            raw = doc.extract_image(xref)["image"]
                            pil = Image.open(io.BytesIO(raw)).convert("RGB")
                            page_pil_images.append(pil) # Thêm ảnh để xử lý DocAI
                            
                            st.image(pil, caption=f"Page {i} - Image {j}", width="stretch")
                            page_text = ocr_image(pil)
                            full_text.append(page_text)
                            st.text_area(f"OCR Text (p{i}-img{j})", page_text, height=140)
                    else:
                        st.text_area(f"Embedded Text (p{i})", txt, height=140)
                        full_text.append(txt)
                        # Lưu ý: Sẽ không chạy DocAI trên trang có text
                        # vì chúng ta không có ảnh PIL cho nó (giống code gốc)

                    # Chạy DocAI trên TỪNG ảnh PIL tìm thấy trên trang này
                    if docai_parser is not None:
                        for pil_img in page_pil_images:
                            try:
                                kv_fields, tables = docai_parser.extract(pil_img)
                                if kv_fields:
                                    all_docai_fields.update(kv_fields) # Gộp fields
                                if tables:
                                    all_docai_tables.extend(tables) # Nối list tables
                            except Exception as e:
                                st.error(f"DocAI error on page {i}: {e}")

                # --- Sau khi xử lý tất cả các trang ---
                combined = tidy_text("\n\n".join(full_text))
                
                # Chạy parser regex (nếu không phải mode Receipt)
                fields = {}
                if docai_mode != "Receipt":
                    fields = parse_id_fields(combined)
                
                if combined:
                    st.download_button("Download text", data=combined, file_name="pdf_ocr.txt", mime="text/plain")

                # Hiển thị kết quả regex (nếu có)
                if fields:
                    st.subheader("Fields (Regex Parser)")
                    st.json(fields)

                # Hiển thị kết quả DocAI tổng hợp (và merge)
                display_docai_results(all_docai_fields, all_docai_tables, docai_mode, fields)
                # -------------------------

            else:
                # Logic xử lý file ảnh (PNG, JPG...)
                pil = Image.open(f).convert("RGB")
                st.image(pil, caption=f.name, width="stretch")
                
                if st.button(f"Extract: {f.name}"):
                    text = ocr_image(pil)
                    cleaned = tidy_text(text)
                    
                    # Chạy parser regex (nếu không phải mode Receipt)
                    fields = {}
                    if docai_mode != "Receipt":
                        fields = parse_id_fields(cleaned)
                    
                    st.text_area("Text", cleaned, height=220)
                    
                    # Hiển thị kết quả regex (nếu có)
                    if fields:
                        st.subheader("Fields (Regex Parser)")
                        st.json(fields)
                    
                    st.download_button("Download text", data=cleaned, file_name=f"{os.path.splitext(f.name)[0]}_ocr.txt", mime="text/plain")
                    
                    # Chạy DocAI (nếu được chọn)
                    if docai_parser is not None:
                        try:
                            kv_fields, tables = docai_parser.extract(pil)
                            display_docai_results(kv_fields, tables, docai_mode, fields)
                        except Exception as e:
                            st.error(f"DocAI error: {e}")

else:
    # Logic xử lý ảnh paste (dán)
    data = paste(label="📋 Click then Ctrl+V your image", key="paste")
    if data:
        _, b64 = data.split(",", 1)
        pil = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        st.image(pil, caption="Pasted Image", width="stretch")
        
        if st.button("Extract"):
            text = ocr_image(pil)
            cleaned = tidy_text(text)
            
            # Chạy parser regex (nếu không phải mode Receipt)
            fields = {}
            if docai_mode != "Receipt":
                fields = parse_id_fields(cleaned)
                
            st.text_area("Text", cleaned, height=220)
            
            # Hiển thị kết quả regex (nếu có)
            if fields:
                st.subheader("Fields (Regex Parser)")
                st.json(fields)
                
            st.download_button("Download text", data=cleaned, file_name="pasted_ocr.txt", mime="text/plain")
            
            # Chạy DocAI (nếu được chọn)
            if docai_parser is not None:
                try:
                    kv_fields, tables = docai_parser.extract(pil)
                    display_docai_results(kv_fields, tables, docai_mode, fields)
                except Exception as e:
                    st.error(f"DocAI error: {e}")
    else:
        st.info("Click above box and paste an image.")