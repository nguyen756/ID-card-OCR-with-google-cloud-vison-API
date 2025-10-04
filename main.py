import streamlit as st
import fitz
from PIL import Image
import io
import numpy as np
from st_img_pastebutton import paste
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
# Import helper classes for OCR and field extraction
from ocr_utils import OCRUtils
from field_extractors import GenericCardFieldExtractor




"""
truoc luc xai nho trong powershell
setx GOOGLE_APPLICATION_CREDENTIALS "C:\path\vision-key.json"
"""
KEY_PATH = r"D:\College\_hk5\AItesting\simple-OCR\data\vision-key.json"
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GROQ_API_KEY")

ocr_utils = OCRUtils(key_path=KEY_PATH)


st.set_page_config(page_title="OCR with EasyOCR and Google Vision", layout="wide")
st.title("📄 OCR with EasyOCR and Google Vision")
st.write(
    "Upload an **image/PDF** or paste an image from your clipboard to extract text."
)

# Allow the user to choose the OCR engine
engine = st.sidebar.radio(
    "Select OCR Engine:", ["EasyOCR", "Google Vision (Free tier ~1k req/mo)"]
)

# Resize large images to prevent excessive processing time and memory usage
def resize_image(image: Image.Image, max_dim: int = 1600) -> Image.Image:
    return ocr_utils.resize_image(image, max_dim)

# Cache EasyOCR models for Vietnamese and Japanese to avoid reloading on every run
@st.cache_resource(show_spinner=False)
def load_reader_vi():
    """Load an EasyOCR reader for English and Vietnamese using OCRUtils."""
    return ocr_utils.load_reader(["en", "vi"])

@st.cache_resource(show_spinner=False)
def load_reader_ja():
    """Load an EasyOCR reader for English and Japanese using OCRUtils."""
    return ocr_utils.load_reader(["en", "ja"])



# Choose the language set for EasyOCR
lang_choice = st.sidebar.radio(
    "Select OCR language set:", ["English + Vietnamese", "English + Japanese"]
)
reader = load_reader_vi() if lang_choice == "English + Vietnamese" else load_reader_ja()

# Choose the input method: file upload or clipboard paste
method = st.sidebar.radio(
    "Select input method:", ["Upload file", "Paste image"]
)

if method == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload image or PDF", type=["jpg", "jpeg", "png", "pdf","webp"]
    )

    if uploaded_file is not None:
        # Handle PDF files (EasyOCR only)
        if uploaded_file.type == "application/pdf":
            st.info("Processing PDF...")
            pdf_data = uploaded_file.read()
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            extracted_text = ""

            for i, page in enumerate(doc, start=1):
                st.subheader(f"Page {i}")
                page_text = page.get_text().strip()

                # Use text if embedded in the PDF
                if page_text:
                    st.text_area(f"Text (Page {i})", page_text, height=150)
                    extracted_text += f"\n\n--- Page {i} (Text) ---\n{page_text}"
                else:
                    st.warning(f"No direct text found on page {i}. Running OCR...")
                    images = page.get_images(full=True)
                    if images:
                        for j, img in enumerate(images, start=1):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            pil_img = resize_image(pil_img)
                            st.image(pil_img, caption=f"Page {i} - Image {j}", width="stretch")
                            np_image = np.array(pil_img)
                            results = reader.readtext(np_image)
                            ocr_text = "\n".join([res[1] for res in results])
                            st.text_area(
                                f"OCR Text (Page {i} - Image {j})", ocr_text, height=150
                            )
                            extracted_text += f"\n\n--- Page {i}, Image {j} ---\n{ocr_text}"
                    else:
                        st.warning(f"No images found on page {i}.")

            if extracted_text:
                st.download_button(
                    "Download Extracted Text",
                    data=extracted_text,
                    file_name="pdf_ocr_output.txt",
                    mime="text/plain",
                )

        # Handle image files
        else:
            st.info("Processing Image...")
            pil_img = Image.open(uploaded_file).convert("RGB")
            pil_img = resize_image(pil_img)
            st.image(pil_img, caption="Uploaded Image", width="stretch")

            if st.button("Extract Text"):
                if engine == "Google Vision":
                    try:
                        extracted_text = ocr_utils.vision_ocr(pil_img)
                    except Exception as e:
                        st.error(str(e))
                        extracted_text = ""
                else:
                    # Use EasyOCR
                    np_image = np.array(pil_img)
                    results = reader.readtext(np_image)
                    extracted_text = "\n".join([res[1] for res in results])

                # Display extracted text and structured invoice fields
                st.text_area("Detected text", extracted_text, height=200)
                if extracted_text:
                    extractor = GenericCardFieldExtractor()
                    fields = extractor.extract_fields(extracted_text)
                    if fields:
                        st.subheader("Structured ID Fields (beta)")
                        st.json(fields)
                st.download_button(
                    "Download Extracted Text",
                    data=extracted_text,
                    file_name="image_ocr_output.txt",
                    mime="text/plain",
                )

elif method == "Paste image":
    image_data = paste(
        label="📋 Click here, then paste your image (Ctrl+V)", key="pastebox"
    )

    if image_data:
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(binary_data)).convert("RGB")
        pil_img = resize_image(pil_img)
        st.image(pil_img, caption="Pasted Image", width="stretch")

        if st.button("Extract Text"):
            if engine == "Google Vision (Free tier ~1k req/mo)":
                try:
                    extracted_text = ocr_utils.vision_ocr(pil_img)
                except Exception as e:
                    st.error(str(e))
                    extracted_text = ""
            else:
                np_image = np.array(pil_img)
                results = reader.readtext(np_image)
                extracted_text = "\n".join([res[1] for res in results])

            st.text_area("Detected Text", extracted_text, height=200)
            if extracted_text:
                # Use the invoice field extractor for pasted images
                fields = invoice_extractor.extract_fields(extracted_text)
                if fields:
                    st.subheader("Structured Invoice Fields (beta)")
                    st.json(fields)
            st.download_button(
                "Download Extracted Text",
                data=extracted_text,
                file_name="pasted_image_ocr.txt",
                mime="text/plain",
            )


            prompt = f"""If any text in this image is a question, answer it. If not, just say 'no question found'. If there is information, try to elaborate.\n\nImage content:\n{extracted_text}\n"""
            if gemini_api_key:
                client = OpenAI(
                    api_key=gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                response = client.chat.completions.create(
                    model="gemini-1.5-flash",
                    messages=[
                        {
                            "role": "system",
                            "content": "pretend you are rick sanchez, but you don't say Wubba Lubba Dub Dub. You will be blunt yet still answer the question. If there are any answers, wrap them in parentheses.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                )
                st.markdown("### Analysis Result:")
                st.markdown(response.choices[0].message.content)
    else:
        st.info("Click the box above, copy an image, and press Ctrl+V.")