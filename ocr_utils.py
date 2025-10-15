

from typing import List, Optional
import os
import io
from PIL import Image
import numpy as np 
import easyocr
from google.cloud import vision
from google.oauth2 import service_account


class OCRUtils:
    def __init__(self, key_path: Optional[str] = None, credentials_info: Optional[dict] = None) -> None:
        # Resolve key path from argument, env var, or None
        env_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.key_path = key_path or env_key
        self.credentials_info = credentials_info
        self._vision_client =vision.ImageAnnotatorClient()

    def resize_image(self, image: Image.Image, max_dim: int = 1600) -> Image.Image:

        width, height = image.size
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            return image.resize(new_size, Image.LANCZOS)
        return image

    def load_reader(self, languages: List[str]):
        return easyocr.Reader(languages)

    def get_vision_client(self):
        if self._vision_client is not None:
            return self._vision_client
        creds = None
        # Prioritize provided credentials_info over file path
        if self.credentials_info:
            creds = service_account.Credentials.from_service_account_info(self.credentials_info)
        elif self.key_path and os.path.isfile(self.key_path):
            creds = service_account.Credentials.from_service_account_file(self.key_path)
        # If no creds, fallback to ADC (will throw if not configured)
        if creds:
            self._vision_client = vision.ImageAnnotatorClient(credentials=creds)
        else:
            self._vision_client = vision.ImageAnnotatorClient()
        return self._vision_client

    def vision_ocr(self, pil_image: Image.Image) -> str:
        client = self.get_vision_client()
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        content = buffer.getvalue()
        v_image = vision.Image(content=content)
        response = client.document_text_detection(image=v_image)
        if response.error and response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")
        if response.full_text_annotation and response.full_text_annotation.text:
            return response.full_text_annotation.text
        return ""