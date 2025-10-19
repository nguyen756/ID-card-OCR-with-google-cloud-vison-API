#ocr_utils.py
from typing import List, Optional, Tuple
import io
import numpy as np
from PIL import Image
from google.cloud import vision
import easyocr
from layout_utils import easyocr_pretty

class OCRUtils:
    def __init__(self, languages_vi=("en", "vi"), languages_ja=("en", "ja")):
        self._reader_vi = None
        self._reader_ja = None
        self.languages_vi = list(languages_vi)
        self.languages_ja = list(languages_ja)
        self._vision_client = None

    @staticmethod
    def resize_max(pil_image: Image.Image, max_dim: int = 1600) -> Image.Image:
        w, h = pil_image.size
        m = max(w, h)
        if m <= max_dim:
            return pil_image
        scale = max_dim / float(m)
        return pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    def get_vision_client(self):
        if self._vision_client is None:
            self._vision_client = vision.ImageAnnotatorClient()  # ADC on Cloud Run or env creds
        return self._vision_client

    def load_reader(self, lang_set: str):
        if lang_set == "vi":
            if self._reader_vi is None:
                self._reader_vi = easyocr.Reader(self.languages_vi)
            return self._reader_vi
        else:
            if self._reader_ja is None:
                self._reader_ja = easyocr.Reader(self.languages_ja)
            return self._reader_ja

    def easyocr_texts(self, reader, np_img) -> Tuple[str, str]:
        res = reader.readtext(np_img)
        raw = "\n".join([r[1] for r in res]) if res else ""
        pretty = easyocr_pretty(res)
        return raw, pretty

    def vision_texts(self, pil_image: Image.Image, lang_hints: Optional[List[str]] = None) -> Tuple[str, str]:
        buf = io.BytesIO(); pil_image.save(buf, format="PNG")
        vimg = vision.Image(content=buf.getvalue())
        ctx = vision.ImageContext(language_hints=lang_hints or ["en"])
        client = self.get_vision_client()
        resp = client.document_text_detection(image=vimg, image_context=ctx)
        if resp.error and resp.error.message:
            raise RuntimeError(f"Vision OCR error: {resp.error.message}")
        raw = ""
        if resp.full_text_annotation and resp.full_text_annotation.text:
            raw = resp.full_text_annotation.text
        elif resp.text_annotations:
            raw = resp.text_annotations[0].description or ""
        pretty = self._vision_lines_with_gutter(resp.full_text_annotation) if resp.full_text_annotation else raw
        return raw, pretty
    def _vision_lines_with_gutter(self, annotation) -> str:
        import numpy as np
        words = []
        colon_x = []
        for page in getattr(annotation, "pages", []):
            for block in getattr(page, "blocks", []):
                for para in getattr(block, "paragraphs", []):
                    for w in getattr(para, "words", []):
                        token = "".join(s.text for s in getattr(w, "symbols", []) or []) or ""
                        xs = [v.x for v in w.bounding_box.vertices if v.x is not None]
                        ys = [v.y for v in w.bounding_box.vertices if v.y is not None]
                        if not token.strip() or not xs or not ys:
                            continue
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        cy = (y1 + y2) / 2.0
                        h = (y2 - y1) or 1.0
                        words.append({"t": token, "x1": x1, "x2": x2, "y": cy, "h": h})
                        if token == ":":
                            colon_x.append((x1 + x2) / 2.0)
        if not words:
            return ""
        words.sort(key=lambda w: (w["y"], w["x1"]))
        med_h = float(np.median([w["h"] for w in words])) or 1.0
        tol = med_h * 0.6
        lines, cur, last_y = [], [], None
        for w in words:
            if last_y is None or abs(w["y"] - last_y) <= tol:
                cur.append(w)
            else:
                lines.append(cur); cur = [w]
            last_y = w["y"]
        if cur: lines.append(cur)
        if colon_x:
            gutter = float(np.median(colon_x))
        else:
            mids = []
            for ln in lines:
                xs = sorted([(w["x1"] + w["x2"]) / 2.0 for w in ln])
                if xs:
                    mids.append(xs[len(xs)//2])
            gutter = float(np.median(mids)) if mids else None
        out_lines = []
        carry_label = ""
        def is_header(line_text: str) -> bool:
            s = line_text.strip()
            if ("IDENTIFICATION" in s.upper()) or ("ID CARD" in s.upper()):
                return True
            letters = [ch for ch in s if ch.isalpha()]
            return bool(letters) and (sum(1 for ch in letters if ch.isupper()) / len(letters) >= 0.8)
        for ln in lines:
            ln.sort(key=lambda w: w["x1"])
            raw = " ".join(w["t"] for w in ln).strip()
            if is_header(raw):
                out_lines.append(raw)
                carry_label = ""
                continue
            if gutter is None:
                out_lines.append(raw)
                continue
            left = [w for w in ln if w["x2"] <= gutter]
            right = [w for w in ln if w["x1"] > gutter]
            left_text  = " ".join(w["t"] for w in left).strip()
            right_text = " ".join(w["t"] for w in right).strip()
            if left_text == ":": left_text = ""
            if right_text == ":": right_text = ""
            if left_text and right_text:
                carry_label = ""
                if "IDENTIFICATION" in right_text.upper() or "ID CARD" in right_text.upper():
                    out_lines.append(left_text + " " + right_text)
                else:
                    out_lines.append(f"{left_text} : {right_text}")
                continue
            if left_text and not right_text:
                carry_label = left_text
                out_lines.append(left_text)
                continue
            if right_text and not left_text:
                if carry_label:
                    out_lines.append(f"{carry_label} : {right_text}")
                    carry_label = ""
                else:
                    out_lines.append(right_text)
                continue
            if raw:
                out_lines.append(raw)
        return "\n".join(out_lines)