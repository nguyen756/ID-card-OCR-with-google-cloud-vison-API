#docAI.py
from typing import Dict, List, Tuple
import io, os, re
from PIL import Image
from google.api_core.client_options import ClientOptions
from google.cloud import documentai as docai

def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", s)

class BaseDocAIParser:
    """Base class to call Document AI and extract key‑value fields and tables.
    Subclasses set a field map in __init__.
    """
    norm_field_map: Dict[str, str] = {}

    def __init__(self, processor_id_env_var: str):
        project_id = os.getenv("DOC_AI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("DOC_AI_LOCATION", "us")
        processor_id = os.getenv(processor_id_env_var)
        if not processor_id:
            raise RuntimeError(f"Missing env var {processor_id_env_var}")
        endpoint = f"{location}-documentai.googleapis.com"
        self.client = docai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )
        self.name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    # ---- helpers ----
    def _img_bytes(self, image: Image.Image) -> bytes:
        buf = io.BytesIO(); image.save(buf, format="PNG"); return buf.getvalue()

    def _layout_text(self, document, layout) -> str:
        if not layout or not getattr(layout, "text_anchor", None):
            return ""
        text = document.text or ""
        parts = []
        segs = getattr(layout.text_anchor, "text_segments", None) or []
        for seg in segs:
            start = int(getattr(seg, "start_index", 0) or 0)
            end = int(getattr(seg, "end_index", 0) or 0)
            if end > start:
                parts.append(text[start:end])
        return "".join(parts).strip()

    def _tables(self, document) -> List[Dict]:
        tables_out: List[Dict] = []
        pages = getattr(document, "pages", None) or []
        for page in pages:
            tbls = getattr(page, "tables", None) or []
            for t in tbls:
                headers: List[str] = []
                if t.header_rows:
                    hdr_row = t.header_rows[0]
                    headers = [self._layout_text(document, cell.layout) for cell in hdr_row.cells]
                rows: List[List[str]] = []
                for brow in (t.body_rows or []):
                    rows.append([self._layout_text(document, cell.layout) for cell in brow.cells])
                tables_out.append({"headers": headers, "rows": rows})
        return tables_out

    def _extract_fields_and_tables(self, doc):
        fields: Dict[str, str] = {}
        ents = getattr(doc, "entities", None) or []
        if ents:
            def put(k_raw: str, v: str):
                k = _norm(k_raw)
                if not k or not v: return
                canon = self.norm_field_map.get(k)
                val = v.strip()
                if not canon:
                    fields.setdefault("_extras", {});
                    fields["_extras"][k] = val; return
                existing = fields.get(canon)
                if not existing:
                    fields[canon] = val
                else:
                    if isinstance(existing, list):
                        if val not in existing: existing.append(val)
                    elif val != existing:
                        fields[canon] = [existing, val]
            for ent in ents:
                if getattr(ent, "type_", None) and getattr(ent, "mention_text", None):
                    put(ent.type_, ent.mention_text)
                for p in getattr(ent, "properties", None) or []:
                    if getattr(p, "type_", None) and getattr(p, "mention_text", None):
                        put(p.type_, p.mention_text)
        tables = self._tables(doc)
        return fields, tables

    def extract(self, pil_img: Image.Image) -> Tuple[Dict[str, str], List[Dict]]:
        raw = self._img_bytes(pil_img)
        req = docai.ProcessRequest(
            name=self.name,
            raw_document=docai.RawDocument(content=raw, mime_type="image/png"),
        )
        result = self.client.process_document(request=req)
        return self._extract_fields_and_tables(result.document)

    def extract_with_text(self, pil_img: Image.Image):
        raw = self._img_bytes(pil_img)
        req = docai.ProcessRequest(
            name=self.name,
            raw_document=docai.RawDocument(content=raw, mime_type="image/png"),
        )
        result = self.client.process_document(request=req)
        fields, tables = self._extract_fields_and_tables(result.document)
        fulltext = result.document.text or ""
        return fields, tables, fulltext

# ---- Student ID parser ----
STUDENT_ID_FIELD_MAP = {
    "student_name": "name",
    "student id": "adm_no",
    "student_id": "adm_no",
    "date_of_birth": "dob",
    "university_name": "school",
    "major": "major",
    "school_year": "school_year",
    "bank_name": "bank_name",
    "bank_card_number": "bank_card_number",
    "bank_card_valid_from": "bank_card_valid_from",
    "bank_card_expiry": "bank_card_expiry",
}

class StudentIdParser(BaseDocAIParser):
    def __init__(self):
        super().__init__(processor_id_env_var="DOC_AI_PROCESSOR_ID")
        self.norm_field_map = { _norm(k): v for k, v in STUDENT_ID_FIELD_MAP.items() }

# ---- Receipt parser (pretrained expense) ----
RECEIPT_FIELD_MAP = {
    "supplier_name": "supplier_name",
    "supplier_phone": "supplier_phone",
    "total_amount": "total_amount",
    "currency": "currency",
    "payment_type": "payment_type",
    "net_amount": "net_amount",
    "total_tax_amount": "total_tax_amount",
    "receipt_date": "receipt_date",
    "purchase_time": "purchase_time",
}

class ReceiptParser(BaseDocAIParser):
    def __init__(self):
        super().__init__(processor_id_env_var="RECEIPT_PROCESSOR_ID")
        self.norm_field_map = { _norm(k): v for k, v in RECEIPT_FIELD_MAP.items() }