# docAI.py
from typing import Dict, List, Tuple
import io, os, re
from PIL import Image
from google.api_core.client_options import ClientOptions
from google.cloud import documentai as docai

def _norm(s: str) -> str:
    """Hàm chuẩn hóa chuỗi key (lowercase, bỏ gạch dưới, bỏ khoảng trắng thừa)."""
    s = (s or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", s)


# -----------------------------------------------------------------
# LỚP CƠ SỞ (BASE CLASS)
# -----------------------------------------------------------------

class BaseDocAIParser:
    """
    Lớp cơ sở để gọi Document AI và trích xuất các trường (fields) và bảng (tables).
    Lớp này được thiết kế để kế thừa, không sử dụng trực tiếp.
    """
    
    # Lớp con sẽ định nghĩa map này trong __init__ của nó
    norm_field_map: Dict[str, str] = {}

    def __init__(self, processor_id_env_var: str):
        """
        Khởi tạo client DocAI.
        :param processor_id_env_var: Tên của biến môi trường (env var) 
                                     chứa Processor ID (ví dụ: "DOC_AI_PROCESSOR_ID")
        """
        project_id = os.getenv("DOC_AI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("DOC_AI_LOCATION", "us")  # 'us' or 'eu'
        processor_id = os.getenv(processor_id_env_var) # THAY ĐỔI: Lấy ID từ tham số
        
        if not processor_id:
            raise RuntimeError(f"Biến môi trường {processor_id_env_var} chưa được đặt")

        endpoint = f"{location}-documentai.googleapis.com"
        self.client = docai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )
        self.name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    def _img_bytes(self, image: Image.Image) -> bytes:
        """Chuyển đổi PIL Image sang bytes."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _layout_text(self, document, layout) -> str:
        """Trích xuất text từ document dựa trên layout và text_anchor."""
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
        """Chuyển đổi DocAI tables sang cấu trúc list/dict đơn giản."""
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

    def extract(self, pil_img: Image.Image) -> Tuple[Dict[str, str], List[Dict]]:
        """Gửi ảnh đến DocAI và trích xuất fields và tables."""
        raw = self._img_bytes(pil_img)
        req = docai.ProcessRequest(
            name=self.name,
            raw_document=docai.RawDocument(content=raw, mime_type="image/png"),
        )
        result = self.client.process_document(request=req)
        doc = result.document

        # Trích xuất Key-Value fields
        fields: Dict[str, str] = {}
        ents = getattr(doc, "entities", None) or []
        if ents:
            def put(k_raw: str, v: str):
                """Hàm nội bộ để thêm field vào dict, sử dụng norm_field_map."""
                k = _norm(k_raw)
                if not k or not v:
                    return
                
                # THAY ĐỔI: Sử dụng self.norm_field_map thay vì NORM_FIELD_MAP toàn cục
                canon = self.norm_field_map.get(k)
                val = v.strip()
                
                if not canon:
                    # Nếu không có trong map, thêm vào _extras
                    fields.setdefault("_extras", {})
                    fields["_extras"][k] = val
                    return
                
                existing = fields.get(canon)
                if not existing:
                    fields[canon] = val
                else:
                    # Xử lý trường hợp một key có nhiều giá trị
                    if isinstance(existing, list):
                        if val not in existing:
                            existing.append(val)
                    else:
                        if val != existing:
                            fields[canon] = [existing, val]

            # Lặp qua các entities (thực thể) mà DocAI tìm thấy
            for ent in ents:
                if getattr(ent, "type_", None) and getattr(ent, "mention_text", None):
                    put(ent.type_, ent.mention_text)
                # Lặp qua các thuộc tính con (nested properties)
                for p in getattr(ent, "properties", None) or []:
                    if getattr(p, "type_", None) and getattr(p, "mention_text", None):
                        put(p.type_, p.mention_text)

        # Trích xuất Tables (dùng cho line_items)
        tables = self._tables(doc)
        return fields, tables


# -----------------------------------------------------------------
# PARSER CHO THẺ SINH VIÊN (CODE GỐC CỦA BẠN)
# -----------------------------------------------------------------

# Map DocAI keys -> tên trường chuẩn hóa cho Thẻ Sinh Viên
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
    """
    Trình phân tích (parser) cho Thẻ Sinh Viên.
    Sử dụng biến môi trường: DOC_AI_PROCESSOR_ID
    """
    def __init__(self):
        # Gọi __init__ của lớp cha, truyền vào tên biến env
        super().__init__(processor_id_env_var="DOC_AI_PROCESSOR_ID")
        
        # Đặt field map chuẩn hóa cho riêng parser này
        self.norm_field_map = { _norm(k): v for k, v in STUDENT_ID_FIELD_MAP.items() }


# -----------------------------------------------------------------
# PARSER MỚI CHO HÓA ĐƠN (RECEIPT)
# -----------------------------------------------------------------

# Map DocAI keys -> tên trường chuẩn hóa cho Hóa Đơn
# Dựa trên các trường bạn cung cấp từ pretrained-expense-v1.4.2
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
    
    # LƯU Ý: Các trường 'line_item' (như description, amount, quantity)
    # thường được trích xuất dưới dạng BẢNG (table).
    # Phương thức _tables() của lớp cha đã xử lý việc này,
    # nên chúng ta không cần map chúng ở đây.
    # Kết quả của 'line_item' sẽ nằm trong biến 'tables' trả về từ hàm extract().
}

class ReceiptParser(BaseDocAIParser):
    """
    Trình phân tích (parser) cho Hóa Đơn (Receipts).
    Sử dụng biến môi trường: RECEIPT_PROCESSOR_ID
    """
    def __init__(self):
        # Gọi __init__ của lớp cha, truyền vào tên biến env MỚI
        super().__init__(processor_id_env_var="RECEIPT_PROCESSOR_ID")
        
        # Đặt field map chuẩn hóa cho riêng parser này
        self.norm_field_map = { _norm(k): v for k, v in RECEIPT_FIELD_MAP.items() }