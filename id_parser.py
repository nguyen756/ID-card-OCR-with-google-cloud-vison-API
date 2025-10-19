#id_parser.py
import re
from typing import Dict, List, Tuple

# ---- Label detection ----
LABEL_MAP: List[Tuple[str, str]] = [
    ("name",      r"name|student\s*name"),
    ("dob",       r"d(?:\.?\s*o\.?\s*b\.?|ate\s*of\s*birth)"),
    ("adm_no",    r"(?:admission\s*no\.?|adm\s*no\.?|student\s*id|id)"),
    ("phone",     r"(?:phone|tel)"),
    ("website",   r"website"),
    ("email",     r"(?:email|e-?mail)"),
    ("social",    r"(?:social(?:\s*media)?|socialmedia)"),
    ("address",   r"(?:address|addr)"),
    ("school",    r"(?:school|college|university)"),
    ("card_type", r"(?:student\s*id\s*card|id\s*card|.*identification\s*card)"),
]
LABEL_RE = {k: re.compile(rf"^\s*(?:{pat})\b\s*: ?\s*", re.IGNORECASE) for k, pat in LABEL_MAP}

BLACKLIST = re.compile(r"\b(getty|istock|shutterstock|depositphotos|adobe\s*stock|pixabay|pexels)\b", re.IGNORECASE)

def _clean_spaces(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s.strip())
    s = s.replace(" ,", ",").replace(" .", ".")
    return s.strip()

def _normalize_phone(s: str) -> str:
    s = re.sub(r"[^\d\-().\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*-\s*", "-", s)
    return s

def _normalize_site(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^WWW\b", "www", s, flags=re.IGNORECASE)
    s = re.sub(r"^www(?!\.)", "www.", s)
    parts = s.split()
    if len(parts) == 2 and parts[0].lower().startswith("www"):
        s = parts[0] + "." + parts[1].lstrip(".")
    m = re.match(r"(?:(https?)://)?([^/\s]+)(/.*)?$", s)
    if m:
        scheme = (m.group(1) or "").lower()
        domain = (m.group(2) or "").lower()
        path = m.group(3) or ""
        s = (scheme + "://" if scheme else "") + domain + path
    return s

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)

def _is_date(s: str) -> bool:
    t = s.strip()
    return bool(_DATE_RE.fullmatch(t) or _DATE_RE.search(t))


def tidy_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln == ":" and out:
            nxt = lines[i+1].strip() if i + 1 < len(lines) else ""
            out[-1] = out[-1].rstrip(" :") + " : " + nxt
            i += 2
            continue
        ln = re.sub(r"\s*:\s*", " : ", ln)
        if out and ln.startswith(":"):
            out[-1] = out[-1].rstrip(" :") + " : " + ln[1:].strip()
        else:
            out.append(ln)
        i += 1
    return "\n".join(out)

def _merge_header_school(lines: List[str]) -> str:
    header = []
    seen_card = False
    for ln in lines:
        if re.search(r"(IDENTIFICATION\s*CARD|ID\s*CARD)", ln, re.IGNORECASE):
            seen_card = True
            break
        if ln.isupper() and re.search(r"[A-Z]", ln):
            header.append(ln)
        else:
            header = []  # reset if interrupted
    if header:
        s = " ".join(header).strip()
        m = re.search(r"(.*\b(?:SCHOOL|COLLEGE|UNIVERSITY)\b)", s, re.IGNORECASE)
        return m.group(1).strip() if m else s
    return ""


def parse_id_fields(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in raw_lines if not BLACKLIST.search(ln)]
    N = len(lines)

    for ln in lines:
        if re.search(r"(IDENTIFICATION\s*CARD|ID\s*CARD)", ln, re.IGNORECASE):
            out.setdefault("card_type", "STUDENT ID CARD")
            break
    school_hdr = _merge_header_school(lines)
    if school_hdr:
        out["school"] = school_hdr

    pending: List[str] = []
    values: List[str] = []
    i = 0
    while i < N:
        ln = _clean_spaces(lines[i])
        matched_key = None
        for key, rx in LABEL_RE.items():
            m = rx.match(ln)
            if m:
                matched_key = key
                val = _clean_spaces(ln[m.end():])
                if val:
                    if key == "website": val = _normalize_site(val)
                    if key == "phone":   val = _normalize_phone(val)
                    if key == "card_type": val = "STUDENT ID CARD" if "card" in val.lower() else val.upper()
                    out[key] = val
                else:
                    pending.append(key)
                break
        if not matched_key:
            values.append(ln)
        i += 1

    clean_values = [v for v in values if not any(rx.match(v) for rx in LABEL_RE.values()) and v != ":"]
    vi = 0
    for key in pending:
        if vi >= len(clean_values):
            break
        v = clean_values[vi].strip()
        if key == "dob" and not _is_date(v) and vi + 1 < len(clean_values) and _is_date(clean_values[vi+1]):
            vi += 1
            v = clean_values[vi].strip()
        if key == "website":
            out[key] = _normalize_site(v)
        elif key == "phone":
            out[key] = _normalize_phone(v)
        elif key == "address":
            v = re.sub(r",\s*", ", ", v)
            v = re.sub(r"\.(?=\S)", ". ", v)
            out[key] = re.sub(r"\s{2,}", " ", v).strip(" .,")
        elif key == "dob":
            if _is_date(v):
                out[key] = v
        elif key == "adm_no":
            out[key] = v
        elif key == "card_type":
            out[key] = "STUDENT ID CARD"
        else:
            out[key] = v
        vi += 1

    blob = " ".join(lines)
    if "email" not in out:
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", blob)
        if m: out["email"] = m.group(0)
    if "website" not in out:
        m = (
            re.search(r"(?:https?://)?(?:www\.|WWW)[A-Za-z0-9\-.]+\.[A-Za-z]{2,}(?:/[^\s]*)?", blob)
            or re.search(r"(?:https?://)?[A-Za-z0-9\-.]+\.[A-Za-z]{2,}(?:/[^\s]*)?", blob)
        )
        if m: out["website"] = _normalize_site(m.group(0))
    if "dob" not in out:
        m = _DATE_RE.search(blob)
        if m: out["dob"] = m.group(1)
    if "adm_no" not in out:
        m_id = re.search(r"\b[0-9][0-9\s\-/.]{6,}\b", blob)
        if m_id:
            cand = m_id.group(0)
            if not cand.strip().startswith("+"):
                out["adm_no"] = cand

    if "phone" in out:   out["phone"]   = _normalize_phone(out["phone"])
    if "website" in out: out["website"] = _normalize_site(out["website"])
    if "card_type" in out: out["card_type"] = out["card_type"].upper()
    return out