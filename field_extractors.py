"""
field_extractors.py
This module groups different field extraction strategies into classes. Each
extractor implements an `extract_fields` method that accepts the OCR'd text
and returns a dictionary of parsed values. New extraction strategies can be
implemented by subclassing `BaseFieldExtractor`.
"""

import re
from typing import Dict, List


class BaseFieldExtractor:
    """Abstract base class for field extraction from OCR text."""

    def extract_fields(self, text: str) -> Dict[str, str]:
        """Extract structured fields from a block of OCR text."""
        raise NotImplementedError


class InvoiceFieldExtractor(BaseFieldExtractor):
    """
    Extract vendor, date, and total from OCR text representing an invoice.

    The extraction heuristics are designed for simple invoice layouts
    and may need to be extended for complex documents.
    """

    def extract_fields(self, text: str) -> Dict[str, str]:
        fields = {}
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # Vendor: first line that is all uppercase letters (and not purely numeric/punctuation)
        for ln in lines:
            if ln.upper() == ln and re.search(r"[A-Z]", ln) and not re.fullmatch(r"[\d\W]+", ln):
                fields["vendor"] = ln
                break

        # Date patterns (YYYY-MM-DD, DD/MM/YYYY, Month DD, YYYY, etc.)
        date_patterns = [
            r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",      # 2025-10-02
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",    # 02/10/2025 or 2-10-25
            r"\b([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b", # October 2, 2025
        ]
        for pattern in date_patterns:
            m = re.search(pattern, text)
            if m:
                fields["date"] = m.group(1)
                break

        # Total/Amount: find lines with keywords and numeric amounts
        amount_keywords = ["total", "amount due", "grand total", "balance"]
        amount_pattern = re.compile(r"([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)")
        for ln in lines:
            for kw in amount_keywords:
                if kw in ln.lower():
                    amt = amount_pattern.search(ln)
                    if amt:
                        fields["total"] = amt.group(1)
                    break
            if "total" in fields:
                break

        return fields


class StudentIDFieldExtractor(BaseFieldExtractor):
    """
    Extract structured fields from OCR text representing a student ID card.

    Fields extracted include name, date of birth (dob), admission number
    (adm_no), address, school, and card_type (e.g. "STUDENT ID CARD").
    The extraction logic uses label patterns to identify key-value pairs
    across lines and handles multi-line addresses and header merging.
    """

    # Define patterns for all known labels. Patterns are case-insensitive.
    LABEL_PATTERNS = {
        # Recognize names; student name variant
        "name": r"^(?:name|student\s*name)\b",
        # Date of birth or DOB (case-insensitive, with optional punctuation)
        "dob": r"^(?:d\.?o\.?b\.?|date\s*of\s*birth)\b",
        # Admission numbers and generic ID labels
        "adm_no": r"^(?:adm\s*no\.?|admn?o\.?|admission\s*no\.?|id)\b",
        # Phone labels; telephone abbreviations
        "phone": r"^(?:phone|tel)\b",
        # Website labels; do not match 'www' alone as a label
        "website": r"^(?:website)\b",
        # Email labels; accept both 'email' and 'e-mail'
        "email": r"^(?:email|e-mail)\b",
        # Social handle labels; optionally followed by 'media' on the same line
        "social": r"^(?:social|social\s*media)\b",
        # Address labels
        "address": r"^(?:address|addr)\b",
        # School names; includes school/college/university
        "school": r"^(?:school|college|university)\b",
        # Card type; e.g. Student ID Card
        "card_type": r"^(?:student\s*id\s*card|id\s*card)\b",
    }

    def _is_label(self, line: str) -> str:
        s = line.strip().lower()
        s = re.sub(r"\s+", " ", s)
        for key, pat in self.LABEL_PATTERNS.items():
            if re.match(pat, s):
                return key
        return ""

    def _after_colon_value(self, line: str) -> str:
        m = re.search(r"[:\-]\s*(.+)$", line)
        return m.group(1).strip() if m else ""

    def _clean_address(self, s: str) -> str:
        s = re.sub(r"\s{2,}", " ", s)
        s = s.replace(" ,", ",").replace(" .", ".")
        s = s.replace(".,", ".")
        return s.strip(" ,.")

    def _title_name(self, s: str) -> str:
        if s.isupper() and len(s) <= 4:
            return s
        return " ".join(w.capitalize() if w.isalpha() else w for w in s.split())

    def extract_fields(self, text: str) -> Dict[str, str]:
        """
        Extract structured fields from a student ID card OCR block.

        This implementation supports multi-line labels and multi-line values.
        It will automatically assign values for all configured labels, including
        name, date of birth (dob), admission number (adm_no), ID, phone, website,
        email, social handle, address, school name, and card type.

        Parameters
        ----------
        text : str
            The OCR text extracted from an ID card image.

        Returns
        -------
        Dict[str, str]
            A dictionary of extracted fields keyed by the canonical label names.
        """
        out: Dict[str, str] = {}
        # Split and clean lines, dropping obvious separators
        raw_lines: List[str] = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in raw_lines if ln and not re.fullmatch(r"[_\-–=~]+", ln)]

        # Heuristic to merge consecutive ALL-CAPS header lines (e.g. FAUGET HIGH / SCHOOL)
        # Only consider lines before the first recognized label as potential headers
        first_label_index = next((idx for idx, ln in enumerate(lines) if self._is_label(ln)), len(lines))
        header_candidates = lines[:first_label_index]
        all_caps = [ln for ln in header_candidates if ln.upper() == ln and re.search(r"[A-Z]", ln)]
        merged_caps: List[str] = []
        if all_caps:
            idx = 0
            while idx < len(all_caps):
                cur = all_caps[idx]
                nxt = all_caps[idx + 1] if idx + 1 < len(all_caps) else ""
                # Merge two consecutive ALL CAPS lines if the second is short (likely part of school name)
                if nxt and nxt.isupper() and re.fullmatch(r"[A-Z\s]+", nxt) and len(nxt) <= 12:
                    merged_caps.append((cur + " " + nxt).strip())
                    idx += 2
                else:
                    merged_caps.append(cur)
                    idx += 1
            # Assign school and card_type from headers if not yet found
            hdr_school = next(
                (h for h in merged_caps if re.search(r"\b(SCHOOL|COLLEGE|UNIVERSITY)\b", h)),
                "",
            )
            if hdr_school and "school" not in out:
                out["school"] = hdr_school
            hdr_card = next(
                (h for h in merged_caps if re.search(r"\bSTUDENT\s*ID\s*CARD\b", h)),
                "",
            )
            if hdr_card and "card_type" not in out:
                out["card_type"] = "STUDENT ID CARD"

        # Iterate through lines and extract key-value pairs
        i = 0
        N = len(lines)
        while i < N:
            line = lines[i]
            # Check if current line is a label
            key = self._is_label(line)

            # Handle multi-line labels (e.g., "Social" followed by "Media")
            if key:
                # Default: do not merge additional lines into the label unless explicitly handled
                label_line = line
                j = i
                # If the label is 'social' and the next line is 'media', join them to form 'social media'
                if key == "social":
                    while j + 1 < N and lines[j + 1].strip().lower() == "media":
                        label_line = (label_line + " " + lines[j + 1]).strip()
                        j += 1
                # Update key after potential merging
                key = self._is_label(label_line)
                # If lines were merged, adjust index to skip them
                i = j

                # Determine value associated with the label
                value = self._after_colon_value(label_line)
                # If no value on same line, gather subsequent lines until next label
                if not value:
                    k = i + 1
                    value_lines: List[str] = []
                    while k < N:
                        nxt = lines[k]
                        nxt_key = self._is_label(nxt)
                        # Stop if we encounter a new label (but allow merging of multi-word labels)
                        if nxt_key:
                            break
                        # Append non-empty lines that are not trivial punctuation
                        if nxt and not re.fullmatch(r"[^\w]*", nxt):
                            value_lines.append(nxt)
                        k += 1
                    # Join collected value lines with spaces
                    value = " ".join(value_lines).strip()
                    # Advance index to the last line consumed
                    i = k - 1

                # Normalize and store the extracted field
                if value:
                    norm_value = value.strip()
                    if key == "name":
                        out["name"] = self._title_name(norm_value)
                    elif key in ("dob", "date", "birth"):
                        out["dob"] = norm_value
                    elif key in ("adm_no", "admission_no", "id"):
                        # Normalize spaces/hyphens in IDs
                        out["adm_no"] = re.sub(r"\s{2,}", " ", norm_value)
                    elif key == "phone":
                        out["phone"] = norm_value
                    elif key == "website":
                        out["website"] = norm_value
                    elif key == "email":
                        out["email"] = norm_value
                    elif key == "social":
                        out["social"] = norm_value
                    elif key == "address":
                        out["address"] = self._clean_address(norm_value)
                    elif key == "school":
                        out["school"] = norm_value.upper()
                    elif key == "card_type":
                        out["card_type"] = norm_value.upper()

            i += 1

        # Fallbacks: infer missing DOB, ADM NO., and school from patterns
        if "dob" not in out:
            m = re.search(
                r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
                text,
            )
            if m:
                out["dob"] = m.group(1)
        if "adm_no" not in out:
            # Match patterns like 0123 4567 8901 or 123-456-7890
            m = re.search(
                r"\b(?:\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4})\b",
                text,
            )
            if m:
                out["adm_no"] = m.group(0).replace(" ", "")
        # If no school was found in the label sweep, use the first merged header if it exists
        if "school" not in out and merged_caps:
            out["school"] = merged_caps[0]

        # Final cleanup: ensure address formatting and remove trailing punctuation
        if "address" in out:
            out["address"] = self._clean_address(out["address"])

        return out
    
# field_extractors.py

class GenericCardFieldExtractor(StudentIDFieldExtractor):
    """Fallback extractor that captures any label-value pairs."""

    def extract_fields(self, text: str) -> Dict[str, str]:
        # Start with known fields from StudentIDFieldExtractor
        fields = super().extract_fields(text)

        # Generic fallback: capture unknown labels and values
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        i = 0
        N = len(lines)
        while i < N:
            line = lines[i]
            # Skip if already parsed as a known label
            if self._is_label(line):
                i += 1
                continue
            key = None
            value = None
            if ":" in line:
                lhs, rhs = line.split(":", 1)
                key = lhs.strip().lower().replace(" ", "_")
                value = rhs.strip() or (lines[i+1].strip() if i+1 < N else "")
                i += 1 if not rhs.strip() and i+1 < N else 0
            elif i + 1 < N and not self._is_label(lines[i+1]):
                key = line.lower().replace(" ", "_")
                value = lines[i+1].strip()
                i += 1
            if key and value and key not in fields:
                fields[key] = value
            i += 1
        return fields
