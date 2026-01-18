import frappe
import json
import pytesseract
import re
import requests
import difflib
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from frappe.utils.file_manager import get_file_path
from frappe.model.document import Document
from PIL import Image, ImageOps, ImageFilter
from frappe.utils import nowdate
import string

def preprocess_image(img):
    gray = img.convert("L")
    enhanced = gray.filter(ImageFilter.SHARPEN)
    inverted = ImageOps.invert(enhanced)
    binarized = inverted.point(lambda x: 0 if x < 180 else 255, '1')
    return binarized

def get_ocr_lang():
    try:
        langs = pytesseract.get_languages(config="")
    except Exception:
        return "eng"
    if "eng" in langs and "ara" in langs:
        return "eng+ara"
    if "ara" in langs:
        return "ara"
    return "eng"

def clean_ocr_text(raw_text, allow_unicode=False):
    if allow_unicode:
        return raw_text
    return ''.join(c for c in raw_text if c in string.printable)

def extract_item_code(text):
    if not text:
        return ""
    cleaned = re.sub(r"^\s*\d+\s*[-.)]?\s+", "", text.strip())
    cleaned = re.sub(r"^\s*\d+(?=[A-Za-z])", "", cleaned)
    m = re.match(r"([A-Za-z0-9][A-Za-z0-9./-]{2,})", cleaned)
    if not m:
        return ""
    code = m.group(1).strip(" -:")
    if code.isdigit():
        return ""
    return code

def extract_text_from_pdf(file_path):
    text = ""
    ocr_lang = get_ocr_lang()
    allow_unicode = "ara" in ocr_lang

    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        pass

    if not text.strip():
        images = convert_from_path(file_path, dpi=300)
        for img in images:
            processed_img = preprocess_image(img)
            raw_text = pytesseract.image_to_string(
                processed_img, lang=ocr_lang, config="--psm 6"
            )
            clean_text = clean_ocr_text(raw_text, allow_unicode=allow_unicode)
            text += clean_text + "\n"

    return text


class InvoiceUpload(Document):
    def get_gemini_config(self):
        api_key = None
        model = None
        if frappe.db.exists("DocType", "Invoice OCR Settings"):
            api_key = frappe.db.get_single_value("Invoice OCR Settings", "gemini_api_key") or api_key
            model = frappe.db.get_single_value("Invoice OCR Settings", "gemini_model") or model
        if frappe.db.exists("DocType", "DoppioBot Settings"):
            api_key = frappe.db.get_single_value("DoppioBot Settings", "gemini_api_key") or api_key
            model = frappe.db.get_single_value("DoppioBot Settings", "gemini_model") or model
        api_key = api_key or frappe.conf.get("gemini_api_key")
        model = model or frappe.conf.get("gemini_model") or "gemini-2.0-flash"
        return api_key, model

    def extract_with_gemini(self, text):
        api_key, model = self.get_gemini_config()
        if not api_key:
            return None

        prompt = (
            "Extract invoice data from the OCR text below. "
            "Return ONLY valid JSON with keys: "
            "party_name, invoice_no, date, currency, total, items. "
            "items is a list of objects with keys: description, qty, rate, amount, item_code. "
            "Use numbers for qty/rate/amount when possible, otherwise null. "
            "If a value is missing, use null or empty string. "
            "Do not include any markdown or explanation.\n\n"
            f"OCR_TEXT:\n{text}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
        }
        try:
            response = requests.post(url, json=payload, timeout=40)
            response.raise_for_status()
        except Exception:
            frappe.log_error(frappe.get_traceback(), "Gemini API request failed")
            return None

        try:
            data = response.json()
            parts = data.get("candidates", [])[0].get("content", {}).get("parts", [])
            text_out = parts[0].get("text", "") if parts else ""
        except Exception:
            frappe.log_error(frappe.get_traceback(), "Gemini API response parse failed")
            return None

        parsed = self.parse_json_block(text_out)
        return parsed

    def parse_json_block(self, text):
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None

    def parse_number(self, value):
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = re.sub(r"[^\d.\-]", "", str(value))
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except Exception:
            return None

    def normalize_items(self, items):
        normalized = []
        for item in items or []:
            description = (item.get("description") or "").strip()
            if not description:
                continue
            qty = self.parse_number(item.get("qty"))
            rate = self.parse_number(item.get("rate"))
            item_code = (item.get("item_code") or "").strip()
            normalized.append({
                "description": description,
                "item_code": item_code,
                "qty": qty or 1,
                "rate": rate or 0,
            })
        return normalized

    def extract_items(self, text):
        lines = text.splitlines()
        items = []
        description_buffer = []
        skip_keywords = (
            "total", "tax", "net", "payable", "amount due", "invoice", "voucher",
            "supplier", "address", "contact", "mobile", "creditors", "date",
            "vat number", "cr number", "iban", "account", "bank", "tel", "fax",
            "email", "amount in words", "balance due", "discount", "round off",
            "total amount", "vat %", "nature of goods", "services", "u/price",
            "qty", "sl.no", "ext amt", "vat amt", "unit", "disc"
        )
        skip_locations = (
            "jubail", "ksa", "saudi", "arabia", "eastern province", "country",
            "city", "state", "street", "po.box", "po box", "building"
        )

        def looks_like_summary(line_text):
            lower = line_text.lower()
            if any(keyword in lower for keyword in skip_keywords):
                return True
            if any(keyword in lower for keyword in skip_locations):
                return True
            if re.search(r"\b(st|street|road|rd|avenue|ave)\b", lower):
                return True
            if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lower):
                return True
            if re.search(r"\b[a-z]{2,}-[a-z]{2,}-\d{4,}\b", lower, re.IGNORECASE):
                return True
            return False

        def normalize_description(line_text):
            cleaned = re.sub(r"^\s*\d+\s*[-.)]?\s+", "", line_text)
            cleaned = re.sub(r"^\s*\d+(?=[A-Za-z])", "", cleaned)
            cleaned = re.sub(r"\b(rs|pkr|usd|amount)\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\b(no|nos|qty|quantity)\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\bnone\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\bpn\b\s+tey\s+relates\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\btey\s+relates\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^[A-Za-z]{1,2}\b\s*", "", cleaned)
            cleaned = cleaned.replace("}", " ")
            return re.sub(r"\s{2,}", " ", cleaned).strip(" -:|")

        def parse_item_line(line_text, preferred_desc=None):
            working = re.sub(r"[|]+", " ", line_text)
            working = re.sub(r"\s{2,}", " ", working).strip()
            working = re.sub(r"^\d+\s*[-.)]?\s+", "", working)
            working = re.sub(r"^\d+(?=[A-Za-z])", "", working)

            uom_pattern = r"(?:Pcs|PCS|EA|Pair|PAIR|Pkt|PKT|KG|Kg|Nos|NOS)"
            qty_uom_attached = re.search(
                rf"(\d+(?:\.\d+)?)\s*({uom_pattern})\b",
                working,
                flags=re.IGNORECASE,
            )
            if qty_uom_attached:
                qty = float(qty_uom_attached.group(1))
                before = working[:qty_uom_attached.start()].strip()
                number_matches = list(re.finditer(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", working[qty_uom_attached.end():]))
                if number_matches:
                    rate_str = number_matches[0].group(1)
                    rate = float(rate_str.replace(",", ""))
                    raw_desc = before
                    if preferred_desc:
                        desc = normalize_description(preferred_desc)
                        code = extract_item_code(preferred_desc)
                    else:
                        desc = normalize_description(raw_desc)
                        code = extract_item_code(raw_desc)
                    return desc, code, qty, rate
            qty_after_uom = re.search(rf"{uom_pattern}\s*(\d+(?:\.\d+)?)\b", working)
            if qty_after_uom:
                qty = float(qty_after_uom.group(1))
                before = working[:qty_after_uom.start()].strip()
                number_matches = list(re.finditer(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", before))
                if number_matches:
                    rate_str = number_matches[-1].group(1)
                    rate = float(rate_str.replace(",", ""))
                    raw_desc = before[:number_matches[-1].start()].strip()
                    if preferred_desc:
                        desc = normalize_description(preferred_desc)
                        code = extract_item_code(preferred_desc)
                    else:
                        desc = normalize_description(raw_desc)
                        code = extract_item_code(raw_desc)
                    return desc, code, qty, rate

            qty_uom_matches = list(
                re.finditer(
                    rf"(\d+(?:\.\d+)?)\s*{uom_pattern}\b(?=\s+\d)",
                    working
                )
            )
            if qty_uom_matches:
                match = qty_uom_matches[-1]
                qty = float(match.group(1))
                after = working[match.end():]
                rate_match = re.search(r"\d+(?:\.\d+)?", after)
                if rate_match:
                    rate = float(rate_match.group(0))
                    raw_desc = working[:match.start()]
                    if preferred_desc:
                        desc = normalize_description(preferred_desc)
                        code = extract_item_code(preferred_desc)
                    else:
                        desc = normalize_description(raw_desc)
                        code = extract_item_code(raw_desc)
                    return desc, code, qty, rate

            tail_match = re.search(rf"(\d+(?:\.\d+)?)\s*{uom_pattern}\s*$", working)
            if tail_match:
                qty = float(tail_match.group(1))
                before = working[:tail_match.start()].strip()
                money_match = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*$", before)
                if money_match:
                    rate_str = money_match.group(1)
                    rate = float(rate_str.replace(",", ""))
                    raw_desc = before[:money_match.start()].strip()
                    if preferred_desc:
                        desc = normalize_description(preferred_desc)
                        code = extract_item_code(preferred_desc)
                    else:
                        desc = normalize_description(raw_desc)
                        code = extract_item_code(raw_desc)
                    return desc, code, qty, rate

            tokens = working.split()
            if tokens:
                qty_idx = None
                qty_val = None
                for idx, token in enumerate(tokens):
                    if re.search(r"[A-Za-z]", token):
                        continue
                    if "/" in token:
                        continue
                    qty_candidate = self.parse_number(token)
                    if qty_candidate is None:
                        continue
                    if token.endswith("%"):
                        continue
                    if "/" in token and "'" in token:
                        continue
                    if token.endswith("%"):
                        continue
                    qty_idx = idx
                    qty_val = qty_candidate
                    break
                if qty_idx is not None:
                    desc_tokens = tokens[:qty_idx]
                    raw_desc = " ".join(desc_tokens).strip()
                    rate_val = None
                    for token in tokens[qty_idx + 1:]:
                        if re.search(r"\d+\.\d+", token):
                            rate_val = self.parse_number(token)
                            if rate_val is not None:
                                break
                    if rate_val is None:
                        for token in tokens[qty_idx + 1:]:
                            rate_val = self.parse_number(token)
                            if rate_val is not None:
                                break
                    if raw_desc and qty_val is not None and rate_val is not None:
                        if preferred_desc:
                            desc = normalize_description(preferred_desc)
                            code = extract_item_code(preferred_desc)
                        else:
                            desc = normalize_description(raw_desc)
                            code = extract_item_code(raw_desc)
                        return desc, code, float(qty_val), float(rate_val)

            return None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if looks_like_summary(stripped):
                description_buffer = []
                continue

            candidate = " ".join(description_buffer + [stripped]).strip() if description_buffer else stripped
            preferred_desc = None
            if description_buffer:
                for buffered in reversed(description_buffer):
                    if re.search(r"[A-Za-z]", buffered):
                        preferred_desc = buffered
                        break
            parsed = parse_item_line(candidate, preferred_desc=preferred_desc)
            if parsed:
                line_desc, item_code, qty, rate = parsed
                if qty > 10000 or rate > 100000000 or len(line_desc) < 4:
                    description_buffer = []
                    continue
                if not re.search(r"[A-Za-z]", line_desc):
                    description_buffer = []
                    continue
                items.append({
                    "description": line_desc,
                    "item_code": item_code,
                    "qty": qty,
                    "rate": rate
                })
                description_buffer = []
            else:
                description_buffer.append(stripped)
                if len(description_buffer) > 3:
                    description_buffer.pop(0)

        return items

    def extract_party_from_text(self, text):
        def clean_party_line(value):
            value = re.sub(r"\b(name|customer|buyer|seller|details)\b", " ", value, flags=re.IGNORECASE)
            value = re.sub(r"\s{2,}", " ", value).strip(" -:")
            return value

        def looks_like_company(value):
            lower = value.lower()
            if not re.search(r"[A-Za-z]", value):
                return False
            if any(word in lower for word in ("jubail", "ksa", "saudi", "arabia", "eastern", "province", "country", "city", "street", "po.box", "po box", "building")):
                return False
            if re.search(r"\b(st|street|road|rd|avenue|ave)\b", lower):
                return False
            if any(word in lower for word in ("nature of goods", "servicesالجموع", "u/price", "qty", "sl.no", "ext amt")):
                return False
            if any(word in lower for word in ("account name", "bank", "iban", "branch", "account no", "vat number", "cr number", "tax id")):
                return False
            if any(word in lower for word in ("ltd", "limited", "llc", "est", "systems", "supply", "services", "company", "co.", "m/s")):
                return True
            return len(value.split()) >= 3

        def looks_like_supplier(value):
            lower = value.lower()
            return any(word in lower for word in ("supply", "services", "est", "trading", "factory"))

        def extract_english_company(value):
            match = re.search(r"([A-Za-z][A-Za-z\s&.\-]{4,})", value)
            if match:
                cleaned = match.group(1).strip(" -:")
                if cleaned.lower().endswith(" est"):
                    return cleaned + "."
                return cleaned
            return value

        def extract_ms_company(value):
            match = re.search(r"\b(M/S|MS)\s*([A-Za-z][A-Za-z\s&.\-]{4,})", value)
            if match:
                return f"M/S {match.group(2).strip(' -:')}"
            return ""

        def score_company(value):
            lower = value.lower()
            score = 0
            if looks_like_supplier(value):
                score += 4
            if any(word in lower for word in ("ltd", "limited", "llc", "est", "systems", "supply", "services", "company", "co.", "m/s", "trading")):
                score += 3
            if len(value.split()) >= 3:
                score += 1
            if any(word in lower for word in ("jubail", "ksa", "saudi", "arabia", "eastern", "province", "country", "city", "street", "po.box", "po box", "building")):
                score -= 5
            if "global heritage" in lower and self.party_type == "Supplier":
                score -= 5
            return score

        def pick_best(lines_subset):
            best = None
            best_score = -999
            for line in lines_subset:
                candidate = clean_party_line(line.strip())
                if not looks_like_company(candidate):
                    continue
                score = score_company(candidate)
                if score > best_score:
                    best_score = score
                    best = candidate
            if best:
                return extract_english_company(best)
            return ""

        lines = text.splitlines()
        reversed_lines = list(reversed(lines))
        if self.party_type == "Customer":
            for line in lines:
                if "to:" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else ""
                    value = clean_party_line(value)
                    if looks_like_company(value):
                        return extract_english_company(value)
            for line in lines:
                candidate = extract_ms_company(line)
                if candidate:
                    return candidate
            for line in lines:
                if "customer details" in line.lower():
                    continue
                if "tax id" in line.lower():
                    continue
                candidate = clean_party_line(line.strip())
                if looks_like_company(candidate):
                    return extract_english_company(candidate)
            return ""
        if self.party_type == "Supplier":
            for line in lines:
                lower = line.lower()
                if "account name" in lower and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    value = clean_party_line(value)
                    if looks_like_company(value):
                        return extract_english_company(value)
            stop_keys = ("customer details", "buyer details", "to:")
            supplier_block = []
            for line in lines:
                lower = line.lower().strip()
                if any(key in lower for key in stop_keys):
                    break
                supplier_block.append(line)
            picked = pick_best(supplier_block)
            if picked:
                return picked
            for line in reversed_lines:
                lower = line.lower()
                if "global heritage" in lower or "to:" in lower:
                    continue
                candidate = clean_party_line(line.strip())
                if looks_like_supplier(candidate):
                    return extract_english_company(candidate)
            for line in lines:
                candidate = clean_party_line(line.strip())
                if looks_like_company(candidate) and looks_like_supplier(candidate):
                    return extract_english_company(candidate)
            for line in lines:
                lower = line.lower()
                if "global heritage" in lower:
                    continue
                if "buyer" in lower or "customer" in lower:
                    continue
                if "to:" in lower or lower.startswith("to"):
                    continue
                candidate = clean_party_line(line.strip())
                if looks_like_supplier(candidate):
                    return extract_english_company(candidate)

        seller_window = 0
        buyer_window = 0
        for i, line in enumerate(lines):
            lower = line.lower()
            if "seller details" in lower or "المورد" in lower:
                seller_window = 6
            if "buyer details" in lower or "customer details" in lower:
                buyer_window = 6
            if "account name" in lower and ":" in line:
                value = line.split(":", 1)[1].strip()
                value = clean_party_line(value)
                if looks_like_company(value):
                    return value
            if lower.strip().startswith("to:") or lower.strip().startswith("to :"):
                value = line.split(":", 1)[1].strip() if ":" in line else ""
                value = clean_party_line(value)
                if looks_like_company(value):
                    return value
            if "supplier" in lower or "seller" in lower:
                if ":" in line:
                    value = line.split(":", 1)[1].strip()
                else:
                    value = ""
                value = clean_party_line(value)
                if looks_like_company(value):
                    return value
            if seller_window:
                cand = clean_party_line(line.strip())
                if looks_like_company(cand) and looks_like_supplier(cand):
                    return cand
                seller_window -= 1
            if buyer_window:
                cand = clean_party_line(line.strip())
                if looks_like_company(cand):
                    return cand
                buyer_window -= 1
            candidate = clean_party_line(line.strip())
            if looks_like_company(candidate):
                return candidate
        return ""

    def get_default_item_group(self):
        if frappe.db.exists("Item Group", "All Item Groups"):
            return "All Item Groups"
        return (
            frappe.db.get_value("Item Group", {"is_group": 0}, "name")
            or frappe.db.get_value("Item Group", {}, "name")
        )

    def get_default_uom(self):
        if frappe.db.exists("UOM", "Nos"):
            return "Nos"
        if frappe.db.exists("UOM", "Unit"):
            return "Unit"
        return frappe.db.get_value("UOM", {}, "name")

    def get_or_create_item(self, description):
        description = (description or "").strip()
        if len(description) < 3:
            return ""
        existing = frappe.db.get_value("Item", {"item_name": description}, "name")
        if existing:
            if frappe.db.get_value("Item", existing, "disabled"):
                frappe.db.set_value("Item", existing, "disabled", 0)
            return existing

        base = re.sub(r"[^A-Za-z0-9]+", " ", description).strip()
        if not base:
            base = "OCR Item"
        base = base[:120].strip()
        item_code = base
        counter = 1
        while frappe.db.exists("Item", item_code):
            counter += 1
            item_code = f"{base}-{counter}"

        item = frappe.new_doc("Item")
        item.item_code = item_code
        item.item_name = description
        item.item_group = self.get_default_item_group()
        item.stock_uom = self.get_default_uom()
        item.is_stock_item = 0
        item.disabled = 0
        item.insert(ignore_permissions=True)
        return item.name



    def extract_invoice(self):
        if not self.file:
            frappe.throw("No file attached.")

        file_path = get_file_path(self.file)
        text = extract_text_from_pdf(file_path)

        if not text.strip():
            frappe.msgprint("⚠️ No OCR text found.")
            return {"added_rows": []}

        # ✅ Debug OCR preview
        frappe.msgprint(f"<pre>{text[:4000]}</pre>")

        text_for_llm = text[:20000]
        gemini_data = self.extract_with_gemini(text_for_llm)

        extraction_source = "ocr"
        if gemini_data:
            extraction_source = "gemini"
            items = self.normalize_items(gemini_data.get("items", []))
            fallback_items = self.extract_items(text)
            if fallback_items and len(fallback_items) > len(items):
                items = fallback_items
                extraction_source = "ocr"
            if not self.party_type:
                lower_text = text.lower()
                if "customer" in lower_text and "supplier" not in lower_text:
                    self.party_type = "Customer"
                elif "supplier" in lower_text:
                    self.party_type = "Supplier"
                else:
                    self.party_type = "Supplier"
            if not self.party:
                extracted_party = (gemini_data.get("party_name") or "").strip()
                if extracted_party:
                    self.party = extracted_party
        else:
            items = self.extract_items(text)
            if not self.party:
                extracted_party = self.extract_party_from_text(text)
                if extracted_party:
                    self.party = extracted_party
        if not self.party:
            extracted_party = self.extract_party_from_text(text)
            if extracted_party:
                self.party = extracted_party
        if self.party and self.party_type in ("Customer", "Supplier"):
            if not frappe.db.exists(self.party_type, self.party):
                party_doc = frappe.new_doc(self.party_type)
                if self.party_type == "Customer":
                    party_doc.customer_name = self.party
                    party_doc.customer_type = "Company"
                else:
                    party_doc.supplier_name = self.party
                    party_doc.supplier_type = "Company"
                party_doc.insert(ignore_permissions=True)
        extracted_data = {
            "items": items,
            "party": self.party,
        }
        if gemini_data:
            extracted_data.update({
                "invoice_no": gemini_data.get("invoice_no"),
                "date": gemini_data.get("date"),
                "total": gemini_data.get("total"),
                "currency": gemini_data.get("currency"),
                "extraction_source": extraction_source,
            })
        else:
            extracted_data["extraction_source"] = extraction_source

        def find_item_match(row):
            item_code = (row.get("item_code") or "").strip()
            if item_code:
                exact = frappe.db.get_value("Item", {"item_code": item_code})
                if exact:
                    return exact
                normalized = item_code.replace("/", "-")
                if normalized != item_code:
                    normalized_match = frappe.db.get_value("Item", {"item_code": normalized})
                    if normalized_match:
                        return normalized_match
                by_name = frappe.db.get_value("Item", {"item_name": item_code})
                if by_name:
                    return by_name
            if row.get("description"):
                return frappe.db.get_value("Item", {"item_name": row["description"]})
            return None

        def find_fuzzy_match(description):
            if not description:
                return None, 0
            tokens = re.findall(r"[A-Za-z0-9]+", description.lower())
            if not tokens:
                return None, 0
            seed = tokens[0]
            min_score = 0.82
            candidates = frappe.get_all(
                "Item",
                filters={"item_name": ["like", f"%{seed}%"]},
                fields=["name", "item_name"],
                limit=50,
            )
            best_name = None
            best_score = 0
            for row in candidates:
                score = difflib.SequenceMatcher(
                    None, description.lower(), (row.get("item_name") or "").lower()
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_name = row.get("name")
            if best_score < min_score:
                return None, best_score
            return best_name, best_score

        # ✅ Update child table cleanly
        self.set("invoice_upload_item", [])
        added_rows = []
        for row in items:
            matched_item = find_item_match(row)
            match_score = 0
            if not matched_item:
                matched_item, match_score = find_fuzzy_match(row.get("description"))
            if not matched_item:
                matched_item = self.get_or_create_item(row.get("description"))
            self.append("invoice_upload_item", {
                "ocr_description": row["description"][:140],
                "qty": row["qty"],
                "rate": row["rate"],
                "item": matched_item
            })
            added_rows.append({
                "item_code": row.get("item_code"),
                "matched_item": matched_item,
                "match_score": round(match_score, 3) if match_score else 0,
                "qty": row.get("qty"),
                "rate": row.get("rate"),
                "description": row.get("description"),
            })

        # ✅ Correctly set JSON field AFTER extraction
        self.extracted_data = json.dumps(extracted_data, indent=2)
        self.ocr_status = "Extracted"
        self.save()

        if not items:
            frappe.msgprint("⚠️ OCR done but no items found. Please check PDF format.")
        else:
            frappe.msgprint("✅ OCR Extraction completed.")
        return {"added_rows": added_rows, "items_count": len(items)}


@frappe.whitelist()
def extract_invoice(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    return doc.extract_invoice()
