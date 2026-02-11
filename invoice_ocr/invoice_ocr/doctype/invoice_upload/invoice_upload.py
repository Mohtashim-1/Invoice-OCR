import frappe
import json
import pytesseract
import re
import requests
import difflib
import base64
import io
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
    def _get_single_value_if_field_exists(self, doctype, fieldname):
        if not frappe.db.exists("DocType", doctype):
            return None
        try:
            meta = frappe.get_meta(doctype)
            if meta.has_field(fieldname):
                return frappe.db.get_single_value(doctype, fieldname)
        except Exception:
            return None
        return None

    def _build_invoice_prompt(self, text):
        return (
            "You are an invoice data extraction expert. "
            "Extract data from the attached invoice image(s). The invoice may have MULTIPLE PAGES - examine ALL pages carefully. "
            "OCR text below is noisy and should only be used as a secondary hint if image is unclear.\n\n"
            "Return ONLY valid JSON (no markdown, no explanation) with these keys:\n"
            "{\n"
            '  "supplier_name": "the supplier/vendor company name from the invoice header",\n'
            '  "invoice_no": "invoice number",\n'
            '  "date": "invoice date",\n'
            '  "currency": "currency code",\n'
            '  "total": total_amount_number,\n'
            '  "items": [\n'
            '    {"description": "product/item name", "qty": number, "uom": "unit of measure (e.g. PCS, EA, KG, NOS, Pair, Roll, Pkt, Set, Mtr)", "rate": unit_price_number}\n'
            "  ]\n"
            "}\n\n"
            "IMPORTANT RULES:\n"
            "- Extract ALL line items from ALL pages of the invoice\n"
            "- description = product/item name only, not codes or numbers\n"
            "- qty = quantity as a number\n"
            "- uom = unit of measure (PCS, EA, KG, NOS, Pair, Roll, Pkt, Set, Mtr, etc.)\n"
            "- rate = unit price per item as a number\n"
            "- Do NOT include totals, subtotals, VAT/tax rows, discounts, shipping, or summary lines\n"
            "- supplier_name = the company that issued/sold on the invoice (NOT the buyer/customer)\n"
            "- If a value is unclear, use null\n\n"
            f"OCR_TEXT_HINT:\n{text}"
        )

    def _clean_ocr_line(self, line):
        if not line:
            return ""
        cleaned = re.sub(r"[\x00-\x1F\x7F]", " ", line)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def _is_useful_ocr_line(self, line):
        if not line:
            return False
        if len(line) < 5:
            return False
        tokens = re.findall(r"[A-Za-z0-9]+", line)
        if len(tokens) < 2:
            return False
        # Skip mostly-symbol noise lines.
        alpha_num = len(re.findall(r"[A-Za-z0-9]", line))
        symbol = len(re.findall(r"[^A-Za-z0-9\s]", line))
        if alpha_num and symbol > (alpha_num * 1.5):
            return False
        return True

    def prepare_llm_input_text(self, text):
        lines = [self._clean_ocr_line(line) for line in (text or "").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return ""

        header_keywords = (
            "tax invoice", "invoice no", "inv no", "inv date", "date",
            "customer", "buyer", "supplier", "vat no", "vat",
        )
        header_lines = []
        for line in lines[:140]:
            lower = line.lower()
            if not any(k in lower for k in header_keywords):
                continue
            if not re.search(r"\d", line):
                continue
            if self._is_useful_ocr_line(line):
                header_lines.append(line)
        header_lines = header_lines[:30]

        start_idx = -1
        for i, line in enumerate(lines):
            lower = line.lower()
            if ("description" in lower and "rate" in lower and ("qty" in lower or "quantity" in lower)) or (
                "description of goods" in lower
            ) or ("sl." in lower and "description" in lower and "rate" in lower):
                start_idx = i
                break

        table_lines = []
        stop_keywords = (
            "amount in words", "grand total", "tax vat", "total taxes", "total / excluding vat",
            "total discount", "balance due", "net amount",
        )
        uom_re = re.compile(r"\b(EA|PCS|Nos|NOS|KG|Kg|Pair|PAIR|Pkt|PKT|Roll|ROLL|Rol|ROL)\b")
        money_re = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")

        if start_idx != -1:
            for line in lines[start_idx:start_idx + 120]:
                lower = line.lower()
                if any(k in lower for k in stop_keywords):
                    break
                if not uom_re.search(line):
                    continue
                numeric_count = len(money_re.findall(line))
                if numeric_count < 2:
                    continue
                if self._is_useful_ocr_line(line):
                    table_lines.append(line)

        if not table_lines:
            # Fallback: keep only lines that look like item rows.
            for line in lines[:250]:
                if not uom_re.search(line):
                    continue
                if len(money_re.findall(line)) < 2:
                    continue
                if self._is_useful_ocr_line(line):
                    table_lines.append(line)
            table_lines = table_lines[:80]

        focused = "INVOICE_HEADER:\n"
        focused += "\n".join(header_lines[:30])
        focused += "\n\nITEM_TABLE_TEXT:\n"
        focused += "\n".join(table_lines[:100])
        return focused[:12000]

    def get_invoice_image_base64(self, file_path):
        """Convert ALL pages of the PDF to base64 JPEG images."""
        try:
            pages = convert_from_path(file_path, dpi=200)
            if not pages:
                return None
            result = []
            for page in pages:
                img = page.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=80, optimize=True)
                result.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
            return result
        except Exception:
            frappe.log_error(frappe.get_traceback(), "Invoice image conversion failed")
            return None

    def get_llm_config(self):
        provider = "gemini"
        gemini_api_key = None
        gemini_model = None
        claude_api_key = None
        claude_model = None

        provider = self._get_single_value_if_field_exists("Invoice OCR Settings", "llm_provider") or provider
        gemini_api_key = self._get_single_value_if_field_exists("Invoice OCR Settings", "gemini_api_key") or gemini_api_key
        gemini_model = self._get_single_value_if_field_exists("Invoice OCR Settings", "gemini_model") or gemini_model
        claude_api_key = self._get_single_value_if_field_exists("Invoice OCR Settings", "claude_api_key") or claude_api_key
        claude_model = self._get_single_value_if_field_exists("Invoice OCR Settings", "claude_model") or claude_model

        gemini_api_key = self._get_single_value_if_field_exists("DoppioBot Settings", "gemini_api_key") or gemini_api_key
        gemini_model = self._get_single_value_if_field_exists("DoppioBot Settings", "gemini_model") or gemini_model
        provider = self._get_single_value_if_field_exists("DoppioBot Settings", "llm_provider") or provider
        claude_api_key = self._get_single_value_if_field_exists("DoppioBot Settings", "claude_api_key") or claude_api_key
        claude_model = self._get_single_value_if_field_exists("DoppioBot Settings", "claude_model") or claude_model

        provider = (provider or frappe.conf.get("invoice_ocr_llm_provider") or "gemini").strip().lower()
        gemini_api_key = gemini_api_key or frappe.conf.get("gemini_api_key")
        gemini_model = gemini_model or frappe.conf.get("gemini_model") or "gemini-2.0-flash"
        claude_api_key = claude_api_key or frappe.conf.get("claude_api_key") or frappe.conf.get("anthropic_api_key")
        claude_model = claude_model or frappe.conf.get("claude_model") or frappe.conf.get("anthropic_model") or "claude-sonnet-4-5"

        if provider == "claude":
            if claude_api_key:
                return provider, claude_api_key, claude_model
            if gemini_api_key:
                return "gemini", gemini_api_key, gemini_model
            return provider, None, claude_model

        if gemini_api_key:
            return "gemini", gemini_api_key, gemini_model
        if claude_api_key:
            return "claude", claude_api_key, claude_model
        return provider, None, gemini_model

    def extract_with_gemini(self, text, image_b64=None):
        provider, api_key, model = self.get_llm_config()
        if provider != "gemini" or not api_key:
            return None

        prompt = self._build_invoice_prompt(text)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        parts = []
        if image_b64:
            images = image_b64 if isinstance(image_b64, list) else [image_b64]
            for img in images:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img,
                    }
                })
        parts.append({"text": prompt})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
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

    def extract_with_claude(self, text, image_b64=None):
        provider, api_key, model = self.get_llm_config()
        frappe.logger().info(f"[Invoice OCR] Claude: provider={provider}, has_key={bool(api_key)}, model={model}")
        if provider != "claude" or not api_key:
            frappe.logger().info("[Invoice OCR] Claude skipped: provider not claude or no key")
            return None

        prompt = self._build_invoice_prompt(text)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # Build content: images first, then text prompt
        content_blocks = []
        if image_b64:
            images = image_b64 if isinstance(image_b64, list) else [image_b64]
            frappe.logger().info(f"[Invoice OCR] Sending {len(images)} page images to Claude")
            for img in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img,
                    },
                })
        content_blocks.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "max_tokens": 4096,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": content_blocks}
            ],
        }
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=120)
            frappe.logger().info(f"[Invoice OCR] Claude API status: {response.status_code}")
            if response.status_code != 200:
                frappe.logger().error(f"[Invoice OCR] Claude API error body: {response.text[:2000]}")
            response.raise_for_status()
        except Exception as e:
            frappe.log_error(frappe.get_traceback(), "Claude API request failed")
            frappe.logger().error(f"[Invoice OCR] Claude API exception: {str(e)}")
            return None

        try:
            data = response.json()
            blocks = data.get("content", [])
            text_blocks = [block.get("text", "") for block in blocks if block.get("type") == "text"]
            text_out = "\n".join(text_blocks).strip()
            frappe.logger().info(f"[Invoice OCR] Claude raw response length: {len(text_out)}")
            frappe.logger().info(f"[Invoice OCR] Claude raw response preview: {text_out[:1000]}")
        except Exception:
            frappe.log_error(frappe.get_traceback(), "Claude API response parse failed")
            return None

        parsed = self.parse_json_block(text_out)
        if parsed:
            frappe.logger().info(f"[Invoice OCR] Claude parsed OK, items count: {len(parsed.get('items', []))}")
        else:
            frappe.logger().error(f"[Invoice OCR] Claude JSON parse FAILED. Raw text: {text_out[:2000]}")
        return parsed

    def extract_with_llm(self, text, image_b64=None):
        provider, api_key, _model = self.get_llm_config()
        if not api_key:
            return None, "ocr"

        if provider == "claude":
            parsed = self.extract_with_claude(text, image_b64=image_b64)
            if parsed:
                return parsed, "claude"
            fallback = self.extract_with_gemini(text, image_b64=image_b64)
            if fallback:
                return fallback, "gemini"
            return None, "ocr"

        parsed = self.extract_with_gemini(text, image_b64=image_b64)
        if parsed:
            return parsed, "gemini"
        fallback = self.extract_with_claude(text, image_b64=image_b64)
        if fallback:
            return fallback, "claude"
        return None, "ocr"

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

    def is_noise_item(self, description, qty, rate):
        desc = (description or "").strip()
        if len(desc) < 4:
            return True
        if not re.search(r"[A-Za-z]", desc):
            return True

        lower = desc.lower()
        noise_keywords = (
            "seventy", "halala", "only", "amount in words", "total",
            "balance due", "tax vat", "round off", "discount", "subtotal",
            "invoice", "customer details", "buyer details",
        )
        if any(keyword in lower for keyword in noise_keywords):
            return True

        if qty is None or rate is None:
            return True
        if qty <= 0 or rate <= 0:
            return True
        if qty > 10000 or rate > 100000000:
            return True

        return False

    def sanitize_items(self, items):
        cleaned = []
        seen = set()

        for item in items or []:
            description = re.sub(r"\s{2,}", " ", (item.get("description") or "")).strip(" -:|")
            qty = self.parse_number(item.get("qty"))
            rate = self.parse_number(item.get("rate"))
            item_code = (item.get("item_code") or "").strip()
            uom = (item.get("uom") or item.get("unit") or "").strip()

            if self.is_noise_item(description, qty, rate):
                continue

            key = (
                description.lower(),
                round(float(qty), 4),
                round(float(rate), 4),
            )
            if key in seen:
                continue
            seen.add(key)

            cleaned.append({
                "description": description,
                "item_code": item_code,
                "qty": float(qty),
                "rate": float(rate),
                "uom": uom,
            })

        return cleaned

    def items_quality_score(self, items):
        if not items:
            return -999

        score = 0.0
        noise_keywords = ("amount", "words", "total", "vat", "discount", "balance", "invoice")
        for row in items:
            desc = (row.get("description") or "").strip()
            qty = self.parse_number(row.get("qty"))
            rate = self.parse_number(row.get("rate"))
            if not desc:
                score -= 4
                continue

            lower = desc.lower()
            alpha_chars = len(re.findall(r"[A-Za-z]", desc))
            symbol_chars = len(re.findall(r"[^A-Za-z0-9\s]", desc))
            good_words = re.findall(r"\b[A-Za-z]{3,}\b", desc)
            good_words_with_vowel = [w for w in good_words if re.search(r"[aeiouAEIOU]", w)]

            row_score = 0.0
            row_score += min(len(good_words_with_vowel), 6) * 0.6
            row_score += 1.0 if qty and qty > 0 else -1.0
            row_score += 1.0 if rate and rate > 0 else -1.0
            if alpha_chars > 0:
                row_score -= (symbol_chars / alpha_chars) * 1.5
            if any(k in lower for k in noise_keywords):
                row_score -= 1.5
            if "copper" in lower or "coil" in lower:
                row_score += 1.0
            if len(desc) < 8:
                row_score -= 0.8

            score += row_score

        # Small boost for plausible row count, but avoid favoring noisy long lists.
        score += min(len(items), 4) * 0.3
        return score

    def pick_best_items(self, llm_items, ocr_items):
        llm_items = self.sanitize_items(llm_items)
        ocr_items = self.sanitize_items(ocr_items)

        if llm_items and not ocr_items:
            return llm_items, "llm"
        if ocr_items and not llm_items:
            return ocr_items, "ocr"
        if not llm_items and not ocr_items:
            return [], "ocr"

        llm_score = self.items_quality_score(llm_items)
        ocr_score = self.items_quality_score(ocr_items)

        if llm_score >= ocr_score:
            return llm_items, "llm"
        return ocr_items, "ocr"

    def extract_items_from_table_section(self, lines):
        normalized_lines = [re.sub(r"\s{2,}", " ", (line or "").strip()) for line in lines if (line or "").strip()]
        if not normalized_lines:
            return []

        start_idx = -1
        for i, line in enumerate(normalized_lines):
            lower = line.lower()
            if ("description" in lower and "quantity" in lower and "rate" in lower) or (
                "description of goods" in lower
            ) or (
                "sl" in lower and "description" in lower and "rate" in lower
            ):
                start_idx = i
                break
        if start_idx == -1:
            return []

        stop_keywords = (
            "amount in words", "tax vat", "subtotal", "grand total", "balance due",
            "total", "vat", "net", "discount", "round off",
        )
        uom_pattern = r"(?:EA|PCS|Nos|NOS|KG|Kg|Pair|PAIR|Pkt|PKT|Roll|ROLL|Rol|ROL)"

        table_lines = []
        for line in normalized_lines[start_idx + 1:]:
            lower = line.lower()
            if any(k in lower for k in stop_keywords):
                break
            table_lines.append(line)

        items = []
        buffer = ""
        for line in table_lines:
            work = line
            if re.match(r"^\d+\s+", work):
                work = re.sub(r"^\d+\s+", "", work)

            # Merge wrapped description lines before pattern match
            candidate = (buffer + " " + work).strip() if buffer else work
            pattern = re.compile(
                rf"^(?P<desc>.+?)\s+(?P<qty>\d+(?:\.\d+)?)\s*(?P<uom>{uom_pattern})\b.*?\s+(?P<rate>\d+(?:,\d{{3}})*(?:\.\d+)?)\b",
                re.IGNORECASE,
            )
            m = pattern.search(candidate)
            if not m:
                buffer = candidate if len(candidate) < 260 else ""
                continue

            desc = re.sub(r"\s{2,}", " ", m.group("desc")).strip(" -:|")
            qty = self.parse_number(m.group("qty"))
            rate = self.parse_number(m.group("rate"))
            buffer = ""

            if self.is_noise_item(desc, qty, rate):
                continue
            items.append({
                "description": desc,
                "item_code": extract_item_code(desc),
                "qty": qty,
                "rate": rate,
            })

        return self.sanitize_items(items)

    def normalize_items(self, items):
        normalized = []
        for item in items or []:
            description = (item.get("description") or "").strip()
            if not description:
                continue
            qty = self.parse_number(item.get("qty"))
            rate = self.parse_number(item.get("rate"))
            item_code = (item.get("item_code") or "").strip()
            uom = (item.get("uom") or item.get("unit") or "").strip()
            normalized.append({
                "description": description,
                "item_code": item_code,
                "qty": qty or 1,
                "rate": rate or 0,
                "uom": uom,
            })
        return self.sanitize_items(normalized)

    def extract_items(self, text):
        lines = text.splitlines()

        table_items = self.extract_items_from_table_section(lines)
        if table_items:
            return table_items

        items = []
        description_buffer = []
        skip_keywords = (
            "total", "tax", "net", "payable", "amount due", "invoice", "voucher",
            "supplier", "address", "contact", "mobile", "creditors", "date",
            "vat number", "cr number", "iban", "account", "bank", "tel", "fax",
            "email", "amount in words", "balance due", "discount", "round off",
            "total amount", "vat %", "nature of goods", "services", "u/price",
            "qty", "sl.no", "ext amt", "vat amt", "unit", "disc",
            "addl", "addl no", "additional no", "building no", "zip code",
            "city name", "district", "issue date", "invoice number", "po no",
            "project name", "delivery date", "dn no", "buino", "bui no", "add.buino"
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
            if re.search(r"\badd\.?\s*bui\.?\s*no\b", lower):
                return True
            if re.search(r"\bbui\.?\s*no\b", lower):
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
            if re.search(r"\badd\.?\s*bui\.?\s*no\b", working, re.IGNORECASE):
                return None

            uom_pattern = r"(?:Pcs|PCS|EA|Pair|PAIR|Pkt|PKT|KG|Kg|Nos|NOS|Roll|ROLL|Rol|ROL)"
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
                spec_prefixes = {
                    "sch", "cl", "class", "astm", "ansi", "api", "asme",
                    "dn", "nps", "nb", "od", "id", "thk", "wt", "bw", "sw", "rf", "rtj"
                }
                for idx, token in enumerate(tokens):
                    if re.search(r"[A-Za-z]", token):
                        continue
                    if token.startswith(("-", "+")):
                        continue
                    if any(ch in token for ch in ("'", "\"")):
                        continue
                    if "/" in token:
                        continue
                    qty_candidate = self.parse_number(token)
                    if qty_candidate is None:
                        continue
                    if token.endswith("%"):
                        continue
                    prev_token = tokens[idx - 1] if idx > 0 else ""
                    prev_alpha = re.sub(r"[^A-Za-z]", "", prev_token).lower()
                    if prev_alpha in spec_prefixes:
                        continue
                    if prev_token.endswith("-"):
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

        def extract_items_from_qty_lines(lines_list):
            results = []
            last_english = ""
            qty_uom_re = re.compile(r"(\d+(?:\.\d+)?)\s*(Pcs|PCS|EA|Pair|PAIR|Pkt|PKT|KG|Kg|Nos|NOS|Roll|ROLL|Rol|ROL)\b", re.IGNORECASE)
            for line in lines_list:
                qty_match = qty_uom_re.search(line)
                if re.search(r"[A-Za-z]", line) and not qty_match:
                    last_english = line.strip()
                if not qty_match or not last_english:
                    continue
                qty = self.parse_number(qty_match.group(1))
                after = line[qty_match.end():]
                rate_match = re.search(r"(\d+(?:\.\d+)?)", after)
                rate = self.parse_number(rate_match.group(1)) if rate_match else None
                if qty is None or rate is None:
                    continue
                desc = normalize_description(last_english)
                if not desc:
                    continue
                results.append({
                    "description": desc,
                    "item_code": extract_item_code(desc),
                    "qty": float(qty),
                    "rate": float(rate),
                })
            return results

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

        fallback = extract_items_from_qty_lines(lines)
        if len(fallback) > len(items):
            return self.sanitize_items(fallback)
        return self.sanitize_items(items)

    def extract_party_from_text(self, text):
        def clean_party_line(value):
            value = re.sub(r"\b(name|customer|buyer|seller|details)\b", " ", value, flags=re.IGNORECASE)
            value = re.sub(r"\s{2,}", " ", value).strip(" -:")
            return value

        def looks_like_company(value):
            lower = value.lower()
            if not re.search(r"[A-Za-z]", value):
                return False
            if any(word in lower for word in ("total amount", "balance due", "round off", "discount", "vat", "total (sar)", "amount before vat")):
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
            match = re.search(r"(M/S|MS)\s*([A-Za-z][A-Za-z\s&.\-]{4,})", value)
            if match:
                return f"M/S {match.group(2).strip(' -:')}"
            return ""

        def prefer_supplier_brand(lines_subset):
            for line in lines_subset:
                lower = line.lower()
                if "gulf pipe" in lower or "pipe supply" in lower:
                    candidate = clean_party_line(line.strip())
                    if looks_like_company(candidate):
                        return extract_english_company(candidate)
                if "techno zone" in lower:
                    candidate = clean_party_line(line.strip())
                    if looks_like_company(candidate):
                        return extract_english_company(candidate)
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
                candidate = extract_ms_company(line)
                if candidate:
                    return candidate
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
            picked = prefer_supplier_brand(lines)
            if picked:
                return picked
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
        debug = {}
        if not self.file:
            frappe.throw("No file attached.")

        file_path = get_file_path(self.file)
        debug["file_path"] = file_path

        text = extract_text_from_pdf(file_path)
        debug["ocr_text_length"] = len(text)
        debug["ocr_text_preview"] = text[:3000]

        if not text.strip():
            frappe.msgprint("No OCR text found.")
            return {"added_rows": [], "debug": debug}

        # Keep it simple: send full OCR text to LLM.
        text_for_llm = text[:50000]
        image_b64 = self.get_invoice_image_base64(file_path)
        debug["image_pages_count"] = len(image_b64) if isinstance(image_b64, list) else (1 if image_b64 else 0)

        llm_data, llm_source = self.extract_with_llm(text_for_llm, image_b64=image_b64)
        debug["llm_source"] = llm_source
        debug["llm_data_raw"] = llm_data

        extraction_source = "ocr"
        items_source = "ocr"
        if llm_data:
            extraction_source = llm_source
            debug["llm_items_raw"] = llm_data.get("items", [])
            debug["llm_items_raw_count"] = len(llm_data.get("items", []))
            llm_items = self.normalize_items(llm_data.get("items", []))
            debug["llm_items_after_normalize"] = llm_items
            debug["llm_items_after_normalize_count"] = len(llm_items)
            if llm_items:
                items = llm_items
                items_source = "llm"
            else:
                debug["llm_items_empty_falling_back_to_ocr"] = True
                items = self.sanitize_items(self.extract_items(text))
                items_source = "ocr"
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
                extracted_party = (llm_data.get("supplier_name") or llm_data.get("party_name") or "").strip()
                debug["llm_supplier_name"] = extracted_party
                if extracted_party:
                    self.party = extracted_party
        else:
            debug["llm_data_is_null"] = True
            items = self.sanitize_items(self.extract_items(text))
            if not self.party:
                extracted_party = self.extract_party_from_text(text)
                if extracted_party:
                    self.party = extracted_party
        if not self.party:
            extracted_party = self.extract_party_from_text(text)
            if extracted_party:
                self.party = extracted_party

        debug["final_items_count"] = len(items)
        debug["final_items"] = items
        debug["final_items_source"] = items_source
        debug["final_party"] = self.party
        debug["final_party_type"] = self.party_type
        if self.party and self.party_type in ("Customer", "Supplier"):
            if not frappe.db.exists(self.party_type, self.party):
                party_doc = frappe.new_doc(self.party_type)
                if self.party_type == "Customer":
                    party_doc.customer_name = self.party
                    party_doc.customer_type = "Company"
                    party_doc.customer_group = (
                        frappe.defaults.get_user_default("Customer Group")
                        or frappe.db.get_value("Customer Group", {"is_group": 0}, "name")
                        or "Commercial"
                    )
                    party_doc.territory = (
                        frappe.defaults.get_user_default("Territory")
                        or frappe.db.get_value("Territory", {"is_group": 0}, "name")
                        or "All Territories"
                    )
                    if party_doc.meta.has_field("customer_code"):
                        party_doc.customer_code = self.party[:140]
                else:
                    party_doc.supplier_name = self.party
                    party_doc.supplier_type = "Company"
                party_doc.insert(ignore_permissions=True)
        extracted_data = {
            "items": items,
            "party": self.party,
            "items_source": items_source,
            "vision_input_used": bool(image_b64),
        }
        if llm_data:
            extracted_data.update({
                "supplier_name": llm_data.get("supplier_name") or self.party,
                "invoice_no": llm_data.get("invoice_no"),
                "date": llm_data.get("date"),
                "total": llm_data.get("total"),
                "currency": llm_data.get("currency"),
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

            # Resolve UOM - check if it exists in the system
            raw_uom = (row.get("uom") or "").strip()
            resolved_uom = ""
            if raw_uom:
                # Try exact match first
                if frappe.db.exists("UOM", raw_uom):
                    resolved_uom = raw_uom
                else:
                    # Try case-insensitive match
                    found = frappe.db.get_value("UOM", {"name": ["like", raw_uom]}, "name")
                    if found:
                        resolved_uom = found
                    else:
                        resolved_uom = self.get_default_uom()
            else:
                resolved_uom = self.get_default_uom()

            self.append("invoice_upload_item", {
                "ocr_description": row["description"][:140],
                "qty": row["qty"],
                "uom": resolved_uom,
                "rate": row["rate"],
                "item": matched_item
            })
            added_rows.append({
                "item_code": row.get("item_code"),
                "matched_item": matched_item,
                "match_score": round(match_score, 3) if match_score else 0,
                "qty": row.get("qty"),
                "uom": resolved_uom,
                "rate": row.get("rate"),
                "description": row.get("description"),
            })

        # ✅ Correctly set JSON field AFTER extraction
        self.extracted_data = json.dumps(extracted_data, indent=2)
        self.ocr_status = "Extracted"
        self.save()

        if not items:
            frappe.msgprint("OCR done but no items found. Please check PDF format.")
        else:
            frappe.msgprint(f"Extraction completed. {len(items)} items found via {items_source}.")
        return {"added_rows": added_rows, "items_count": len(items), "debug": debug}


@frappe.whitelist()
def extract_invoice(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    return doc.extract_invoice()
