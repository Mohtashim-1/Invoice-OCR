import frappe
import json
import pytesseract
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from frappe.utils.file_manager import get_file_path
from frappe.model.document import Document
from PIL import Image
from frappe.utils import add_days, get_url_to_form, nowdate

class InvoiceUpload(Document):
    def on_submit(self):
        try:
            self.reload()
            self.create_invoice_from_child()
        except Exception:
            frappe.db.set_value("Invoice Upload", self.name, "ocr_status", "Failed")
            frappe.db.commit()
            frappe.log_error(frappe.get_traceback(), "Invoice Creation Failed")
            raise

    def extract_invoice(self):
        if not self.file:
            frappe.throw("No file attached.")

        file_path = get_file_path(self.file)
        text = ""

        if file_path.endswith(".pdf"):
            images = convert_from_path(file_path)
            for img in images:
                text += pytesseract.image_to_string(img)
        else:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)

        items = self.extract_items(text)
        extracted_data = {
            "items": items,
            "party": None
        }

        self.set("invoice_upload_item", [])
        for row in items:
            matched_item = frappe.db.get_value("Item", {"item_name": row["description"]})
            self.append("invoice_upload_item", {
                "ocr_description": row["description"],
                "qty": row["qty"],
                "rate": row["rate"],
                "item": matched_item
            })

        party_code = self.extract_party(text)
        if party_code:
            extracted_data["party"] = party_code.strip()

        self.extracted_data = json.dumps(extracted_data, indent=2)
        self.ocr_status = "Extracted"
        self.save()
        frappe.msgprint("OCR Extraction completed. Please review data before submitting.")

    def ensure_party_exists(self):
        extracted = json.loads(self.extracted_data or '{}')
        party = extracted.get("party")

        if not party or not party.strip():
            frappe.throw("Party is missing. Cannot create invoice.")

        if self.party_type == "Customer" and not frappe.db.exists("Customer", party):
            doc = frappe.get_doc({
                "doctype": "Customer",
                "customer_name": party.strip(),
                "customer_group": "All Customer Groups",
                "territory": "All Territories"
            })
            doc.insert(ignore_permissions=True)
            frappe.db.commit()
            extracted["party"] = doc.name

        elif self.party_type == "Supplier" and not frappe.db.exists("Supplier", party):
            doc = frappe.get_doc({
                "doctype": "Supplier",
                "supplier_name": party.strip(),
                "supplier_group": "All Supplier Groups",
                "country": "Pakistan"
            })
            doc.insert(ignore_permissions=True)
            frappe.db.commit()
            extracted["party"] = doc.name

        self.db_set("party", extracted["party"])  # Safe update after submit
        self.extracted_data = json.dumps(extracted, indent=2)
        self.save()

    def create_invoice_from_child(self):
        extracted = json.loads(self.extracted_data or '{}')
        items = extracted.get("items", [])
        party = extracted.get("party")

        if not items:
            frappe.throw("No items found. Please extract first.")

        self.ensure_party_exists()

        if not self.party:
            frappe.throw("Unable to determine or create party.")

        if self.party_type == "Supplier":
            inv = frappe.new_doc("Purchase Invoice")
            inv.supplier = self.party
        else:
            inv = frappe.new_doc("Sales Invoice")
            inv.customer = self.party

        expense_account = self.get_expense_account()

        for row in items:
            item_code = frappe.db.get_value("Item", {"item_name": row["description"]})
            if not item_code:
                item_code = self.ensure_item_exists(row["description"])

            inv.append("items", {
                "item_code": item_code,
                "qty": row["qty"],
                "rate": row["rate"],
                "uom": "Nos",
                "expense_account": expense_account
            })

        posting_date = getattr(self, "posting_date", None) or nowdate()
        inv.posting_date = posting_date
        inv.due_date = add_days(posting_date, 30)
        inv.insert(ignore_permissions=True)

        frappe.msgprint(f"<a href='{get_url_to_form(inv.doctype, inv.name)}'>{inv.name}</a> created")

    def get_expense_account(self):
        company = frappe.defaults.get_user_default("Company")
        account = frappe.db.get_value("Company", company, "default_expense_account")
        if not account:
            account = frappe.db.get_value("Account", {
                "account_type": "Expense",
                "company": company,
                "is_group": 0
            }, "name")
        if not account:
            frappe.throw("No default Expense Account found for the company.")
        return account

    def ensure_item_exists(self, description):
        item_code = frappe.db.get_value("Item", {"item_name": description})
        if not item_code:
            item = frappe.get_doc({
                "doctype": "Item",
                "item_name": description,
                "item_code": description,
                "item_group": "All Item Groups",
                "stock_uom": "Nos",
                "is_stock_item": 0
            })
            item.insert(ignore_permissions=True)
            item_code = item.name
        return item_code

    def extract_items(self, text):
        lines = text.splitlines()
        items = []
        pattern = re.compile(r"(.+?)\s+(\d+\.\d{1,2})\s+(\d+\.\d{1,2})\s+\$?(\d+\.\d{1,2})")

        for line in lines:
            match = pattern.search(line)
            if match:
                try:
                    description = match.group(1).strip()
                    qty = float(match.group(2))
                    rate = float(match.group(3))
                    items.append({
                        "description": description,
                        "qty": qty,
                        "rate": rate
                    })
                except:
                    continue
        return items

    def extract_party(self, text):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if any(key.lower() in line.lower() for key in ["Customer Code", "Supplier Code", "Customer:", "Supplier:"]):
                parts = line.split(":")
                if len(parts) > 1 and parts[1].strip():
                    return parts[1].strip()
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        return next_line
        return None

@frappe.whitelist()
def extract_invoice(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    doc.extract_invoice()
