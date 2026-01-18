import frappe
import json
import pytesseract
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from frappe.utils.file_manager import get_file_path
from PIL import Image

@frappe.whitelist()
def create_invoice(name):
    doc = frappe.get_doc("Invoice Upload", name)
    if not doc.extracted_data:
        frappe.throw("No OCR data extracted.")
    data = json.loads(doc.extracted_data)

    party_type = doc.party_type if doc.party_type in ("Supplier", "Customer") else "Supplier"
    if party_type == "Supplier":
        pi = frappe.new_doc("Purchase Invoice")
        pi.supplier = doc.party
        if doc.party and frappe.db.get_value("Supplier", doc.party, "disabled"):
            frappe.db.set_value("Supplier", doc.party, "disabled", 0)
    else:
        pi = frappe.new_doc("Sales Invoice")
        pi.customer = doc.party
        if doc.party and frappe.db.get_value("Customer", doc.party, "disabled"):
            frappe.db.set_value("Customer", doc.party, "disabled", 0)

    company = frappe.defaults.get_user_default("Company")
    if company:
        pi.company = company

    def get_default_item_group():
        if frappe.db.exists("Item Group", "All Item Groups"):
            return "All Item Groups"
        return (
            frappe.db.get_value("Item Group", {"is_group": 0}, "name")
            or frappe.db.get_value("Item Group", {}, "name")
        )

    def get_default_uom():
        if frappe.db.exists("UOM", "Nos"):
            return "Nos"
        if frappe.db.exists("UOM", "Unit"):
            return "Unit"
        return frappe.db.get_value("UOM", {}, "name")

    def get_or_create_item(description):
        if not description:
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
        item.item_group = get_default_item_group()
        item.stock_uom = get_default_uom()
        item.is_stock_item = 0
        item.disabled = 0
        item.insert(ignore_permissions=True)
        return item.name

    invoice_items = doc.get("invoice_upload_item") or []
    if invoice_items:
        for row in invoice_items:
            item_code = row.item
            if not item_code and row.ocr_description:
                item_code = get_or_create_item(row.ocr_description)

            item_row = {
                "qty": row.qty,
                "rate": row.rate,
            }
            if item_code:
                item_row["item_code"] = item_code
                item_row["uom"] = frappe.db.get_value("Item", item_code, "stock_uom") or get_default_uom()
            else:
                item_row["item_name"] = row.ocr_description
                item_row["uom"] = get_default_uom()

            pi.append("items", item_row)
    else:
        for item in data.get("items", []):
            pi.append("items", {
                "item_name": item.get("description"),
                "qty": item.get("qty"),
                "rate": item.get("rate"),
                "uom": get_default_uom(),
            })

    pi.posting_date = data.get("date")
    if pi.doctype == "Purchase Invoice":
        if not pi.get("bill_no"):
            pi.bill_no = doc.name
        if not pi.get("bill_date"):
            pi.bill_date = doc.date or frappe.utils.nowdate()

    for item in pi.items:
        if not item.get("uom"):
            item.uom = get_default_uom()

    if pi.company:
        default_expense_account = frappe.db.get_value(
            "Company", pi.company, "default_expense_account"
        )
        default_payable = frappe.db.get_value(
            "Company", pi.company, "default_payable_account"
        )
        default_receivable = frappe.db.get_value(
            "Company", pi.company, "default_receivable_account"
        )
        if default_expense_account:
            for item in pi.items:
                if not item.get("expense_account"):
                    item.expense_account = default_expense_account
        if pi.doctype == "Purchase Invoice" and default_payable and not pi.get("credit_to"):
            pi.credit_to = default_payable
        if pi.doctype == "Sales Invoice" and default_receivable and not pi.get("debit_to"):
            pi.debit_to = default_receivable

    pi.insert(ignore_permissions=True)
    return {"doctype": pi.doctype, "name": pi.name}

# def extract_invoice_data(docname):
#     doc = frappe.get_doc("Invoice Upload", docname)
#     doc.ocr_status = "Processing"
#     doc.save()
#     frappe.db.commit()

#     try:
#         file_path = get_file_path(doc.file)
#         frappe.logger().info(f"[OCR] File path: {file_path}")
#         text = ""

#         if file_path.endswith(".pdf"):
#             images = convert_from_path(file_path)
#             for img in images:
#                 text += pytesseract.image_to_string(img)
#         else:
#             img = Image.open(file_path)
#             text = pytesseract.image_to_string(img)

#         frappe.logger().info(f"[OCR] Extracted Text:\n{text}")

#         invoice_data = {
#             "invoice_no": extract_keyword(text, ["Invoice#", "Invoice No", "Invoice Number"]),
#             "date": extract_keyword(text, ["Date"]),
#             "items": extract_items(text),
#             "total": extract_keyword(text, ["Total", "Amount Due"])
#         }

#         doc.ocr_status = "Extracted"
#         doc.extracted_data = json.dumps(invoice_data, indent=2)
#         doc.save()
#         frappe.db.commit()

#     except Exception:
#         doc.ocr_status = "Failed"
#         doc.save()
#         frappe.db.commit()
#         frappe.log_error(frappe.get_traceback(), "OCR Failed")


def extract_invoice_data(docname):
    doc = frappe.get_doc("Invoice Upload", docname)
    doc.ocr_status = "Processing"
    doc.save()
    frappe.db.commit()

    try:
        # FORCED SAMPLE TEXT
        text = """
        INVOICE
        Invoice No: INV-2025-001
        Date: 2025-05-20
        Tramadol Tablet 100mg   10    50.00   500.00
        Paracetamol Syrup 250ml 5     80.00   400.00
        Vitamin D3 Drops        2     150.00  300.00
        Total:                          PKR 1200.00
        """

        invoice_data = {
            "invoice_no": extract_keyword(text, ["Invoice#", "Invoice No", "Invoice Number"]),
            "date": extract_keyword(text, ["Date"]),
            "items": extract_items(text),
            "total": extract_keyword(text, ["Total", "Amount Due"])
        }

        doc.ocr_status = "Extracted"
        doc.extracted_data = json.dumps(invoice_data, indent=2)
        doc.save()
        frappe.db.commit()

    except Exception:
        doc.ocr_status = "Failed"
        doc.save()
        frappe.db.commit()
        frappe.log_error(frappe.get_traceback(), "OCR Failed")


        
def extract_keyword(text, keys):
    for line in text.splitlines():
        for key in keys:
            if key.lower() in line.lower():
                return line.split()[-1]
    return ""


def extract_items(text):
    lines = text.splitlines()
    items = []
    for line in lines:
        if "Qty" in line or "Description" in line or "Rate" in line:
            continue
        if any(char.isdigit() for char in line):
            parts = line.split()
            if len(parts) >= 3:
                items.append({
                    "description": parts[0],
                    "qty": parts[1],
                    "rate": parts[2],
                })
    return items
