import cv2
import pytesseract
import numpy as np
import frappe
import json
import re
import traceback
import difflib
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
            # Create draft invoice on submit
            self.create_invoice_from_child(submit_invoice=False)
            # Make document non-editable
            frappe.db.set_value("Invoice Upload", self.name, "docstatus", 1)
        except Exception as e:
            frappe.db.set_value("Invoice Upload", self.name, "ocr_status", "Failed")
            frappe.db.commit()
            error_message = f"Invoice Creation Failed: {str(e)}\n{traceback.format_exc()}"
            frappe.log_error(error_message, "Invoice Creation Failed")
            frappe.throw(f"Invoice creation failed: {str(e)}")

    def before_save(self):
        # Make submitted documents read-only
        if self.docstatus == 1:
            self.flags.read_only = True

    def extract_invoice(self):
        try:
            if not self.file:
                frappe.throw("No file attached.")

            file_path = get_file_path(self.file)
            text = ""

            # Enhanced Odoo-style preprocessing
            def preprocess_image(pil_img):
                try:
                    img = np.array(pil_img.convert("RGB"))
                    channels = img.shape[-1] if img.ndim == 3 else 1
                    
                    if channels == 3:
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img
                        
                    # Enhance resolution (Odoo style)
                    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # Apply CLAHE for contrast enhancement (Odoo style)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(scaled)
                    
                    # Apply adaptive thresholding
                    thresh = cv2.adaptiveThreshold(
                        enhanced, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 15, 10
                    )
                    
                    # Apply erosion to reduce noise (Odoo style)
                    kernel = np.ones((3, 3), np.uint8)
                    processed = cv2.erode(thresh, kernel, iterations=1)
                    
                    return processed
                except Exception as e:
                    frappe.log_error(f"Image processing failed: {str(e)}", "OCR Error")
                    return pil_img  # Return original if processing fails

            if file_path.endswith(".pdf"):
                images = convert_from_path(file_path, dpi=300)
                for img in images:
                    processed = preprocess_image(img)
                    text += pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
            else:
                img = Image.open(file_path)
                processed = preprocess_image(img)
                text = pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")

            # Save extracted text for debugging
            self.raw_ocr_text = text[:10000]  # Save first 10k characters
            self.save()
            
            items = self.extract_items(text)
            extracted_data = {
                "items": items,
                "party": None
            }

            # Get all items for matching
            all_items = self.get_items_for_matching()
            
            self.set("invoice_upload_item", [])
            seen_descriptions = set()  # Track seen descriptions to avoid duplicates
            
            for row in items:
                # Skip empty or invalid descriptions
                if not row["description"] or len(row["description"]) < 3:
                    continue
                    
                # Normalize description for duplicate check
                normalized_desc = re.sub(r'\W+', '', row["description"].lower())
                
                # Skip duplicate items
                if normalized_desc in seen_descriptions:
                    continue
                seen_descriptions.add(normalized_desc)
                
                # First try to match text in square brackets (item codes)
                bracket_match = None
                bracket_text = self.extract_bracket_text(row["description"])
                
                # Try matching with bracket text first
                if bracket_text:
                    bracket_match = self.fuzzy_match_item(bracket_text, all_items)
                    if bracket_match and bracket_match["score"] > 85:
                        matched_item = bracket_match["item_name"]
                        self.append("invoice_upload_item", {
                            "ocr_description": row["description"],
                            "qty": row["qty"],
                            "rate": row["rate"],
                            "item": matched_item
                        })
                        continue
                
                # If bracket match not found, try full description
                full_match = self.fuzzy_match_item(row["description"], all_items)
                if full_match and full_match["score"] > 75:
                    matched_item = full_match["item_name"]
                else:
                    matched_item = None
                    
                self.append("invoice_upload_item", {
                    "ocr_description": row["description"],
                    "qty": row["qty"],
                    "rate": row["rate"],
                    "item": matched_item
                })

            # Extract party with fuzzy matching
            party_name = self.extract_party(text)
            if party_name:
                party_match = self.fuzzy_match_party(party_name)
                if party_match:
                    extracted_data["party"] = party_match["name"]
                else:
                    extracted_data["party"] = party_name

            self.extracted_data = json.dumps(extracted_data, indent=2)
            self.ocr_status = "Extracted"
            self.save()
            frappe.msgprint("OCR Extraction completed. Please review data before submitting.")
            
            return {
                "status": "success",
                "items": items,
                "party": extracted_data["party"]
            }
        except Exception as e:
            error_message = f"Extraction failed: {str(e)}\n{traceback.format_exc()}"
            frappe.log_error(error_message, "OCR Extraction Failed")
            frappe.throw(f"Extraction failed: {str(e)}")

    def ensure_party_exists(self):
        extracted = json.loads(self.extracted_data or '{}')
        party = extracted.get("party")

        if not party or not party.strip():
            frappe.throw("Party is missing. Cannot create invoice.")
        
        # Check if party exists
        if frappe.db.exists(self.party_type, party):
            self.party = party
            return
            
        # Try fuzzy matching again in case of close matches
        party_match = self.fuzzy_match_party(party)
        if party_match:
            self.party = party_match["name"]
            return

        # If no match found, throw error
        frappe.throw(f"Party '{party}' not found in the system. Please create it first.")

    def create_invoice_from_child(self, submit_invoice=False):
        """Create invoice, optionally submit it based on parameter"""
        # Check if invoice already created
        if self.invoice_created:
            frappe.throw("Invoice already created for this document")
            
        # Ensure party is set and exists
        self.ensure_party_exists()

        # Create the appropriate invoice type
        if self.party_type == "Supplier":
            inv = frappe.new_doc("Purchase Invoice")
            inv.supplier = self.party
            inv.bill_no = self.name
            inv.bill_date = self.date
        else:
            inv = frappe.new_doc("Sales Invoice")
            inv.customer = self.party

        # Get appropriate account based on invoice type
        if self.party_type == "Supplier":
            account = self.get_expense_account()
            account_field = "expense_account"
        else:
            account = self.get_income_account()
            account_field = "income_account"

        # Add items from the child table
        items_added = 0
        for row in self.invoice_upload_item:
            item_code = row.item
            if not item_code:
                frappe.msgprint(f"Skipping item: {row.ocr_description} - no item matched", alert=True)
                continue

            try:
                # Get item details
                item_doc = frappe.get_doc("Item", item_code)
                
                # Create item dictionary
                item_dict = {
                    "item_code": item_code,
                    "item_name": item_doc.item_name,
                    "description": item_doc.description or row.ocr_description,
                    "qty": row.qty,
                    "rate": row.rate,
                    "uom": item_doc.stock_uom or "Nos"
                }
                
                # Set account field based on invoice type
                item_dict[account_field] = account
                
                inv.append("items", item_dict)
                items_added += 1
            except Exception as e:
                frappe.msgprint(f"Error adding item {item_code}: {str(e)}", alert=True, indicator="red")

        if items_added == 0:
            frappe.throw("No valid items found to create invoice")

        # Set dates
        posting_date = getattr(self, "posting_date", None) or nowdate()
        inv.posting_date = posting_date
        inv.due_date = add_days(posting_date, 30)
        
        # Calculate totals
        inv.run_method("set_missing_values")
        inv.run_method("calculate_taxes_and_totals")
        
        # Save invoice with appropriate validation
        try:
            # Bypass validations for draft invoices
            inv.flags.ignore_validate = True
            inv.flags.ignore_mandatory = True
            inv.insert(ignore_permissions=True)
            status = "Draft"
        except Exception as e:
            frappe.msgprint(f"Invoice creation failed: {str(e)}", alert=True, indicator="red")
            frappe.log_error(f"Invoice creation failed: {str(e)}", "Invoice Creation Error")
            return
        
        # Update status and reference
        frappe.db.set_value(self.doctype, self.name, {
            "invoice_created": 1,
            "invoice_reference": inv.name,
            "invoice_type": inv.doctype,
            "invoice_status": status
        })

        frappe.msgprint(f"<a href='{get_url_to_form(inv.doctype, inv.name)}'>{inv.name}</a> created ({status})")

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

    def get_income_account(self):
        company = frappe.defaults.get_user_default("Company")
        account = frappe.db.get_value("Company", company, "default_income_account")
        if not account:
            account = frappe.db.get_value("Account", {
                "account_type": "Income",
                "company": company,
                "is_group": 0
            }, "name")
        if not account:
            frappe.throw("No default Income Account found for the company.")
        return account

    def extract_items(self, text):
        # First try to extract as structured table items
        table_items = self.extract_table_items(text)
        if table_items:
            return table_items

        # Then try to extract as bill of charges
        charge_items = self.extract_charges(text)
        if charge_items:
            return charge_items

        # Fallback to original method if no structured data found
        items = []
        # Look for quantity patterns in the text
        qty_matches = re.finditer(r'(\d+,\d+\.\d{3}|\d+\.\d{3}|\d+)\s*(kg|Units)?', text, re.IGNORECASE)
        
        for match in qty_matches:
            try:
                qty_str = match.group(1).replace(',', '')
                qty = float(qty_str)
                
                # Find description in previous lines
                desc_start = text.rfind('\n', 0, match.start()) + 1
                desc_end = match.start()
                description = text[desc_start:desc_end].strip()
                
                # Clean up description
                description = re.sub(r'^\W+|\W+$', '', description)  # Remove surrounding symbols
                description = re.sub(r'\s+', ' ', description)  # Collapse multiple spaces
                description = re.sub(r'\.{3,}', '', description)  # Remove ellipses
                
                # Skip short descriptions
                if len(description) < 3:
                    continue
                
                # Find rate in the same line or next
                rate_match = re.search(r'(\d+,\d+\.\d{2,3}|\d+\.\d{2,3}|\d+)', 
                                      text[match.start():match.start()+100])
                if rate_match:
                    rate_str = rate_match.group(1).replace(',', '')
                    rate = float(rate_str)
                else:
                    rate = 0.0
                
                items.append({
                    "description": description,
                    "qty": qty,
                    "rate": rate
                })
            except Exception as e:
                frappe.log_error(f"Item extraction failed: {str(e)}", "Item Extraction Error")
                continue
        
        return items

    def extract_table_items(self, text):
        """Extract items from structured tables with pipe format"""
        items = []
        lines = text.splitlines()
        
        # Find the start of the items table
        start_index = -1
        for i, line in enumerate(lines):
            if "QUANTITY" in line and "UNIT PRICE" in line and "AMOUNT" in line:
                start_index = i + 1
                break
        
        # If table header found, process subsequent lines
        if start_index != -1:
            for i in range(start_index, min(start_index + 10, len(lines))):
                line = lines[i]
                if not line.strip():
                    break
                
                # Split line by pipe character
                parts = [part.strip() for part in line.split('|')]
                if len(parts) < 4:
                    continue
                
                try:
                    description = parts[0]
                    qty_str = parts[1].replace(',', '')
                    rate_str = parts[2].replace(',', '')
                    
                    # Extract quantity number
                    qty_match = re.search(r'(\d+\.\d{3})', qty_str)
                    if not qty_match:
                        continue
                    qty = float(qty_match.group(1))
                    
                    # Extract rate number
                    rate_match = re.search(r'(\d+\.\d{2,3})', rate_str)
                    if not rate_match:
                        continue
                    rate = float(rate_match.group(1))
                    
                    # Clean up description
                    description = re.sub(r'\s+', ' ', description)  # Collapse spaces
                    description = re.sub(r'\.{3,}', '', description)  # Remove ellipses
                    description = re.sub(r'^\W+|\W+$', '', description)  # Remove surrounding symbols
                    
                    # Skip short descriptions
                    if len(description) < 3:
                        continue
                    
                    items.append({
                        "description": description,
                        "qty": qty,
                        "rate": rate
                    })
                except Exception as e:
                    continue
        
        # If no pipe items found, try alternative table format
        if not items:
            # Pattern to match table rows without pipes
            pattern = re.compile(
                r'^(.+?)\s+(\d{1,3}(?:,\d{3})*\.\d{3})\s*(kg|Units)?\s+(\d{1,3}(?:,\d{3})*\.\d{2,3})\s+.*?\d+\.\d{2}',
                re.IGNORECASE
            )
            
            for line in lines:
                match = pattern.search(line)
                if match:
                    try:
                        description = match.group(1).strip()
                        qty = float(match.group(2).replace(',', ''))
                        rate = float(match.group(4).replace(',', ''))
                        
                        # Clean up description
                        description = re.sub(r'\s+', ' ', description)
                        description = re.sub(r'\.{3,}', '', description)
                        description = re.sub(r'^\W+|\W+$', '', description)
                        
                        if len(description) < 3:
                            continue
                            
                        items.append({
                            "description": description,
                            "qty": qty,
                            "rate": rate
                        })
                    except Exception as e:
                        continue
        
        return items

    def extract_charges(self, text):
        """Extract items from bill of charges format"""
        items = []
        clean_text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Find the PARTICULARS section
        start = clean_text.find("PARTICULARS")
        if start == -1:
            return items

        # Extract table data
        table_pattern = r'Custom Duties(.+?)Service Charges'
        table_match = re.search(table_pattern, clean_text, re.DOTALL)
        if not table_match:
            return items
            
        table_text = table_match.group(1)
        
        # Process each charge line
        charge_pattern = r'(\w[\w\s\/-]+)\s+(\d{1,3}(?:,\d{3})*)\s+(\d{1,3}(?:,\d{3})*)'
        for match in re.finditer(charge_pattern, table_text):
            try:
                charge_name = match.group(1).strip()
                consignee_amount = float(match.group(2).replace(',', ''))
                balance_amount = float(match.group(3).replace(',', ''))
                total_amount = consignee_amount + balance_amount
                
                # Skip zero-amount lines
                if total_amount > 0:
                    items.append({
                        "description": charge_name,
                        "qty": 1,
                        "rate": total_amount
                    })
            except Exception:
                continue
                
        return items

    def extract_party(self, text):
        """Extract the actual partner name from the invoice"""
        # 1. First look for explicit "Partner Name" field
        partner_match = re.search(r'Partner\s*Name\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if partner_match:
            party = partner_match.group(1).strip()
            # Remove any trailing non-alphanumeric characters
            party = re.sub(r'[^\w\s\-]$', '', party).strip()
            if party:
                return party

        # 2. Look for the most prominent name in the top section
        # This is usually the customer/supplier name
        top_section = text.split("Invoice Date:")[0] if "Invoice Date:" in text else text[:500]
        
        # Find the longest word sequence that looks like a name
        name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', top_section)
        if name_candidates:
            # Get the longest candidate as it's likely the partner name
            name_candidates.sort(key=len, reverse=True)
            return name_candidates[0]

        # 3. Look for other common labels
        party_labels = ["Customer", "Client", "Supplier", "Vendor", "Bill To", "Sold To"]
        for label in party_labels:
            pattern = re.compile(fr'{label}\s*:\s*([^\n]+)', re.IGNORECASE)
            match = pattern.search(text)
            if match:
                party = match.group(1).strip()
                party = re.sub(r'[^\w\s\-]$', '', party).strip()
                if party:
                    return party

        # 4. Look for a name-like string near the invoice title
        title_match = re.search(r'Invoice\s+\w+/\d+/\d+', text, re.IGNORECASE)
        if title_match:
            # Look before and after the title for a name
            start_pos = max(0, title_match.start() - 100)
            end_pos = min(len(text), title_match.end() + 100)
            context = text[start_pos:end_pos]
            
            # Find the most prominent name in this context
            name_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', context)
            if name_candidates:
                name_candidates.sort(key=len, reverse=True)
                return name_candidates[0]

        return None

    def get_items_for_matching(self):
        """Get all items with their names and codes for matching"""
        # Get all active items
        items = frappe.get_all("Item", 
                              fields=["item_code", "item_name"],
                              filters={"disabled": 0})
        
        # Create a list of all possible names and codes
        item_data = []
        for item in items:
            # Add item code as primary identifier
            if item.item_code:
                item_data.append({
                    "item_name": item.item_code,  # Actual item code
                    "match_text": re.sub(r'[\[\]]', '', item.item_code.lower()),
                    "type": "code"
                })
            
            # Add item name as secondary identifier
            if item.item_name and item.item_name.lower() != item.item_code.lower():
                item_data.append({
                    "item_name": item.item_code,  # Still use item code as identifier
                    "match_text": re.sub(r'[\[\]]', '', item.item_name.lower()),
                    "type": "name"
                })
        
        return item_data
    
    def extract_bracket_text(self, description):
        """Extract text within square brackets"""
        matches = re.findall(r'\[(.*?)\]', description)
        return matches[0] if matches else None
    
    def fuzzy_match_item(self, text, all_items):
        """Find the best item match using fuzzy matching"""
        if not text:
            return None
            
        # Clean text by removing special characters and brackets
        clean_text = re.sub(r'[\[\]]', '', text).lower().strip()
        best_match = None
        best_score = 0
        
        for item in all_items:
            # Clean match text similarly
            clean_match = item["match_text"]
            
            # Calculate similarity score
            score = difflib.SequenceMatcher(None, clean_text, clean_match).ratio() * 100
            
            # Give extra weight to code matches
            if item["type"] == "code":
                score = min(score * 1.2, 100)  # Boost code matches by 20%
                
            if score > best_score:
                best_score = score
                best_match = {
                    "item_name": item["item_name"],  # Actual item code
                    "score": score,
                    "match_type": item["type"],
                    "match_text": item["match_text"]
                }
        
        # Return match only if it meets minimum confidence
        if best_score > 70:
            return best_match
            
        # Try again with bracket extraction if first match failed
        bracket_text = self.extract_bracket_text(text)
        if bracket_text:
            return self.fuzzy_match_item(bracket_text, all_items)
            
        return None

    def fuzzy_match_party(self, party_name):
        """Fuzzy match party against existing parties"""
        if not party_name:
            return None
            
        clean_name = party_name.lower().strip()
        party_type = self.party_type
        
        # Get all parties of the specified type
        if party_type == "Customer":
            parties = frappe.get_all("Customer", fields=["name", "customer_name"])
            names = [p["customer_name"] for p in parties]
        else:
            parties = frappe.get_all("Supplier", fields=["name", "supplier_name"])
            names = [p["supplier_name"] for p in parties]
            
        # Find best match
        best_match = None
        best_score = 0
        
        for i, name in enumerate(names):
            score = difflib.SequenceMatcher(None, clean_name, name.lower()).ratio() * 100
            if score > best_score:
                best_score = score
                best_match = {
                    "name": parties[i]["name"],
                    "score": score,
                    "match_name": name
                }
        
        # Return match only if above confidence threshold
        return best_match if best_score > 80 else None


@frappe.whitelist()
def extract_invoice(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        result = doc.extract_invoice()
        return result
    except Exception as e:
        frappe.log_error(f"Extract invoice failed: {str(e)}", "Extract Invoice Error")
        return {"status": "error", "message": str(e)}


@frappe.whitelist()
def create_invoice(docname, submit_invoice=False):
    """Create invoice from the Create Invoice button"""
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        # Create draft invoice by default
        doc.create_invoice_from_child(submit_invoice=submit_invoice)
        return {
            "status": "success",
            "invoice_name": doc.invoice_reference,
            "doctype": doc.invoice_type,
            "status": doc.invoice_status
        }
    except Exception as e:
        frappe.log_error(f"Create invoice failed: {str(e)}", "Create Invoice Error")
        return {"status": "error", "message": str(e)}


# Debug method to test OCR safely
@frappe.whitelist()
def debug_ocr_preview(docname):
    try:
        doc = frappe.get_doc("Invoice Upload", docname)
        file_path = get_file_path(doc.file)

        def preprocess_image(pil_img):
            try:
                img = np.array(pil_img.convert("RGB"))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(scaled)
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 10
                )
                kernel = np.ones((3, 3), np.uint8)
                processed = cv2.erode(thresh, kernel, iterations=1)
                return processed
            except Exception as e:
                frappe.log_error(f"Debug image processing failed: {str(e)}", "OCR Debug Error")
                return pil_img

        text = ""
        if file_path.endswith(".pdf"):
            images = convert_from_path(file_path, dpi=300)
            for img in images:
                processed = preprocess_image(img)
                text += pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")
        else:
            img = Image.open(file_path)
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, config="--psm 4 --oem 3 -l eng+urd")

        # Save to document for debugging
        doc.raw_ocr_text = text[:10000]
        doc.save()
        
        return text[:5000]  # Limit output to first 5000 characters
    except Exception as e:
        frappe.log_error(f"OCR debug failed: {str(e)}", "OCR Debug Error")
        return f"Error: {str(e)}"
