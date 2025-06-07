# 📄 Invoice OCR App for ERPNext v15

Automatically extract invoice data from scanned **PDF** or **image** files and generate Sales or Purchase Invoices in ERPNext using Tesseract OCR.

**This project is supported by AgroVisions — thank you for powering open source!**
---

## 🚀 Features

- 🔍 OCR extraction using `pytesseract` from PDF or image
- 📄 Parses invoice number, date, line items, and total
- 🧾 Creates:
  - ✅ Sales Invoice (for Customer)
  - ✅ Purchase Invoice (for Supplier)
- 📦 Automatically creates Items if not found
- 🔄 Dynamic Party Link (Customer or Supplier)
- 📂 JSON preview of extracted data for audit

---

## 📁 Doctypes

### `Invoice Upload`

| Field           | Type          | Description                              |
|------------------|---------------|------------------------------------------|
| Party Type       | Select         | Customer / Supplier                      |
| Party            | Dynamic Link   | Links to Customer or Supplier            |
| File             | Attach         | Upload scanned invoice file              |
| OCR Status       | Select         | Pending / Processing / Extracted / Failed |
| Extracted Data   | Code           | Raw JSON preview of OCR results          |
| Create Invoice   | Button         | Manually trigger invoice creation        |

---

## ⚙️ Full Installation Guide

### ✅ 1. Prerequisites

Install required system packages:

```bash
sudo apt update
sudo apt install -y poppler-utils tesseract-ocr tesseract-ocr-urd

# Get the app from GitHub

bench get-app https://github.com/Mohtashim-1/Invoice-OCR.git
# Activate your Frappe virtual environment
source ~/frappe-bench/env/bin/activate

# Install required Python libraries
pip install -r apps/invoice_ocr/requirements.txt

# Or manually install requirements
pip install pytesseract pdf2image Pillow PyPDF2 easyocr opencv-python

# Deactivate virtual enviroment
deactivate

# 4. Install the app on your site
cd ~/frappe-bench
bench --site yoursite.com install-app invoice_ocr

#Apply necessary migrations
bench migrate

#Restart bench or supervisor
bench restart #for production
bench start #for development

#Video tutorials
will be added shortly



