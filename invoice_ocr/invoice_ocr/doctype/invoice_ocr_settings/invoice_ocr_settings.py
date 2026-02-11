import frappe
import requests
from frappe.model.document import Document


class InvoiceOCRSettings(Document):
    pass


def _response_error_details(response):
    try:
        data = response.json()
        error = data.get("error") or {}
        if isinstance(error, dict):
            return error.get("message") or error.get("status") or str(data)
        return str(error or data)
    except Exception:
        return (response.text or "").strip()[:500]


@frappe.whitelist()
def verify_api_connection():
    settings = frappe.get_single("Invoice OCR Settings")
    provider = (settings.llm_provider or "Gemini").strip().lower()

    if provider == "claude":
        api_key = settings.get_password("claude_api_key")
        model = settings.claude_model or "claude-sonnet-4-5"
        if not api_key:
            frappe.throw("Claude API Key is missing in Invoice OCR Settings.")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
        }
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=20,
            )
            if not response.ok:
                details = _response_error_details(response)
                frappe.throw(f"Claude verification failed ({response.status_code}): {details}")
        except Exception:
            frappe.log_error(frappe.get_traceback(), "Invoice OCR Claude verification failed")
            raise

        return {
            "ok": True,
            "provider": "Claude",
            "model": model,
            "message": "Claude API verified successfully.",
        }

    api_key = settings.get_password("gemini_api_key")
    model = settings.gemini_model or "gemini-2.0-flash"
    if not api_key:
        frappe.throw("Gemini API Key is missing in Invoice OCR Settings.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with OK only."}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8},
    }
    try:
        response = requests.post(url, json=payload, timeout=20)
        if not response.ok:
            details = _response_error_details(response)
            frappe.throw(f"Gemini verification failed ({response.status_code}): {details}")
    except Exception:
        frappe.log_error(frappe.get_traceback(), "Invoice OCR Gemini verification failed")
        raise

    return {
        "ok": True,
        "provider": "Gemini",
        "model": model,
        "message": "Gemini API verified successfully.",
    }
