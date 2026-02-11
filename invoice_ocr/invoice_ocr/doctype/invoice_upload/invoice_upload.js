// Copyright (c) 2025, mohtashim and contributors
// For license information, please see license.txt

frappe.ui.form.on('Invoice Upload', {
    create_invoice: function (frm) {
        frappe.call({
            method: "invoice_ocr.utils.create_invoice",
            args: { name: frm.doc.name },
            callback: function (r) {
                if (r.message) {
                    frappe.set_route("Form", r.message.doctype, r.message.name);
                }
            }
        });
    }
});

frappe.ui.form.on("Invoice Upload", {
  refresh(frm) {
    // if (!frm.is_new() && frm.doc.ocr_status !== "Extracted") {
      const extract_btn = frm.add_custom_button("Extract from File", function () {
        if (frm.__is_extracting) return;
        frm.__is_extracting = true;
        extract_btn.prop("disabled", true);

        frappe.call({
          method: "invoice_ocr.invoice_ocr.doctype.invoice_upload.invoice_upload.extract_invoice",
          args: { docname: frm.doc.name },
          freeze: true,
          freeze_message: __("Extracting invoice data from file..."),
          callback: function (r) {
              console.log("=== INVOICE OCR DEBUG START ===");
              console.log("[Invoice OCR] full response:", JSON.stringify(r.message, null, 2));
              if (r && r.message) {
                  var msg = r.message;
                  var dbg = msg.debug || {};
                  console.log("[OCR] file_path:", dbg.file_path);
                  console.log("[OCR] ocr_text_length:", dbg.ocr_text_length);
                  console.log("[OCR] ocr_text_preview:", dbg.ocr_text_preview);
                  console.log("[OCR] image_pages_count:", dbg.image_pages_count);
                  console.log("[LLM] source used:", dbg.llm_source);
                  console.log("[LLM] raw data from LLM:", JSON.stringify(dbg.llm_data_raw, null, 2));
                  console.log("[LLM] raw items count:", dbg.llm_items_raw_count);
                  console.log("[LLM] raw items:", JSON.stringify(dbg.llm_items_raw, null, 2));
                  console.log("[LLM] items after normalize:", JSON.stringify(dbg.llm_items_after_normalize, null, 2));
                  console.log("[LLM] items after normalize count:", dbg.llm_items_after_normalize_count);
                  console.log("[LLM] supplier_name:", dbg.llm_supplier_name);
                  console.log("[FINAL] items_count:", dbg.final_items_count);
                  console.log("[FINAL] items_source:", dbg.final_items_source);
                  console.log("[FINAL] items:", JSON.stringify(dbg.final_items, null, 2));
                  console.log("[FINAL] party:", dbg.final_party);
                  console.log("[FINAL] party_type:", dbg.final_party_type);
                  if (dbg.llm_data_is_null) {
                      console.log("[WARNING] LLM returned NULL - fell back to OCR extraction");
                  }
                  if (dbg.llm_items_empty_falling_back_to_ocr) {
                      console.log("[WARNING] LLM returned items but normalize removed them all - fell back to OCR");
                  }
                  console.log("[RESULT] added_rows:", JSON.stringify(msg.added_rows, null, 2));
              }
              console.log("=== INVOICE OCR DEBUG END ===");
              frm.reload_doc();
          },
          always: function () {
              frm.__is_extracting = false;
              extract_btn.prop("disabled", false);
          }
        });
      });
    // }
  }
});
