frappe.ui.form.on("Invoice OCR Settings", {
	refresh(frm) {
		frm.add_custom_button(__("Verify API"), () => {
			if (frm.is_dirty()) {
				frappe.msgprint(__("Please save settings first, then click Verify API."));
				return;
			}

			frappe.call({
				method: "invoice_ocr.invoice_ocr.doctype.invoice_ocr_settings.invoice_ocr_settings.verify_api_connection",
				freeze: true,
				freeze_message: __("Verifying API connection..."),
				callback: function(r) {
					const result = r.message || {};
					if (result.ok) {
						frappe.msgprint({
							title: __("Verification Passed"),
							indicator: "green",
							message: __("{0} ({1}) is working.", [result.provider || "Provider", result.model || "default model"]),
						});
					}
				},
			});
		});
	},
});
