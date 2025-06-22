from fpdf import FPDF
import os

# Create output directory if it doesn't exist
os.makedirs("test_data", exist_ok=True)

# Create PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="SAMPLE INVOICE", ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt="Invoice #: INV-2023-001", ln=True)
pdf.cell(200, 10, txt="Date: 2023-11-15", ln=True)
pdf.ln(10)
pdf.cell(200, 10, txt="From: Test Company GmbH", ln=True)
pdf.cell(200, 10, txt="To: Customer Ltd", ln=True)
pdf.ln(10)
pdf.cell(200, 10, txt="Description: Professional Services", ln=True)
pdf.cell(200, 10, txt="Amount: $1,000.00", ln=True)
pdf.cell(200, 10, txt="VAT (19%): $190.00", ln=True)
pdf.cell(200, 10, txt="Total: $1,190.00", ln=True)

# Save the PDF
output_path = "test_data/sample_invoice.pdf"
pdf.output(output_path)
print(f"Test invoice created at: {os.path.abspath(output_path)}")
