"""
Example script to process an invoice PDF and save the result as JSON.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import AppSettings as Settings
from src.core.pipeline import EuropeanInvoiceProcessor


async def process_invoice(pdf_path: str, output_path: str = None) -> dict:
    """
    Process a single invoice PDF and return the extracted data as a dictionary.
    
    Args:
        pdf_path: Path to the input PDF file
        output_path: Optional path to save the JSON output
        
    Returns:
        dict: Extracted invoice data
    """
    # Initialize settings and processor
    settings = Settings()
    processor = EuropeanInvoiceProcessor(settings)
    
    try:
        # Initialize the processor
        await processor.initialize()
        
        # Process the invoice
        print(f"Processing invoice: {pdf_path}")
        with open(pdf_path, 'rb') as f:
            invoice_data = await processor.process_document(f.read())
        
        # Convert to dict if it's a Pydantic model
        if hasattr(invoice_data, 'dict'):
            result = invoice_data.dict()
        else:
            result = dict(invoice_data)
        
        # Save the result if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error processing invoice: {e}", file=sys.stderr)
        raise
    finally:
        # Clean up resources
        await processor.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process an invoice PDF and extract data.')
    parser.add_argument('input_pdf', help='Path to the input PDF file')
    parser.add_argument('-o', '--output', help='Path to save the output JSON file')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_pdf))[0]
        args.output = os.path.join('data', 'output', f'{base_name}.json')
    
    # Run the async function
    asyncio.run(process_invoice(args.input_pdf, args.output))
