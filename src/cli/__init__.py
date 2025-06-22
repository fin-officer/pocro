"""POCRO command line interface."""

import click
from pathlib import Path

@click.group()
def cli():
    """POCRO - European Invoice OCR and Data Extraction Tool."""
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: input_file.json)')
@click.option('--format', '-f', type=click.Choice(['json', 'csv'], case_sensitive=False), default='json',
              help='Output format (default: json)')
def process(input_file, output, format):
    """Process an invoice file.
    
    Args:
        input_file: Path to the input invoice file (PDF, PNG, JPG, etc.)
        output: Path to the output file (default: input_file.json)
        format: Output format (json or csv)
    """
    from src.cli.process_invoice import main
    main(input_file, output, format)

if __name__ == '__main__':
    cli()
