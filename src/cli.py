"""Click CLI for the KYC multi-provider document processor.

Author: Greg Hamer (https://github.com/databased)
License: MIT
"""

import logging
import sys
from pathlib import Path

import click

from src.clients.factory import get_client
from src.config import ProviderConfigManager
from src.loader import DocumentLoader
from src.processor import BatchProcessor, VisionProcessor


def _setup(provider_id: str | None = None):
    """Wire up all components and return (manager, client, vision, loader).

    Resets the singleton so each invocation gets fresh config.
    """
    ProviderConfigManager.reset()
    manager = ProviderConfigManager()

    # Logging
    log_cfg = manager.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
    )

    config = manager.get_provider_config(provider_id)
    client = get_client(config)
    vision = VisionProcessor(client, manager.get_processing_config())
    loader = DocumentLoader()
    return manager, client, vision, loader


@click.group()
def cli():
    """KYC Multi-Provider Vision Document Processor."""


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--provider", default=None, help="Provider key from providers.yaml")
@click.option("--output", "-o", default=None, type=click.Path(path_type=Path))
def process(image_path: Path, provider: str | None, output: Path | None):
    """Process a single identity document image."""
    manager, client, vision, loader = _setup(provider)

    doc = loader._validate(image_path)
    if not doc.is_valid:
        click.echo(f"Invalid document: {doc.error_message}", err=True)
        sys.exit(1)

    click.echo(
        f"Processing {image_path.name} with "
        f"{client.provider_name()} ({client.model_name()})..."
    )

    result = vision.process(doc)

    if not result.success:
        click.echo(f"Failed: {result.error_message}", err=True)
        sys.exit(1)

    json_out = result.extracted_data.model_dump_json(indent=2)

    if output:
        output.write_text(json_out)
        click.echo(f"Result saved to {output}")
    else:
        click.echo(json_out)


@cli.command()
@click.option("--provider", default=None, help="Provider key from providers.yaml")
@click.option("--input-dir", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", default=None, type=click.Path(path_type=Path))
def batch(
    provider: str | None,
    input_dir: Path | None,
    output_dir: Path | None,
):
    """Batch-process all document images in a directory."""
    manager, client, vision, loader = _setup(provider)

    docs_cfg = manager.get_documents_config()
    in_dir = input_dir or Path(docs_cfg.get("input_directory", "documents"))
    if output_dir:
        docs_cfg["output_directory"] = str(output_dir)
        docs_cfg["individual_output_directory"] = str(output_dir / "individual")

    documents = loader.discover(in_dir)
    valid = [d for d in documents if d.is_valid]

    if not valid:
        click.echo(f"No valid documents found in {in_dir}", err=True)
        sys.exit(1)

    click.echo(
        f"Processing {len(valid)} document(s) with "
        f"{client.provider_name()} ({client.model_name()})..."
    )

    bp = BatchProcessor(vision, docs_cfg)
    summary = bp.run(documents)

    click.echo(
        f"\nDone: {summary.successful_extractions}/{summary.total_documents} "
        f"succeeded ({summary.success_rate:.0f}%) in "
        f"{summary.total_processing_time:.1f}s"
    )


@cli.command("list-providers")
def list_providers():
    """Show all providers and their configuration status."""
    ProviderConfigManager.reset()
    manager = ProviderConfigManager()

    click.echo(f"Default provider: {manager.default_provider}\n")
    click.echo(f"{'Provider':<16} {'Name':<22} {'Status':<12} {'Default Model'}")
    click.echo("-" * 80)

    for pid, cfg in manager.providers.items():
        if cfg.is_configured:
            status = click.style("ready", fg="green")
        else:
            status = click.style("no key", fg="red")
        marker = " *" if pid == manager.default_provider else ""
        click.echo(f"{pid:<16} {cfg.name:<22} {status:<21} {cfg.default_model}{marker}")


@cli.command("test-connection")
@click.option("--provider", default=None, help="Provider key to test")
def test_connection(provider: str | None):
    """Test connectivity to a provider."""
    _, client, _, _ = _setup(provider)

    click.echo(f"Testing {client.provider_name()} ({client.model_name()})...")
    success, msg = client.test_connection()

    if success:
        click.echo(click.style(f"OK: {msg}", fg="green"))
    else:
        click.echo(click.style(f"FAILED: {msg}", fg="red"), err=True)
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
