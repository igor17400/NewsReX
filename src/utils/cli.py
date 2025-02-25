from typing import Optional

import click

from utils.cache_manager import CacheManager


@click.group()
def cli() -> None:
    """CLI entry point for cache management."""
    pass


@cli.command()
@click.option("--cache-type", type=click.Choice(["datasets", "embeddings", "all"]))
def clear_cache(cache_type: Optional[str]) -> None:
    """Clear cached files."""
    cache_manager = CacheManager()
    if cache_type == "all":
        cache_type = None
    cache_manager.clear_cache(cache_type)
    click.echo("Cache cleared successfully!")
