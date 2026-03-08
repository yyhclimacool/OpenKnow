"""CLI entry point for Knowledge Database management."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.logging_config import setup_logging
from src.searcher import KnowledgeBase

console = Console()


def cmd_add(args: argparse.Namespace) -> None:
    kb = KnowledgeBase()
    for directory in args.directories:
        path = Path(directory).resolve()
        with console.status(f"[bold green]Indexing {path}..."):
            result = kb.add_directory(path)
        console.print(f"[green]Done:[/green] {result['documents']} documents -> {result['chunks']} chunks")


def cmd_search(args: argparse.Namespace) -> None:
    kb = KnowledgeBase()
    query = " ".join(args.query)
    results = kb.search(query, top_k=args.top_k, source_filter=args.source)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.print(f"\n[bold cyan]--- Result {i} ({r.relevance_score:.0%}) ---[/bold cyan]")
        console.print(f"[dim]Source: {r.source_path}[/dim]")
        if r.heading_chain:
            console.print(f"[dim]Section: {r.heading_chain}[/dim]")
        console.print()
        console.print(r.content[:500] + ("..." if len(r.content) > 500 else ""))


def cmd_status(args: argparse.Namespace) -> None:
    kb = KnowledgeBase()
    stats = kb.get_stats()
    sources = kb.list_sources()

    table = Table(title="Knowledge Database Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total chunks", str(stats["total_chunks"]))
    table.add_row("Total files", str(stats["total_files"]))
    table.add_row("Source directories", str(len(sources)))
    console.print(table)

    if sources:
        console.print("\n[bold]Configured sources:[/bold]")
        for s in sources:
            console.print(f"  [dim]-[/dim] {s}")
    else:
        console.print("\n[yellow]No sources configured. Use 'add' to add directories.[/yellow]")


def cmd_reindex(args: argparse.Namespace) -> None:
    kb = KnowledgeBase()
    sources = kb.list_sources()
    if not sources:
        console.print("[yellow]No sources to reindex.[/yellow]")
        return

    with console.status("[bold green]Reindexing all sources..."):
        results = kb.reindex_all()

    for src, result in results.items():
        console.print(f"[green]{src}:[/green] {result['documents']} docs -> {result['chunks']} chunks")


def cmd_remove(args: argparse.Namespace) -> None:
    kb = KnowledgeBase()
    path = Path(args.directory).resolve()
    deleted = kb.remove_directory(path)
    console.print(f"[green]Removed {deleted} chunks from source: {path}[/green]")


def cmd_serve(args: argparse.Namespace) -> None:
    transport = args.transport
    mcp_script = Path(__file__).resolve().parent / "mcp_server.py"
    cmd = [sys.executable, str(mcp_script), "--transport", transport]

    if transport in ("sse", "streamable-http"):
        cmd += ["--host", args.host, "--port", str(args.port)]
        console.print(f"[bold green]Starting MCP Server ({transport})...[/bold green]")
        console.print(f"[dim]Listening on http://{args.host}:{args.port}[/dim]")
    else:
        console.print("[bold green]Starting MCP Server (stdio)...[/bold green]")

    console.print("[dim]Press Ctrl+C to stop[/dim]")
    subprocess.run(cmd, check=True)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge Database - RAG-based document search system"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_add = subparsers.add_parser("add", help="Add directories to the knowledge base")
    p_add.add_argument("directories", nargs="+", help="Directories to index")

    p_search = subparsers.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument("-s", "--source", help="Filter by source path substring")

    subparsers.add_parser("status", help="Show knowledge base status")

    subparsers.add_parser("reindex", help="Rebuild the index from all sources")

    p_remove = subparsers.add_parser("remove", help="Remove a source directory")
    p_remove.add_argument("directory", help="Directory to remove")

    p_serve = subparsers.add_parser("serve", help="Start the MCP server")
    p_serve.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="Transport protocol (default: streamable-http)",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8787, help="HTTP port (default: 8787)")

    args = parser.parse_args()

    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        console=args.verbose,
    )

    commands = {
        "add": cmd_add,
        "search": cmd_search,
        "status": cmd_status,
        "reindex": cmd_reindex,
        "remove": cmd_remove,
        "serve": cmd_serve,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()
