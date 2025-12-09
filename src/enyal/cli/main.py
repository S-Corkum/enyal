"""Command-line interface for Enyal."""

import argparse
import json
import os
import sys

from enyal.core.retrieval import RetrievalEngine
from enyal.core.store import ContextStore
from enyal.models.context import ContextType, ScopeLevel


def get_store(db_path: str | None = None) -> ContextStore:
    """Get or create a context store."""
    path = db_path or os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")
    return ContextStore(path)


def cmd_remember(args: argparse.Namespace) -> int:
    """Handle the remember command."""
    store = get_store(args.db)

    entry_id = store.remember(
        content=args.content,
        content_type=ContextType(args.type),
        scope_level=ScopeLevel(args.scope),
        scope_path=args.scope_path,
        tags=args.tags.split(",") if args.tags else [],
    )

    if args.json:
        print(json.dumps({"success": True, "entry_id": entry_id}))
    else:
        print(f"Stored with ID: {entry_id}")

    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    """Handle the recall command."""
    store = get_store(args.db)
    retrieval = RetrievalEngine(store)

    results = retrieval.search(
        query=args.query,
        limit=args.limit,
        scope_level=ScopeLevel(args.scope) if args.scope else None,
        scope_path=args.scope_path,
        content_type=ContextType(args.type) if args.type else None,
        min_confidence=args.min_confidence,
    )

    if args.json:
        output = [
            {
                "id": r.entry.id,
                "content": r.entry.content,
                "type": r.entry.content_type.value,
                "scope": r.entry.scope_level.value,
                "score": round(r.score, 4),
                "confidence": r.entry.confidence,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print("No results found.")
            return 0

        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r.entry.content_type.value}] (score: {r.score:.3f})")
            print(f"   {r.entry.content}")
            print(f"   ID: {r.entry.id}")
            if r.entry.tags:
                print(f"   Tags: {', '.join(r.entry.tags)}")

    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    """Handle the forget command."""
    store = get_store(args.db)

    success = store.forget(args.entry_id, hard_delete=args.hard)

    if args.json:
        print(json.dumps({"success": success}))
    else:
        if success:
            action = "permanently deleted" if args.hard else "deprecated"
            print(f"Entry {args.entry_id} has been {action}")
        else:
            print(f"Entry {args.entry_id} not found")
            return 1

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Handle the stats command."""
    store = get_store(args.db)
    stats = store.stats()

    if args.json:
        print(
            json.dumps(
                {
                    "total_entries": stats.total_entries,
                    "active_entries": stats.active_entries,
                    "deprecated_entries": stats.deprecated_entries,
                    "entries_by_type": stats.entries_by_type,
                    "entries_by_scope": stats.entries_by_scope,
                    "avg_confidence": round(stats.avg_confidence, 3),
                    "storage_size_bytes": stats.storage_size_bytes,
                },
                indent=2,
            )
        )
    else:
        print("Enyal Context Store Statistics")
        print("=" * 40)
        print(f"Total entries:      {stats.total_entries}")
        print(f"Active entries:     {stats.active_entries}")
        print(f"Deprecated entries: {stats.deprecated_entries}")
        print(f"Average confidence: {stats.avg_confidence:.2%}")
        print(f"Storage size:       {stats.storage_size_bytes / 1024:.1f} KB")

        if stats.entries_by_type:
            print("\nBy type:")
            for t, count in sorted(stats.entries_by_type.items()):
                print(f"  {t}: {count}")

        if stats.entries_by_scope:
            print("\nBy scope:")
            for s, count in sorted(stats.entries_by_scope.items()):
                print(f"  {s}: {count}")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Handle the serve command (MCP server)."""
    # Set environment variables
    if args.db:
        os.environ["ENYAL_DB_PATH"] = args.db
    if args.preload:
        os.environ["ENYAL_PRELOAD_MODEL"] = "true"
    if args.log_level:
        os.environ["ENYAL_LOG_LEVEL"] = args.log_level

    # Import and run the MCP server
    from enyal.mcp.server import main

    main()
    return 0


def cmd_model_download(args: argparse.Namespace) -> int:
    """Handle the model download command."""
    from enyal.core.ssl_config import download_model

    model_name = args.model or "all-MiniLM-L6-v2"

    try:
        print(f"Downloading model: {model_name}")
        print("This may take a few minutes on first download...")

        path = download_model(model_name, cache_dir=args.cache_dir)

        if args.json:
            print(json.dumps({"success": True, "model": model_name, "path": path}))
        else:
            print("\nModel downloaded successfully!")
            print(f"Model: {model_name}")
            print(f"Path: {path}")
            print("\nThe model is now cached and available for offline use.")
            print("Set ENYAL_OFFLINE_MODE=true to prevent future network calls.")

        return 0

    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}))
        else:
            print(f"\nError downloading model: {e}")
            print("\nTroubleshooting:")
            print("  - For corporate networks with SSL inspection:")
            print("    export ENYAL_SSL_CERT_FILE=/path/to/corporate-ca-bundle.crt")
            print("  - If you cannot obtain the CA bundle (last resort, insecure):")
            print("    export ENYAL_SSL_VERIFY=false")
        return 1


def cmd_model_verify(args: argparse.Namespace) -> int:
    """Handle the model verify command."""
    from enyal.core.ssl_config import verify_model

    model_path = args.model

    print(f"Verifying model: {model_path or 'default (all-MiniLM-L6-v2)'}")

    success = verify_model(model_path)

    if args.json:
        print(json.dumps({"success": success, "model": model_path or "all-MiniLM-L6-v2"}))
    else:
        if success:
            print("\nModel verification successful!")
            print("The model is ready for use.")
        else:
            print("\nModel verification failed!")
            print("Check the error messages above for details.")

    return 0 if success else 1


def cmd_model_status(args: argparse.Namespace) -> int:
    """Handle the model status command."""
    from enyal.core.ssl_config import check_ssl_health

    status = check_ssl_health()

    if args.json:
        print(json.dumps(status, indent=2, default=str))
    else:
        print("Enyal SSL/Network Configuration Status")
        print("=" * 45)
        print(f"SSL verification:     {'Enabled' if status['ssl_verify'] else 'DISABLED (insecure)'}")
        print(f"CA certificate file:  {status['cert_file'] or 'Not set (using system default)'}")
        if status['cert_file']:
            print(f"  File exists:        {'Yes' if status['cert_file_exists'] else 'NO - FILE NOT FOUND'}")
        print(f"System CA bundle:     {status['system_ca_bundle'] or 'Not found'}")
        print(f"Local model path:     {status['model_path'] or 'Not set'}")
        if status['model_path']:
            print(f"  Path exists:        {'Yes' if status['model_path_exists'] else 'NO - PATH NOT FOUND'}")
        print(f"Offline mode:         {'Enabled' if status['offline_mode'] else 'Disabled'}")
        print(f"HF cache directory:   {status['hf_home'] or 'Default (~/.cache/huggingface)'}")
        print()
        print("Library versions:")
        print(f"  huggingface_hub:    {status['huggingface_hub_version'] or 'Not installed'}")
        print(f"  sentence_transformers: {status['sentence_transformers_version'] or 'Not installed'}")

    return 0


def cmd_get(args: argparse.Namespace) -> int:
    """Handle the get command."""
    store = get_store(args.db)
    entry = store.get(args.entry_id)

    if entry is None:
        if args.json:
            print(json.dumps({"error": "Entry not found"}))
        else:
            print(f"Entry {args.entry_id} not found")
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "id": entry.id,
                    "content": entry.content,
                    "type": entry.content_type.value,
                    "scope": entry.scope_level.value,
                    "scope_path": entry.scope_path,
                    "confidence": entry.confidence,
                    "tags": entry.tags,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "access_count": entry.access_count,
                    "is_deprecated": entry.is_deprecated,
                },
                indent=2,
            )
        )
    else:
        print(f"ID:         {entry.id}")
        print(f"Content:    {entry.content}")
        print(f"Type:       {entry.content_type.value}")
        print(f"Scope:      {entry.scope_level.value}")
        if entry.scope_path:
            print(f"Scope path: {entry.scope_path}")
        print(f"Confidence: {entry.confidence:.2%}")
        if entry.tags:
            print(f"Tags:       {', '.join(entry.tags)}")
        print(f"Created:    {entry.created_at.isoformat()}")
        print(f"Updated:    {entry.updated_at.isoformat()}")
        print(f"Accessed:   {entry.access_count} times")
        if entry.is_deprecated:
            print("Status:     DEPRECATED")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="enyal",
        description="Persistent, queryable memory for AI coding agents",
    )
    parser.add_argument(
        "--db",
        help="Path to database file (default: ~/.enyal/context.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # remember command
    remember_parser = subparsers.add_parser(
        "remember",
        help="Store new context",
    )
    remember_parser.add_argument("content", help="The context to store")
    remember_parser.add_argument(
        "--type",
        "-t",
        default="fact",
        choices=["fact", "preference", "decision", "convention", "pattern"],
        help="Content type",
    )
    remember_parser.add_argument(
        "--scope",
        "-s",
        default="project",
        choices=["global", "workspace", "project", "file"],
        help="Scope level",
    )
    remember_parser.add_argument(
        "--scope-path",
        help="Path for scope",
    )
    remember_parser.add_argument(
        "--tags",
        help="Comma-separated tags",
    )
    remember_parser.set_defaults(func=cmd_remember)

    # recall command
    recall_parser = subparsers.add_parser(
        "recall",
        help="Search for context",
    )
    recall_parser.add_argument("query", help="Search query")
    recall_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Maximum results",
    )
    recall_parser.add_argument(
        "--type",
        "-t",
        choices=["fact", "preference", "decision", "convention", "pattern"],
        help="Filter by type",
    )
    recall_parser.add_argument(
        "--scope",
        "-s",
        choices=["global", "workspace", "project", "file"],
        help="Filter by scope",
    )
    recall_parser.add_argument(
        "--scope-path",
        help="Filter by scope path",
    )
    recall_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence threshold",
    )
    recall_parser.set_defaults(func=cmd_recall)

    # forget command
    forget_parser = subparsers.add_parser(
        "forget",
        help="Remove context",
    )
    forget_parser.add_argument("entry_id", help="Entry ID to remove")
    forget_parser.add_argument(
        "--hard",
        action="store_true",
        help="Permanently delete instead of deprecate",
    )
    forget_parser.set_defaults(func=cmd_forget)

    # get command
    get_parser = subparsers.add_parser(
        "get",
        help="Get entry by ID",
    )
    get_parser.add_argument("entry_id", help="Entry ID")
    get_parser.set_defaults(func=cmd_get)

    # stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show statistics",
    )
    stats_parser.set_defaults(func=cmd_stats)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run MCP server",
    )
    serve_parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload embedding model",
    )
    serve_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    serve_parser.set_defaults(func=cmd_serve)

    # model command group
    model_parser = subparsers.add_parser(
        "model",
        help="Model management commands",
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)

    # model download subcommand
    model_download_parser = model_subparsers.add_parser(
        "download",
        help="Download embedding model for offline use",
    )
    model_download_parser.add_argument(
        "--model",
        "-m",
        help="Model name (default: all-MiniLM-L6-v2)",
    )
    model_download_parser.add_argument(
        "--cache-dir",
        help="Custom cache directory",
    )
    model_download_parser.set_defaults(func=cmd_model_download)

    # model verify subcommand
    model_verify_parser = model_subparsers.add_parser(
        "verify",
        help="Verify model can be loaded",
    )
    model_verify_parser.add_argument(
        "--model",
        "-m",
        help="Model path or name to verify",
    )
    model_verify_parser.set_defaults(func=cmd_model_verify)

    # model status subcommand
    model_status_parser = model_subparsers.add_parser(
        "status",
        help="Show SSL/network configuration status",
    )
    model_status_parser.set_defaults(func=cmd_model_status)

    args = parser.parse_args()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
