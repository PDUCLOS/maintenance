"""Manual collection management for ChromaDB.

The ingest pipeline never auto-resets a collection — that's a destructive
operation that requires an explicit human decision. This CLI provides
the safe, auditable interface for managing the collection lifecycle:

    python scripts/manage_collection.py list
    python scripts/manage_collection.py info <name>
    python scripts/manage_collection.py new <name>     # create empty collection
    python scripts/manage_collection.py drop <name>     # with confirmation
    python scripts/manage_collection.py use <name>      # update .env

All operations are non-destructive by default except `drop`, which asks
for an explicit "yes" before deleting anything.

Typical migration workflow (e.g. bge-small 384-dim → bge-m3 1024-dim):

    # 1. Inspect current state
    python scripts/manage_collection.py list
    python scripts/manage_collection.py info cmapss_kb

    # 2. Create a new collection alongside (your old one stays)
    python scripts/manage_collection.py new cmapss_kb_bge_m3
    python scripts/manage_collection.py use cmapss_kb_bge_m3

    # 3. Re-ingest (writes to the new collection)
    make ingest

    # 4. Verify, then drop the old one (irreversible!)
    python scripts/manage_collection.py info cmapss_kb_bge_m3
    python scripts/manage_collection.py drop cmapss_kb    # only after you've validated
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

# Project root: 2 levels up from this script (scripts/ → rag-copilot/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _client() -> chromadb.HttpClient:
    """Create a Chroma HTTP client from the current .env (or defaults)."""
    from src.config import settings  # local import to avoid path issues when run as script

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def _settings_active_collection() -> str:
    from src.config import settings

    return settings.chroma_collection


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Tiny table printer (no external dep)."""
    if not rows:
        print(f"  (no rows; headers: {headers})")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  "
    print("  " + sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  " + sep.join("-" * w for w in widths))
    for row in rows:
        print("  " + sep.join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(_args: argparse.Namespace) -> int:
    """List all collections with their dim and count."""
    client = _client()
    active = _settings_active_collection()
    colls = client.list_collections()
    if not colls:
        print("No collections found.")
        return 0
    rows: list[list[str]] = []
    for c in colls:
        marker = "*" if c.name == active else " "
        try:
            n = c.count()
        except Exception as e:  # noqa: BLE001
            n = f"err: {e}"
        try:
            dim = c.dimension or "?"
        except Exception:
            dim = "?"
        rows.append([marker, c.name, str(dim), str(n)])
    _print_table(
        ["", "NAME", "DIM", "COUNT"],
        rows,
    )
    print(f"\n  * = active collection (settings.chroma_collection = '{active}')")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show details for one collection."""
    name = args.name
    client = _client()
    try:
        coll = client.get_collection(name)
    except NotFoundError:
        print(f"❌ Collection '{name}' not found.")
        print("   Use 'list' to see available collections.")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"❌ Cannot reach ChromaDB: {e}")
        return 1
    n = coll.count()
    dim = coll.dimension
    meta = coll.metadata or {}
    active = _settings_active_collection()
    print(f"Collection: {name}")
    print(f"  ID:        {coll.id}")
    print(f"  Dim:       {dim}")
    print(f"  Count:     {n} chunks")
    print(f"  Metadata:  {meta}")
    print(f"  Active:    {'YES (used by the RAG chain)' if name == active else 'no'}")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    """Create an empty collection."""
    name = args.name
    client = _client()
    try:
        existing = client.get_collection(name)
        print(f"❌ Collection '{name}' already exists ({existing.count()} chunks).")
        print("   Use 'drop' first if you really want to recreate it.")
        return 1
    except NotFoundError:
        pass  # good, doesn't exist
    except Exception as e:  # noqa: BLE001
        print(f"❌ Cannot reach ChromaDB: {e}")
        return 1

    coll = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"✓ Created empty collection '{name}' (id={coll.id}).")
    print(f"  Dim is set automatically on first upsert (current setting: see .env).")
    print()
    print("Next steps:")
    print(f"  1. Set it as active:    python scripts/manage_collection.py use {name}")
    print(f"  2. Re-ingest:           make ingest")
    print(f"  3. (optional) drop the old one when satisfied:")
    print(f"     python scripts/manage_collection.py drop <old-name>")
    return 0


def cmd_drop(args: argparse.Namespace) -> int:
    """Drop a collection. Requires --yes flag (no surprise deletion)."""
    name = args.name
    if not args.yes:
        print(f"⚠️  About to DELETE collection '{name}' (IRREVERSIBLE).")
        print("   All vectors, documents, and metadata in that collection will be lost.")
        print("   The source data (CMAPSS + PDFs) is NOT affected — you can rebuild with 'make ingest'.")
        print()
        print("   If you're sure, re-run with --yes:")
        print(f"     python scripts/manage_collection.py drop {name} --yes")
        return 2

    client = _client()
    try:
        coll = client.get_collection(name)
        n = coll.count()
    except NotFoundError:
        print(f"Collection '{name}' not found (already dropped?). Nothing to do.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"❌ Cannot reach ChromaDB: {e}")
        return 1

    active = _settings_active_collection()
    if name == active:
        print(f"❌ Refusing to drop the ACTIVE collection ('{name}').")
        print("   Switch to another collection first: python scripts/manage_collection.py use <other>")
        return 1

    print(f"Dropping '{name}' ({n} chunks)…")
    client.delete_collection(name)
    print(f"✓ Dropped '{name}'.")
    return 0


def cmd_use(args: argparse.Namespace) -> int:
    """Update CHROMA_COLLECTION in .env to the given name."""
    name = args.name

    # Validate that the collection exists (or warn clearly)
    client = _client()
    try:
        client.get_collection(name)
    except NotFoundError:
        print(f"⚠️  Collection '{name}' does not exist yet.")
        print("   Run 'python scripts/manage_collection.py new <name>' first,")
        print("   or check the name with 'list'.")
        if not args.force:
            return 1
        print("   --force given, updating .env anyway.")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Cannot reach ChromaDB: {e}")
        if not args.force:
            return 1

    # Update .env
    if not ENV_PATH.exists():
        print(f"❌ .env not found at {ENV_PATH}. Create one from .env.example first.")
        return 1

    content = ENV_PATH.read_text(encoding="utf-8")
    new_lines: list[str] = []
    found = False
    for line in content.splitlines():
        if line.strip().startswith("CHROMA_COLLECTION="):
            new_lines.append(f"CHROMA_COLLECTION={name}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        # Append
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"CHROMA_COLLECTION={name}")

    new_content = "\n".join(new_lines) + "\n"
    ENV_PATH.write_text(new_content, encoding="utf-8")
    print(f"✓ Updated {ENV_PATH.name}: CHROMA_COLLECTION={name}")
    print("  Restart the API/UI to pick up the new value:")
    print("    make api")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manual ChromaDB collection management (no auto-reset).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser("list", help="List all collections with dim and count.")

    p_info = sub.add_parser("info", help="Show details for one collection.")
    p_info.add_argument("name", help="Collection name.")

    p_new = sub.add_parser("new", help="Create an empty collection.")
    p_new.add_argument("name", help="New collection name.")

    p_drop = sub.add_parser("drop", help="Drop a collection (IRREVERSIBLE).")
    p_drop.add_argument("name", help="Collection name.")
    p_drop.add_argument("--yes", action="store_true",
                       help="Required: confirm the deletion.")

    p_use = sub.add_parser("use", help="Set the active collection in .env.")
    p_use.add_argument("name", help="Collection name to make active.")
    p_use.add_argument("--force", action="store_true",
                      help="Update .env even if the collection doesn't exist yet.")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "list": cmd_list,
        "info": cmd_info,
        "new": cmd_new,
        "drop": cmd_drop,
        "use": cmd_use,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
