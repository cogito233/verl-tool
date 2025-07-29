#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compactly outline the key/element hierarchy of JSON or JSONL files.

Rules
-----
1. If a value is **str**, replace it with "...".
2. If a value is **list**, show the first five elements' outlines; append "..." if the list is longer.
3. If a value is **dict**, list **all** keys and recursively outline the values.
4. For JSONL, union‑merge the outlines of every line.
"""

from __future__ import annotations
import argparse, json, pathlib, itertools
from typing import Any, Dict, List, Iterable, Union

# --------------------------------------------------------------------------- #
# Core outline builder
# --------------------------------------------------------------------------- #
Primitive = Union[str, int, float, bool, None]
Outline    = Union["OutlineDict", "OutlineList", str]       # recursion types
OutlineDict  = Dict[str, Outline]
OutlineList  = List[Outline]

def build_outline(value: Any) -> Outline:
    """Return an outline that follows the rules above."""
    if isinstance(value, dict):
        return {k: build_outline(v) for k, v in value.items()}

    if isinstance(value, list):
        outlined: OutlineList = [build_outline(v) for v in value[:5]]
        if len(value) > 5:
            outlined.append("...")  # indicate truncation
        return outlined

    # string → hide content; other primitives treated the same
    if isinstance(value, (str, bytes)) and len(value) < 10:
        return value
    return value[:10] + "..." if isinstance(value, (str, bytes)) else "..."

# --------------------------------------------------------------------------- #
# Outline merger for JSONL (dict‑wise deep union; list‑wise position union)
# --------------------------------------------------------------------------- #
def merge(a: Outline, b: Outline) -> Outline:
    """Deep‑merge outline *b* into *a* and return *a* (in‑place for dict/list)."""
    # Different types → prefer first non‑"..." outline, else keep "..."
    if type(a) is not type(b):
        return a if a != "..." else b

    if isinstance(a, dict):
        for k, vb in b.items():     # type: ignore[assignment]
            a[k] = merge(a.get(k, "..."), vb)
        return a

    if isinstance(a, list):
        # merge element‑wise up to len(a) || len(b)
        longest = max(len(a), len(b))  # type: ignore[arg-type]
        result: OutlineList = []
        for i in range(longest):
            if i < len(a) and i < len(b):
                result.append(merge(a[i], b[i]))
            elif i < len(a):
                result.append(a[i])
            else:
                result.append(b[i])
        return result

    # primitives → identical type (both "..."), keep either
    return a

# --------------------------------------------------------------------------- #
# Helpers to load JSON / JSONL
# --------------------------------------------------------------------------- #
def load_jsonl(path: pathlib.Path, limit: int | None = None) -> Iterable[Any]:
    with path.open("r", encoding="utf-8") as fh:
        for line in itertools.islice(fh, limit):
            line = line.strip()
            if line:
                yield json.loads(line)

def outline_file(path: pathlib.Path, num_lines: int | None = None) -> Outline:
    """Generate the merged outline for a JSON or JSONL file."""
    if path.suffix.lower() == ".jsonl":
        outline: Outline = {}
        for obj in load_jsonl(path, num_lines):
            outline = merge(outline, build_outline(obj))
        return outline
    else:
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        return build_outline(obj)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Print key hierarchy outline of JSON/JSONL.")
    ap.add_argument("file", type=pathlib.Path, help="Path to .json or .jsonl")
    ap.add_argument("-n", "--num-lines", type=int, help="Sample first N lines for JSONL")
    args = ap.parse_args()

    outline = outline_file(args.file, args.num_lines)
    print(json.dumps(outline, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
