#!/usr/bin/env python3
"""Write every captioned file in the Hiring Intelligence guide to its path.

A file in the guide looks like this (caption line, then one fenced block):

    **File:** `hiring_intel/contracts.py`
    ```python
    ...
    ```

Usage (run from your repo root):
    python3 extract_code.py ../hiring-intel-guide/04-llm-gateway.md          # dry run: what would change
    python3 extract_code.py ../hiring-intel-guide/04-llm-gateway.md --write  # write new files
    python3 extract_code.py ../hiring-intel-guide/*.md --write --force        # also overwrite changed files
    python3 extract_code.py ../hiring-intel-guide/*.md --check                # exit 1 if anything differs

Safety: refuses absolute paths and '..'; never overwrites a file that differs unless --force.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CAPTION = re.compile(r"^\*\*File:\*\* `([^`]+)`\s*$")
FENCE = re.compile(r"^(`{3,}|~{3,})([^`\s]*)\s*$")


def parse(md: str) -> list[tuple[str, str]]:
    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        cap = CAPTION.match(line)
        fence = FENCE.match(line)
        if cap:
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            f = FENCE.match(lines[j]) if j < n else None
            if not f:
                raise SystemExit(f"caption without a code block: {cap.group(1)} (line {i + 1})")
            marker = f.group(1)
            k = j + 1
            while k < n and lines[k].rstrip() != marker:
                k += 1
            if k >= n:
                raise SystemExit(f"unclosed code block for {cap.group(1)}")
            body = "\n".join(lines[j + 1:k])
            out.append((cap.group(1), body + "\n" if k > j + 1 else ""))
            i = k + 1
        elif fence:                                   # skip any other code block (examples, commands)
            marker = fence.group(1)
            k = i + 1
            while k < n and lines[k].rstrip() != marker:
                k += 1
            i = k + 1
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapters", nargs="+")
    ap.add_argument("--write", action="store_true", help="write new files")
    ap.add_argument("--force", action="store_true", help="also overwrite files that differ")
    ap.add_argument("--check", action="store_true", help="exit 1 if any file is missing or differs")
    ap.add_argument("--root", default=".", help="repo root (default: current folder)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    counts = {"new": 0, "same": 0, "differs": 0, "written": 0}
    for ch in a.chapters:
        for rel, body in parse(Path(ch).read_text(encoding="utf-8")):
            p = Path(rel)
            if p.is_absolute() or ".." in p.parts:
                raise SystemExit(f"refusing unsafe path: {rel}")
            dest = root / p
            if dest.exists():
                state = "same" if dest.read_text(encoding="utf-8") == body else "differs"
            else:
                state = "new"
            counts[state] += 1
            action = ""
            if a.write and (state == "new" or (state == "differs" and a.force)):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body, encoding="utf-8")
                counts["written"] += 1
                action = " -> written"
            elif state == "differs" and a.write:
                action = " -> kept (use --force to overwrite)"
            if state != "same" or not a.check:
                print(f"{state.upper():8} {rel}{action}")
    print(f"\n{counts['new']} new, {counts['differs']} differ, {counts['same']} unchanged, {counts['written']} written")
    return 1 if a.check and (counts["new"] or counts["differs"]) else 0


if __name__ == "__main__":
    sys.exit(main())
