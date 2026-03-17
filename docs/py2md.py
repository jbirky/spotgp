#!/usr/bin/env python3
"""Convert Python tutorial scripts to MyST Markdown for Sphinx docs.

Conventions recognised in the .py source:

- Lines of the form ``# === ... ===`` are treated as section separators
  (they are discarded; the comment line immediately after becomes a heading).
- A comment line directly after a separator (``# Some Title``) becomes an
  ``## <Title>`` heading.
- Consecutive ``#``-comment lines (not separators/headings) are emitted as
  prose paragraphs (the ``# `` prefix is stripped).
- Everything else is collected into fenced ``python`` code blocks.
- Blank lines are preserved for readability.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

SEPARATOR_RE = re.compile(r"^#\s*={3,}\s*$")


def _flush_code(buf: list[str], out: list[str]) -> None:
    """Append a fenced code block from *buf* to *out*, then clear *buf*."""
    # strip trailing blank lines inside the block
    while buf and not buf[-1].strip():
        buf.pop()
    if not buf:
        return
    out.append("")
    out.append("```python")
    out.extend(buf)
    out.append("```")
    out.append("")
    buf.clear()


def convert(src: Path) -> str:
    """Return MyST Markdown text for a Python tutorial script."""
    lines = src.read_text().splitlines()

    out: list[str] = []
    code_buf: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ── separator line ── e.g. ``# =====================``
        if SEPARATOR_RE.match(line):
            _flush_code(code_buf, out)
            i += 1
            # The next non-blank comment line is the heading
            while i < n and not lines[i].strip():
                i += 1
            if i < n and lines[i].startswith("#"):
                heading = lines[i].lstrip("# ").strip()
                out.append("")
                out.append(f"## {heading}")
                out.append("")
                i += 1
            # skip a trailing separator line if present
            if i < n and SEPARATOR_RE.match(lines[i]):
                i += 1
            continue

        # ── comment block (not a separator) ──
        if line.startswith("#") and not SEPARATOR_RE.match(line):
            _flush_code(code_buf, out)
            comment_lines: list[str] = []
            while i < n and lines[i].startswith("#") and not SEPARATOR_RE.match(lines[i]):
                # strip leading "# " or lone "#"
                text = re.sub(r"^#\s?", "", lines[i])
                comment_lines.append(text)
                i += 1
            out.append("")
            out.extend(comment_lines)
            out.append("")
            continue

        # ── blank line ──
        if not line.strip():
            code_buf.append("")
            i += 1
            continue

        # ── code line ──
        code_buf.append(line)
        i += 1

    _flush_code(code_buf, out)

    # Collapse runs of >2 blank lines
    cleaned: list[str] = []
    blank_count = 0
    for ln in out:
        if not ln.strip():
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(ln)
        else:
            blank_count = 0
            cleaned.append(ln)

    return "\n".join(cleaned).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Input .py file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .md file (default: same name with .md extension)",
    )
    args = parser.parse_args()
    dest = args.output or args.src.with_suffix(".md")
    dest.write_text(convert(args.src))
    print(f"{args.src} -> {dest}")


if __name__ == "__main__":
    main()
