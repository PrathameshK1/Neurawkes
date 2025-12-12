"""Mechanical modernization for this repo (Python 3.13 syntax).

This project contains legacy Python-2-era syntax (notably `print ...`) which is
invalid on Python 3. Since Python 3.13 removed `lib2to3`, we do a conservative
text-based rewrite:

- `print <expr>` -> `print(<expr>)`
- `xrange(` -> `range(`

The rewrite is intentionally minimal and aims to preserve runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable


_PRINT_STMT_RE = re.compile(r"^(?P<indent>[ \t]*)print(?P<ws>\s+)(?P<body>.*)$")


@dataclass(frozen=True)
class RewriteResult:
    changed: bool
    new_line: str


def _split_code_and_comment(line: str) -> tuple[str, str]:
    """Split `line` into (code, comment) where comment starts at first # outside quotes."""
    in_single = False
    in_double = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            return line[:i], line[i:]
    return line, ""


def rewrite_line(line: str) -> RewriteResult:
    # Fast path: `xrange(` is always safe to replace in this repo.
    if "xrange(" in line:
        return RewriteResult(changed=True, new_line=line.replace("xrange(", "range("))

    m = _PRINT_STMT_RE.match(line)
    if not m:
        return RewriteResult(changed=False, new_line=line)

    code, comment = _split_code_and_comment(m.group("body"))
    body = code.strip()

    # Already function-style print or empty? Leave as-is / normalize empty.
    if body.startswith("("):
        return RewriteResult(changed=False, new_line=line)

    if body == "":
        new_body = "print()"
    else:
        new_body = f"print({body})"

    return RewriteResult(
        changed=True,
        new_line=f"{m.group('indent')}{new_body}{comment}",
    )


def rewrite_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    changed_any = False
    out: list[str] = []
    for line in lines:
        # Preserve exact line ending by keeping the existing `\n` in `line`.
        ending = ""
        if line.endswith("\r\n"):
            core = line[:-2]
            ending = "\r\n"
        elif line.endswith("\n"):
            core = line[:-1]
            ending = "\n"
        else:
            core = line

        res = rewrite_line(core)
        changed_any |= res.changed
        out.append(res.new_line + ending)

    if changed_any:
        path.write_text("".join(out), encoding="utf-8")
    return changed_any


def iter_py_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (f for f in p.rglob("*.py") if f.is_file())
        elif p.is_file() and p.suffix == ".py":
            yield p


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python tools/py313_modernize.py <path> [<path> ...]")
        return 2

    changed = 0
    for f in sorted(set(iter_py_files([Path(a) for a in argv[1:]]))):
        if rewrite_file(f):
            changed += 1

    print(f"Updated {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


