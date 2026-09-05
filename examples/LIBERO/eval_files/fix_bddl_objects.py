"""Normalize the :objects block in custom-scene BDDL files.

The libero BDDL parser (libero/libero/envs/bddl_utils.py) assigns
``objects[category] = object_list`` each time it sees ``- <category>``. If two
instances of the SAME category are declared on separate lines, the second
assignment overwrites the first and the first instance is silently dropped --
which then crashes placement init with a KeyError.

Canonical LIBERO BDDL groups all instances of one category on a single line::

    (:objects
        akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
        cookies_1 - cookies
        ...
    )

This script rewrites the :objects block of every *.bddl under a directory into
that canonical form (one line per category, instances space-joined, preserving
first-seen order). Everything outside the :objects block is left byte-for-byte
unchanged. :fixtures is left untouched (no collisions exist in the data).

Usage:
    python fix_bddl_objects.py --dir <scene_dir> [--apply]
Without --apply it only reports which files would change.
"""
import argparse
import glob
import os
import re
import sys

# Matches a single "inst1 inst2 ... - category" object line.
OBJ_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<insts>\S+(?:\s+\S+)*?)\s+-\s+(?P<cat>\S+)\s*$")


def normalize_objects_block(lines):
    """Given the list of text lines STRICTLY between the ``(:objects`` header
    line and its closing ``)`` line, return a new list of lines with all
    same-category instances merged onto one line (canonical form).

    Non-object lines (blank / comments / unrecognized) are preserved verbatim
    and in-place; object lines are regrouped and emitted as a contiguous group.
    """
    indent = "    "
    order = []          # category names in first-seen order
    grouped = {}        # category -> [instances] preserving first-seen order
    other = []          # (position, verbatim_line) for non-object lines
    pos = 0
    saw_object = False
    for ln in lines:
        m = OBJ_LINE_RE.match(ln)
        if m:
            saw_object = True
            indent = m.group("indent") or indent
            cat = m.group("cat")
            if cat not in grouped:
                grouped[cat] = []
                order.append(cat)
            for inst in m.group("insts").split():
                if inst not in grouped[cat]:
                    grouped[cat].append(inst)
        else:
            other.append((pos, ln))
        pos += 1

    if not saw_object:
        return lines, False  # nothing to do

    rebuilt = []
    # Re-emit non-object lines first if they all precede objects; otherwise
    # just append them after. In practice the objects block is only object
    # lines, so we keep leading blank/comment lines then the grouped objects.
    # To stay safe, preserve leading non-object lines in place, then objects.
    leading_other = [ln for p, ln in other if p < next(
        (i for i, ln2 in enumerate(lines) if OBJ_LINE_RE.match(ln2)), len(lines))]
    rebuilt.extend(leading_other)
    for cat in order:
        rebuilt.append(f"{indent}{' '.join(grouped[cat])} - {cat}")

    changed = rebuilt != lines
    return rebuilt, changed


def fix_file_text(text):
    """Return (new_text, changed). Only the :objects block is rewritten."""
    lines = text.splitlines(keepends=True)

    # Locate the (:objects ... ) block by line.
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].lstrip().startswith("(:objects"):
            header_idx = i
            # find closing line: first subsequent line whose stripped text == ")"
            close_idx = None
            for j in range(header_idx + 1, n):
                if lines[j].strip() == ")":
                    close_idx = j
                    break
            if close_idx is None:
                return text, False  # malformed; leave untouched
            inner = lines[header_idx + 1:close_idx]
            new_inner, changed = normalize_objects_block(
                [ln.rstrip("\n") for ln in inner]
            )
            if not changed:
                return text, False
            # Reattach newlines; preserve the original line endings of the block.
            new_inner_with_nl = [ln + "\n" for ln in new_inner]
            new_lines = lines[:header_idx + 1] + new_inner_with_nl + lines[close_idx:]
            return "".join(new_lines), True
        i += 1
    return text, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes to disk. Without it, only report.")
    ap.add_argument("--show-diff", action="store_true",
                    help="Print a per-file before/after of the objects block.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "**", "*.bddl"), recursive=True))
    files += sorted(glob.glob(os.path.join(args.dir, "*.bddl")))
    files = sorted(set(files))
    print(f"scanned {len(files)} bddl files under {args.dir}", file=sys.stderr)

    changed_files = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read()
        new_text, changed = fix_file_text(text)
        if not changed:
            continue
        changed_files.append(f)
        if args.show_diff:
            print(f"\n===== {f} =====")
            old_block = re.search(r"\(:objects.*?\)", text, re.S).group(0)
            new_block = re.search(r"\(:objects.*?\)", new_text, re.S).group(0)
            print("--- before\n" + old_block)
            print("--- after\n" + new_block)
        if args.apply:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new_text)

    print(f"\n{'modified' if args.apply else 'would modify'} {len(changed_files)} file(s)",
          file=sys.stderr)
    if not args.apply and changed_files:
        print("(run with --apply to write changes)", file=sys.stderr)


if __name__ == "__main__":
    main()
