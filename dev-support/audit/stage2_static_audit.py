#!/usr/bin/env python3
"""Stage 2 static audit for placeholders, stubs, and mock implementations.

This script performs three sweeps over the repository:

1. Marker sweep (TODO/FIXME/etc.) across source, script, and tooling files.
2. Heuristic AST-like scan to identify functions with empty bodies or
   constant return values that look like unimplemented stubs.
3. Search for conditional compilation branches that gate potential mock or
   fallback paths (e.g., GPU fallbacks, mock toggles).

The output is a Markdown report grouped by category, including filenames,
line numbers, and code excerpts. Only the Python standard library is used so
that the script can run in minimal environments.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import re
import textwrap
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

MARKER_PATTERN = re.compile(
    r"\b(?:TODO|FIXME|XXX|HACK|PLACEHOLDER|STUB|NOT[\s_-]*IMPLEMENTED|"
    r"UNIMPLEMENTED|TBD|TBW|TEMPORARY)\b",
    re.IGNORECASE,
)

CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "else",
    "sizeof",
    "return",
    "co_await",
    "co_yield",
    "co_return",
}

TRAILING_SPECIFIERS = {
    "const",
    "noexcept",
    "override",
    "final",
    "requires",
    "volatile",
    "constexpr",
    "[[nodiscard]]",
}

TYPE_QUALIFIERS = {
    "static",
    "inline",
    "constexpr",
    "consteval",
    "constinit",
    "extern",
    "friend",
    "virtual",
    "explicit",
    "typename",
    "using",
    "register",
    "mutable",
}

CONDITIONAL_PRIMARY_KEYWORDS = {
    "MOCK",
    "STUB",
    "PLACEHOLDER",
    "FAKE",
    "DUMMY",
    "SIM",
    "EMU",
    "EMULATION",
    "TEST",
    "FALLBACK",
}

CONDITIONAL_GPU_KEYWORDS = {
    "GPU",
    "CUDA",
    "__CUDA",
    "__CUDA__",
    "__CUDA_ARCH__",
    "__CUDACC__",
    "CUDA_ARCH",
    "DEVICE",
    "GPU_BACKEND",
    "__HIP__",
    "HIP",
}

CONDITIONAL_CPU_KEYWORDS = {
    "CPU_ONLY",
    "CPU_BACKEND",
    "CPU_FALLBACK",
    "CPU_PATH",
    "CPU_MODE",
    "NO_GPU",
    "NOGPU",
    "HOST_ONLY",
    "HOST_FALLBACK",
    "SOFTWARE_FALLBACK",
    "SOFTWARE_BACKEND",
    "SW_FALLBACK",
    "SW_BACKEND",
}

SIMPLE_RETURN_LITERALS = {
    "0",
    "1",
    "-1",
    "false",
    "true",
    "nullptr",
    "NULL",
    "0.0",
    "0.0f",
    "{}",
    "{0}",
    "std::nullopt",
}

MULTI_CHAR_TOKENS = {
    "::",
    "->",
    "++",
    "--",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "<<",
    ">>",
    "&=",
    "|=",
    "^=",
    "->*",
    ".*",
}

SINGLE_CHAR_TOKENS = {
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ";",
    ",",
    ":",
    "*",
    "&",
    "<",
    ">",
    "=",
    "+",
    "-",
    "/",
    "%",
    "^",
    "|",
    "~",
    ".",
    "?",
    "#",
}

Token = namedtuple("Token", "kind value start end")


@dataclass
class MarkerHit:
    file: Path
    line: int
    marker: str
    text: str


@dataclass
class StubHit:
    file: Path
    line: int
    name: str
    stub_type: str
    signature: str
    body_excerpt: str
    header_refs: List[Dict[str, str]]


@dataclass
class ConditionalHit:
    file: Path
    line: int
    directive: str
    keywords: Tuple[str, ...]
    context: str


SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".hpp",
    ".hh",
    ".hxx",
    ".h",
    ".inl",
    ".ipp",
}

SCRIPT_EXTENSIONS = {
    ".py",
    ".sh",
    ".bash",
    ".ps1",
}


def iter_files(root: Path, extensions: Sequence[str]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def sanitize_preserve_positions(source: str) -> str:
    chars = list(source)
    length = len(chars)
    i = 0
    while i < length:
        ch = chars[i]
        if ch == "/" and i + 1 < length:
            nxt = chars[i + 1]
            if nxt == "/":
                j = i
                while j < length and chars[j] != "\n":
                    chars[j] = " "
                    j += 1
                i = j
                continue
            if nxt == "*":
                j = i
                chars[j] = " "
                j += 1
                chars[j] = " "
                j += 1
                while j + 1 < length and not (chars[j] == "*" and chars[j + 1] == "/"):
                    chars[j] = " "
                    j += 1
                if j + 1 < length:
                    chars[j] = " "
                    chars[j + 1] = " "
                    j += 2
                i = j
                continue
        if ch in {'"', "'"}:
            quote = ch
            j = i + 1
            while j < length:
                current = chars[j]
                if current == "\\":
                    chars[j] = " "
                    if j + 1 < length:
                        chars[j + 1] = " "
                    j += 2
                    continue
                if current == quote:
                    break
                chars[j] = " "
                j += 1
            i = j + 1
            continue
        if ch == "R" and i + 1 < length and chars[i + 1] == '"':
            # Crude handling for raw strings: blank until next )"
            j = i + 2
            depth = 1
            while j + 1 < length and depth > 0:
                if chars[j] == '"' and chars[j + 1] == '(':  # nested raw strings unlikely
                    depth += 1
                if chars[j] == ')' and chars[j + 1] == '"':
                    depth -= 1
                chars[j] = " "
                j += 1
            i = j + 2
            continue
        i += 1
    return "".join(chars)


def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    length = len(code)
    i = 0
    while i < length:
        ch = code[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "_" or ch.isalpha() or ch == "~":
            start = i
            i += 1
            while i < length and (code[i].isalnum() or code[i] in {"_", ":"}):
                i += 1
            tokens.append(Token("identifier", code[start:i], start, i))
            continue
        if ch.isdigit():
            start = i
            i += 1
            while i < length and (
                code[i].isalnum() or code[i] in {"_", ".", "x", "X", "b", "B", "e", "E", "+", "-"}
            ):
                i += 1
            tokens.append(Token("number", code[start:i], start, i))
            continue
        if i + 1 < length:
            two = code[i : i + 2]
            if two in MULTI_CHAR_TOKENS:
                tokens.append(Token("operator", two, i, i + 2))
                i += 2
                continue
        if ch in SINGLE_CHAR_TOKENS:
            tokens.append(Token("operator", ch, i, i + 1))
            i += 1
            continue
        tokens.append(Token("other", ch, i, i + 1))
        i += 1
    return tokens


def build_line_offsets(text: str) -> List[int]:
    offsets = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def line_from_offset(offsets: Sequence[int], index: int) -> int:
    return bisect.bisect_right(offsets, index)


def classify_body(body: str) -> Optional[str]:
    compact = "".join(ch for ch in body if not ch.isspace())
    if not compact:
        return "empty-body"

    normalized = " ".join(body.strip().split())
    if normalized.startswith("return"):
        expr = normalized[len("return") :].strip()
        if expr.endswith(";"):
            expr = expr[:-1].strip()
        if expr in SIMPLE_RETURN_LITERALS:
            return "constant-return"
        if re.fullmatch(r"[-+]?0[xX][0-9A-Fa-f]+", expr):
            return "constant-return"
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", expr):
            return "constant-return"
        if expr.startswith("std::") and "(" not in expr:
            return "constant-return"
    if normalized in {"return;"}:
        return "empty-return"
    return None


def find_matching(tokens: Sequence[Token], start_index: int, open_sym: str, close_sym: str) -> Optional[int]:
    depth = 0
    for idx in range(start_index, -1, -1):
        token = tokens[idx]
        if token.value == open_sym:
            depth -= 1
            if depth < 0:
                return idx
        elif token.value == close_sym:
            depth += 1
    return None


def find_matching_brace(text: str, open_index: int) -> Optional[int]:
    depth = 1
    i = open_index + 1
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def find_declaration_start(tokens: Sequence[Token], idx: int) -> int:
    start_idx = idx
    i = idx
    while i > 0:
        prev = tokens[i - 1]
        if prev.value in {";", "}", "{", "#"}:
            break
        start_idx = i - 1
        i -= 1
    return start_idx


def find_function_name(tokens: Sequence[Token], paren_open_idx: int) -> Optional[int]:
    i = paren_open_idx - 1
    while i >= 0:
        token = tokens[i]
        if token.kind == "identifier":
            if token.value in TYPE_QUALIFIERS:
                break
            return i
        if token.value in {"::", "*", "&", "&&", ":"}:
            i -= 1
            continue
        if token.value == "~":
            # destructor name is part of identifier token already, but keep scanning
            i -= 1
            continue
        break
    return None


def detect_stub_functions(
    file_path: Path,
    original: str,
    sanitized: str,
    tokens: Sequence[Token],
    line_offsets: Sequence[int],
    header_index: Dict[str, List[Dict[str, str]]],
) -> List[StubHit]:
    results: List[StubHit] = []
    for idx, token in enumerate(tokens):
        if token.value != "{":
            continue
        # locate matching parenthesis
        paren_close_idx = None
        j = idx - 1
        while j >= 0:
            val = tokens[j].value
            if val in {";", "{", "}"}:
                break
            if val == ")":
                paren_close_idx = j
                break
            j -= 1
        if paren_close_idx is None:
            continue
        # ensure trailing specifiers between ) and { are acceptable
        trailing_valid = True
        for k in range(paren_close_idx + 1, idx):
            val = tokens[k].value
            if val in {",", ";"}:
                trailing_valid = False
                break
            if tokens[k].kind == "identifier" and (
                val in TRAILING_SPECIFIERS or val not in CONTROL_KEYWORDS
            ):
                continue
            if tokens[k].value in {"->", "::", "*", "&", "&&", "<", ">", "[", "]", "(", ")"}:
                continue
            if tokens[k].kind in {"identifier", "number"}:
                continue
            if tokens[k].value == "=":
                trailing_valid = False
                break
        if not trailing_valid:
            continue
        paren_open_idx = find_matching(tokens, paren_close_idx, "(", ")")
        if paren_open_idx is None:
            continue
        name_idx = find_function_name(tokens, paren_open_idx)
        if name_idx is None:
            continue
        name_token = tokens[name_idx]
        base_name = name_token.value.lstrip("~")
        if base_name in CONTROL_KEYWORDS or base_name == "template":
            continue
        body_end = find_matching_brace(sanitized, token.start)
        if body_end is None:
            continue
        body_slice = sanitized[token.start + 1 : body_end]
        stub_type = classify_body(body_slice)
        if stub_type is None:
            continue
        decl_start_idx = find_declaration_start(tokens, name_idx)
        decl_start_pos = tokens[decl_start_idx].start
        signature_text = original[decl_start_pos:token.start].strip()
        if not signature_text:
            signature_text = original[: token.start].strip()
        signature_text = trim_signature(signature_text)
        body_excerpt = extract_body_excerpt(original, token.start + 1, body_end)
        func_line = line_from_offset(line_offsets, name_token.start)
        header_refs = list(header_index.get(base_name, []))
        results.append(
            StubHit(
                file=file_path,
                line=func_line,
                name=name_token.value,
                stub_type=stub_type,
                signature=signature_text,
                body_excerpt=body_excerpt,
                header_refs=header_refs,
            )
        )
    return results


def trim_signature(signature_text: str, max_lines: int = 6) -> str:
    lines = [line.rstrip() for line in signature_text.splitlines()]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines).strip()


def extract_body_excerpt(original: str, body_start: int, body_end: int, max_lines: int = 6) -> str:
    segment = original[body_start:body_end]
    if not segment.strip():
        return segment.strip()
    lines = [line.rstrip() for line in segment.splitlines()]
    excerpt: List[str] = []
    non_empty = 0
    for line in lines:
        excerpt.append(line)
        if line.strip():
            non_empty += 1
        if non_empty >= max_lines:
            break
    return "\n".join(excerpt).strip()


def find_markers(file_path: Path, text: str) -> List[MarkerHit]:
    hits: List[MarkerHit] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        for match in MARKER_PATTERN.finditer(line):
            hits.append(
                MarkerHit(
                    file=file_path,
                    line=idx,
                    marker=match.group(0).upper(),
                    text=line.strip(),
                )
            )
    return hits


def build_header_index(header_paths: Iterable[Path], root: Path) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    word_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
    for path in header_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        rel = path.relative_to(root)
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if "(" not in stripped:
                continue
            for match in word_pattern.finditer(stripped):
                name = match.group(1)
                index[name].append(
                    {"file": str(rel), "line": str(lineno), "text": stripped.strip()}
                )
    return index


def detect_conditionals(file_path: Path, lines: Sequence[str]) -> List[ConditionalHit]:
    results: List[ConditionalHit] = []
    i = 0
    total = len(lines)
    while i < total:
        raw = lines[i]
        stripped = raw.lstrip()
        if stripped.startswith(("#if", "#ifdef", "#ifndef", "#elif")):
            directive_lines = [raw.rstrip("\n")]
            j = i
            while directive_lines[-1].rstrip().endswith("\\") and j + 1 < total:
                j += 1
                directive_lines.append(lines[j].rstrip("\n"))
            directive_text = "\n".join(directive_lines)
            uppercase = directive_text.upper()
            detected: Set[str] = set()
            for keyword in CONDITIONAL_PRIMARY_KEYWORDS:
                if keyword in uppercase:
                    detected.add(keyword)
            if any(key in uppercase for key in CONDITIONAL_GPU_KEYWORDS):
                detected.add("GPU")
            if any(key in uppercase for key in CONDITIONAL_CPU_KEYWORDS):
                detected.add("CPU")
            if detected:
                context_lines = directive_lines.copy()
                k = j + 1
                added = 0
                while k < total and added < 3:
                    next_line = lines[k].rstrip("\n")
                    if next_line.lstrip().startswith("#") and not next_line.lstrip().startswith("#else"):
                        break
                    context_lines.append(next_line)
                    added += 1
                    k += 1
                results.append(
                    ConditionalHit(
                        file=file_path,
                        line=i + 1,
                        directive=directive_text.strip(),
                        keywords=tuple(sorted(detected)),
                        context="\n".join(context_lines).strip(),
                    )
                )
            i = j
        i += 1
    return results


def gather_files(root: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    source_files = []
    script_files = []
    header_files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SOURCE_EXTENSIONS:
            source_files.append(path)
            if suffix in {".h", ".hpp", ".hh", ".hxx", ".cuh", ".inl", ".ipp"}:
                header_files.append(path)
        elif suffix in SCRIPT_EXTENSIONS:
            script_files.append(path)
    return source_files, script_files, header_files


def display_path(path: Path, root: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)
    return str(path)


def format_marker_section(markers: List[MarkerHit], root: Path) -> str:
    if not markers:
        return "No markers found."
    grouped: Dict[str, List[MarkerHit]] = defaultdict(list)
    for hit in markers:
        grouped[hit.marker].append(hit)
    lines: List[str] = []
    lines.append(f"Total marker hits: {len(markers)}\n")
    for marker in sorted(grouped.keys()):
        entries = sorted(grouped[marker], key=lambda x: (str(x.file), x.line))
        lines.append(f"### {marker} ({len(entries)})")
        for item in entries:
            rel = display_path(item.file, root)
            snippet = item.text
            lines.append(f"- `{rel}:{item.line}` — {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def format_stub_section(stubs: List[StubHit], root: Path) -> str:
    if not stubs:
        return "No potential stub implementations detected."
    grouped: Dict[str, List[StubHit]] = defaultdict(list)
    for stub in stubs:
        grouped[stub.stub_type].append(stub)
    lines: List[str] = []
    lines.append(f"Total potential stubs: {len(stubs)}\n")
    for category in sorted(grouped.keys()):
        entries = sorted(grouped[category], key=lambda x: (str(x.file), x.line))
        lines.append(f"### {category} ({len(entries)})")
        for item in entries:
            rel = display_path(item.file, root)
            header_refs = ", ".join(
                f"{ref['file']}:{ref['line']}" for ref in item.header_refs[:5]
            )
            if len(item.header_refs) > 5:
                header_refs += f" … (+{len(item.header_refs) - 5})"
            header_text = f"; header refs: {header_refs}" if header_refs else ""
            body = textwrap.indent(item.body_excerpt, "    ")
            signature = textwrap.indent(item.signature, "    ")
            lines.append(f"- `{rel}:{item.line}` — `{item.name}`{header_text}\n")
            if item.signature:
                lines.append("  Signature:")
                lines.append("  \n" + signature)
            if item.body_excerpt:
                lines.append("  Body excerpt:")
                lines.append("  \n" + body)
            lines.append("")
    return "\n".join(lines).strip()


def format_conditional_section(conditionals: List[ConditionalHit], root: Path) -> str:
    if not conditionals:
        return "No conditional compilation branches referencing mocks or fallbacks found."
    grouped: Dict[Tuple[str, ...], List[ConditionalHit]] = defaultdict(list)
    for hit in conditionals:
        grouped[hit.keywords].append(hit)
    lines: List[str] = []
    lines.append(f"Total conditional branches: {len(conditionals)}\n")
    for keyword_group in sorted(grouped.keys()):
        label = ", ".join(keyword_group)
        entries = sorted(grouped[keyword_group], key=lambda x: (str(x.file), x.line))
        lines.append(f"### {label} ({len(entries)})")
        for item in entries:
            rel = display_path(item.file, root)
            block = textwrap.indent(item.context, "    ")
            lines.append(f"- `{rel}:{item.line}`\n  \n{block}\n")
    return "\n".join(lines).strip()


def generate_report(root: Path, markers: List[MarkerHit], stubs: List[StubHit], conditionals: List[ConditionalHit]) -> str:
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    header = textwrap.dedent(
        f"""
        # Stage 2 Static Audit – Placeholders, Stubs, and Mock Implementations

        _Generated {timestamp} via `stage2_static_audit.py`._

        This report aggregates repository-wide scans for placeholder markers,
        potential stub implementations, and conditional branches that may gate
        mock or fallback code paths. Findings are grouped by category for
        downstream review.
        """
    ).strip()

    sections = [header]
    sections.append("\n## 1. Marker Sweep\n\n" + format_marker_section(markers, root))
    sections.append("\n## 2. Potential Stub Implementations\n\n" + format_stub_section(stubs, root))
    sections.append("\n## 3. Conditional Compilation / Mock Paths\n\n" + format_conditional_section(conditionals, root))
    return "\n\n".join(sections).strip() + "\n"


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Stage 2 static audit helper")
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Repository root (defaults to project root determined from script path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "dev-support" / "audit" / "stage2_static_report.md",
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.output.resolve()

    source_files, script_files, header_files = gather_files(root)
    header_index = build_header_index(header_files, root)

    markers: List[MarkerHit] = []
    stubs: List[StubHit] = []
    conditionals: List[ConditionalHit] = []

    relevant_files = set(source_files) | set(script_files)
    this_script = Path(__file__).resolve()

    for path in sorted(relevant_files):
        if path.resolve() == this_script:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        rel_path = path.relative_to(root)
        markers.extend(find_markers(rel_path, text))

        if path.suffix.lower() in SOURCE_EXTENSIONS:
            lines = text.splitlines()
            sanitized = sanitize_preserve_positions(text)
            tokens = tokenize(sanitized)
            line_offsets = build_line_offsets(text)
            stubs.extend(
                detect_stub_functions(rel_path, text, sanitized, tokens, line_offsets, header_index)
            )
            conditionals.extend(detect_conditionals(rel_path, lines))
        elif path.suffix.lower() in SCRIPT_EXTENSIONS:
            lines = text.splitlines()
            conditionals.extend(detect_conditionals(rel_path, lines))

    report = generate_report(root, markers, stubs, conditionals)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Audit complete. Report written to {output_path}")


if __name__ == "__main__":
    main()
