#!/usr/bin/env python3

from __future__ import annotations

import sys
import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable
import enum

class _SortingMode(enum.StrEnum):
    """Which way to sort the ALGORITHMS.md file."""
    BY_ALGO_ID = 'by_algo_id'
    """Sort by Algorithm ID, then by Language."""

    BY_LANGUAGE = 'by_language'
    """Sort by Language, then by Algorithm ID."""


@dataclass(frozen=True)
class _AlgorithmEntry:
    """Entry in the ALGORITHMS.md file."""
    language: str
    algorithm: str
    algo_id: int
    library: str
    expr: bool
    doc_link: str | None


class _AlgorithmMdFormat:
    """Namespace for formatting ALGORITHMS.md entries."""

    LANG_COL_WIDTH = 13
    ALGO_COL_WIDTH = 30
    ID_COL_WIDTH = 8
    LIB_COL_WIDTH = 25
    EXPR_COL_WIDTH = 7
    DOC_COL_WIDTH = 144

    COLUMN_WIDTHS = [
        LANG_COL_WIDTH,
        ALGO_COL_WIDTH,
        ID_COL_WIDTH,
        LIB_COL_WIDTH,
        EXPR_COL_WIDTH,
        DOC_COL_WIDTH,
    ]

    @classmethod
    def _join_cols(cls, cols: Iterable[str]) -> str:
        """Join columns into a markdown table row."""
        return "|" + "|".join(cols) + "|"

    @classmethod
    def _format_entry(cls, entry: _AlgorithmEntry) -> str:
        """Format an _AlgorithmEntry as a markdown table row."""
        cols: list[str] = [
            entry.language,
            f"`{entry.algorithm}`",
            str(entry.algo_id),
            f"`{entry.library}`",
            "Y" if entry.expr else "",
            f"[doc]({entry.doc_link})" if entry.doc_link else "",
        ]

        return cls._join_cols(c.center(w) for c, w in zip(cols, cls.COLUMN_WIDTHS))

    @classmethod
    def _algorithms_md_header(cls) -> str:
        titles = ["Language", "Algorithm Name", "AlgoId", "Lib", "Expr", "Doc Link"]

        def mk_separator(width: int) -> str:
            return f" :{'-' * (width - 4)}: "

        titlecols = [ f"{t:^{w}}" for t, w in zip(titles, cls.COLUMN_WIDTHS)]
        sepcols = [mk_separator(w) for w in cls.COLUMN_WIDTHS]

        return f"{cls._join_cols(titlecols)}\n{cls._join_cols(sepcols)}\n"

    @classmethod
    def format_entries(cls, entries: Iterable[_AlgorithmEntry]) -> str:
        header = cls._algorithms_md_header()
        rows = (cls._format_entry(entry) for entry in entries)
        return header + "\n".join(rows) + "\n"


def _naive_parse_line(line: str) -> _AlgorithmEntry:
    """Parse a line from the ALGORITHMS.md file into an _AlgorithmEntry.

    I'm not smart enough to do this with a regex.
    """
    # Each column is delimited by '|', however, splitting blindly on '|' will cause
    # issues if any of the columns contain '|' characters. If they do, they're only
    # legal when escaped as '\|'.

    # First, let's get rid of the first '|' character, since we know that every line
    # starts and ends with '|'. We don't trim the last '|' because we want to use it
    # as a termination marker.
    assert line.startswith("|") and line.endswith("|\n")
    trimmed = line[1:-1]

    # Next, let's yoink the language, since we know that can't contain '|' characters.
    lang_end = trimmed.index("|")
    if trimmed[lang_end - 1] == "\\":
        msg = (
            "It seems you have tried to escape a pipe in the Language column. This is "
            "not currently supported. If you need support, implement it."
        )
        raise ValueError(msg)

    language: str = trimmed[:lang_end].strip()

    # Alright, let's grab the algorithm name. This one can contain '|' characters
    # within the name, but only within the backticks.
    algo_start = trimmed.index("`", lang_end) + 1
    algo_end = trimmed.index("`", algo_start)
    algorithm_name: str = trimmed[algo_start:algo_end].strip()

    # Now we can jump to the next column.
    next_col_start = trimmed.index("|", algo_end) + 1
    next_col_end = trimmed.index("|", next_col_start)
    algo_id_str: str = trimmed[next_col_start:next_col_end].strip()
    algo_id: int = int(algo_id_str)

    # Next column is the library name, which can also contain '|' characters within
    # backticks.
    lib_start = trimmed.index("`", next_col_end) + 1
    lib_end = trimmed.index("`", lib_start)
    library: str = trimmed[lib_start:lib_end].strip()

    # Next column is the Expr column, which is either 'Y' or empty.
    expr_col_start = trimmed.index("|", lib_end) + 1
    expr_col_end = trimmed.index("|", expr_col_start)
    expr_str: str = trimmed[expr_col_start:expr_col_end].strip()
    expr: bool = (expr_str == "Y")

    # Finally, the Doc Link column, which is a markdown link of the form [doc](url).
    # Notice that this may not actually be given and if it is not, there will be a
    # failure when seeking for the next '|'.
    def get_doc_str() -> str | None:
        doc_col_start = trimmed.index("|", expr_col_end) + 1
        try:
            doc_col_end = trimmed.index("|", doc_col_start)
        except ValueError:
            # This indicates that no more '|' characters were found, which means that
            # this is the last column and no doc link was given.
            return None

        doc_link_col: str = trimmed[doc_col_start:doc_col_end].strip()
        if doc_link_col == "":
            return None

        if not (doc_link_col.startswith("[doc](") and doc_link_col.endswith(")")):
            msg = f"Malformed doc link: {doc_link_col}. Must be [doc](url)."
            raise ValueError(msg)
        return doc_link_col[6:-1].strip()

    doc_link = get_doc_str()

    return _AlgorithmEntry(
        language=language,
        algorithm=algorithm_name,
        algo_id=algo_id,
        library=library,
        expr=expr,
        doc_link=doc_link,
    )


def _parse_line(line: str, line_idx: int) -> _AlgorithmEntry:
    """Parse a line from the ALGORITHMS.md file into an _AlgorithmEntry."""
    try:
        return _naive_parse_line(line)
    except ValueError as ve:
        msg = f"Error parsing line {line_idx + 1}: {ve}"
        raise ValueError(msg) from ve



def _inplace_sort(filename: pathlib.Path, sorting_mode: _SortingMode) -> None:
    """Sort the ALGORITHMS.md file in place."""
    header_line_count = 2

    # First job -- read the file.
    with filename.open("r", encoding="utf-8") as file:
        algorithm_lines = file.readlines()[header_line_count:]

    algorithms = (
        _parse_line(line, i + header_line_count)
        for i, line in enumerate(algorithm_lines)
    )
    sorting_key = (
        (lambda v: (v.algo_id, v.language)) if sorting_mode == _SortingMode.BY_ALGO_ID
        else (lambda v: (v.language, v.algo_id))
    )

    sorted_algorithms = sorted(algorithms, key=sorting_key)

    # Finally, output the sorted entries.
    formatted_str = _AlgorithmMdFormat.format_entries(sorted_algorithms)
    with filename.open("w", encoding="utf-8") as file:
        file.write(formatted_str)


def _sort_to(src: pathlib.Path, dest: pathlib.Path, sorting: _SortingMode) -> None:
    """Sort the ALGORITHMS.md file and write to a new file."""
    # Lazy implementation -- copy to dest, then sort in place.
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    _inplace_sort(dest, sorting)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Sort and format ALGORITHMS.md file.",
    )

    arg_parser.add_argument(
        "filename",
        type=pathlib.Path,
        help="Path to the ALGORITHMS.md file to be sorted.",
    )

    arg_parser.add_argument(
        "-i,--in-place",
        action="store_true",
        help="Sort the file in place (overwrite the original file).",
        dest="in_place",
        default=False,
    )

    arg_parser.add_argument(
        "-a,--by-algo-id",
        action="store_true",
        help="Sort the file by Algorithm ID (default is by Language).",
        dest="by_algo_id",
        default=True,
    )

    arg_parser.add_argument(
        "-l,--by-language",
        action="store_true",
        help="Sort the file by Language (default is by Algorithm ID).",
        dest="by_language",
        default=False,
    )

    arg_parser.add_argument(
        "-o,--output",
        type=pathlib.Path,
        help="Output file path (if not sorting in place).",
        dest="output",
        default=None,
    )

    args = arg_parser.parse_args()

    if args.in_place and args.output:
        print("Error: Cannot use --in-place and --output together. See --help.")
        sys.exit(1)

    if args.output is None and not args.in_place:
        print("Error: Must specify either --in-place or --output. See --help.")
        sys.exit(1)

    sorting_mode = (
        _SortingMode.BY_LANGUAGE if args.by_language else _SortingMode.BY_ALGO_ID
    )

    try:
        if args.in_place:
            _inplace_sort(args.filename, sorting_mode)
        else:
            output_path = args.output
            _sort_to(args.filename, output_path, sorting_mode)
    except FileNotFoundError as fnfe:
        msg = "Error: File not found - " + str(fnfe)
        print(msg)
        sys.exit(1)
