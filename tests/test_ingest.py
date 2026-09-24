from __future__ import annotations

from pathlib import Path

import pytest

from healthcare.ingest import ImportBoundaryError, read_control_selected_text


def test_control_import_accepts_utf8_regular_text(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text("检验报告\n血肌酐 88.4 umol/L\n", encoding="utf-8")
    selected = read_control_selected_text(report)
    assert selected.filename == "report.txt"
    assert selected.text.startswith("检验报告")
    assert selected.size_bytes > 0
    assert len(selected.sha256) == 64


def test_control_import_rejects_unsupported_or_symlinked_paths(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"not a decoder input")
    with pytest.raises(ImportBoundaryError, match="UTF-8 text"):
        read_control_selected_text(pdf)

    target = tmp_path / "target.txt"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(ImportBoundaryError, match="symbolic links"):
        read_control_selected_text(link)


def test_control_import_rejects_oversized_files(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text("12345", encoding="utf-8")
    with pytest.raises(ImportBoundaryError, match="exceeds"):
        read_control_selected_text(report, max_bytes=4)
