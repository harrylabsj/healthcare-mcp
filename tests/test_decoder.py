from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from healthcare.decoder import DecoderError, DecoderLimits, DecoderUnavailable, PdfTextDecoder, TextDecoder, TesseractOcrAdapter, VisionOcrAdapter, decode_file, sandbox_profile


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


@pytest.mark.skipif(
    not (shutil.which("pdftotext") and shutil.which("pdfinfo")),
    reason="Poppler is not installed",
)
def test_pdf_text_decoder_extracts_pages_with_limits(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(_minimal_pdf("Creatinine 88.4 umol/L"))
    decoded = PdfTextDecoder(sandbox=False).decode(path)
    assert decoded.media_type == "application/pdf"
    assert len(decoded.pages) == 1
    assert "Creatinine 88.4" in decoded.pages[0].text

    with pytest.raises(DecoderError, match="output size"):
        PdfTextDecoder(DecoderLimits(max_output_chars=4), sandbox=False).decode(path)


def test_decode_file_text_path_returns_page_metadata_without_raw_text(tmp_path: Path) -> None:
    path = tmp_path / "report.txt"
    path.write_text("检验报告\n血肌酐 88.4 umol/L\n", encoding="utf-8")
    document = decode_file(path)
    assert isinstance(TextDecoder().decode(path), type(document))
    assert document.pages[0].text.startswith("检验报告")


def test_pdf_decoder_rejects_symlinks_and_non_pdf(tmp_path: Path) -> None:
    target = tmp_path / "real.pdf"
    target.write_bytes(_minimal_pdf("safe"))
    link = tmp_path / "link.pdf"
    link.symlink_to(target)
    with pytest.raises(DecoderError, match="symbolic"):
        PdfTextDecoder(sandbox=False).decode(link)

    unsupported = tmp_path / "report.txt"
    unsupported.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(DecoderError, match="file type"):
        PdfTextDecoder(sandbox=False).decode(unsupported)


def test_tesseract_fails_closed_when_chinese_language_pack_is_missing(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"not an image")
    adapter = TesseractOcrAdapter(languages=("chi_sim",), sandbox=False)
    if not adapter.available():
        pytest.skip("Tesseract is not installed")
    with pytest.raises(DecoderUnavailable, match="language pack"):
        adapter.decode_image(image)


def test_vision_adapter_validates_helper_output(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"synthetic")
    helper = tmp_path / "fake-vision"
    helper.write_text(
        "#!/bin/sh\nprintf '%s\\n' '{\"engine\":\"apple-vision\",\"text\":\"血肌酐 88.4\",\"blocks\":[]}'\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    document = VisionOcrAdapter(helper, sandbox=False).decode_image(image)
    assert document.decoder == "apple-vision"
    assert document.pages[0].text == "血肌酐 88.4"


def test_sandbox_profile_is_read_only_and_denies_network(tmp_path: Path) -> None:
    input_path = tmp_path / "scan.png"
    profile = sandbox_profile(["/usr/bin/true"], input_path)
    assert "(deny network*)" in profile
    assert "(allow file-write*" not in profile
    # macOS system.sb default-deny blocks reading user reports even with
    # explicit file-read rules, so the profile uses allow default with writes
    # and network denied; the decoder stays read-only and offline.
    assert "(allow default)" in profile
    assert "system.sb" in profile
