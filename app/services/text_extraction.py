"""Text extraction for the MVP's supported document formats."""

from __future__ import annotations

from io import BytesIO, TextIOWrapper
from threading import Lock
from zipfile import BadZipFile, ZipFile
import csv
import re

from docx import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from fastapi import HTTPException
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

CSV_MAX_ROWS = 10_000
CSV_MAX_COLUMNS = 100
CSV_MAX_CELL_CHARS = 2_048
CSV_MAX_EXTRACTED_CHARS = 50_000
_CSV_FIELD_LIMIT_LOCK = Lock()


def _extract_csv_text(content: bytes) -> str:
    """Read a bounded CSV while retaining per-row column context."""
    output: list[str] = []
    extracted_chars = 0
    try:
        with _CSV_FIELD_LIMIT_LOCK:
            previous_field_limit = csv.field_size_limit()
            csv.field_size_limit(CSV_MAX_CELL_CHARS + 1)
            try:
                with TextIOWrapper(
                    BytesIO(content),
                    encoding="utf-8-sig",
                    newline="",
                ) as stream:
                    reader = csv.reader(stream, strict=True)
                    headers = next(reader, None)
                    if headers is None:
                        return ""
                    if len(headers) > CSV_MAX_COLUMNS:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"The CSV has more than {CSV_MAX_COLUMNS} columns. "
                                "Reduce its width and upload it again."
                            ),
                        )

                    normalized_headers: list[str] = []
                    seen_headers: dict[str, int] = {}
                    for index, value in enumerate(headers, start=1):
                        header = value.strip() or f"Column {index}"
                        seen_headers[header] = seen_headers.get(header, 0) + 1
                        if seen_headers[header] > 1:
                            header = f"{header} [{seen_headers[header]}]"
                        normalized_headers.append(header)

                    header_line = "CSV columns: " + " | ".join(normalized_headers)
                    if len(header_line) > CSV_MAX_EXTRACTED_CHARS:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"The CSV headers exceed the "
                                f"{CSV_MAX_EXTRACTED_CHARS:,}-character extraction limit."
                            ),
                        )
                    output.append(header_line)
                    extracted_chars = len(header_line) + 1

                    data_rows = 0
                    for row_number, row in enumerate(reader, start=2):
                        data_rows += 1
                        if data_rows > CSV_MAX_ROWS:
                            raise HTTPException(
                                status_code=422,
                                detail=(
                                    f"The CSV exceeds the {CSV_MAX_ROWS:,}-row extraction "
                                    "limit. Split it into smaller files and upload again."
                                ),
                            )
                        if not row:
                            continue
                        if len(row) > CSV_MAX_COLUMNS:
                            raise HTTPException(
                                status_code=422,
                                detail=(
                                    f"CSV row {row_number} has more than "
                                    f"{CSV_MAX_COLUMNS} columns. Reduce its width and upload again."
                                ),
                            )
                        if len(row) != len(normalized_headers):
                            raise HTTPException(
                                status_code=422,
                                detail=(
                                    f"CSV row {row_number} has {len(row)} cells, but the "
                                    f"header has {len(normalized_headers)} columns. "
                                    "Check the CSV delimiters and quoting."
                                ),
                            )

                        values: list[str] = []
                        for header, raw_value in zip(
                            normalized_headers, row, strict=True
                        ):
                            value = raw_value.strip()
                            if len(value) > CSV_MAX_CELL_CHARS:
                                raise HTTPException(
                                    status_code=422,
                                    detail=(
                                        f"CSV row {row_number} contains a cell longer than "
                                        f"{CSV_MAX_CELL_CHARS:,} characters."
                                    ),
                                )
                            if value:
                                values.append(f"{header}: {value}")
                        if not values:
                            continue

                        row_text = f"Row {row_number}: " + " | ".join(values)
                        if extracted_chars + len(row_text) + 1 > CSV_MAX_EXTRACTED_CHARS:
                            raise HTTPException(
                                status_code=422,
                                detail=(
                                    f"The CSV exceeds the {CSV_MAX_EXTRACTED_CHARS:,}-character "
                                    "extraction limit. No partial document was indexed; split "
                                    "the file into smaller CSVs and upload again."
                                ),
                            )
                        output.append(row_text)
                        extracted_chars += len(row_text) + 1

                    if len(output) == 1:
                        return ""
            finally:
                csv.field_size_limit(previous_field_limit)
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="The CSV could not be decoded as UTF-8. Save it as UTF-8 and upload again.",
        ) from exc
    except csv.Error as exc:
        if "field larger than field limit" in str(exc):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"A CSV cell exceeds the {CSV_MAX_CELL_CHARS:,}-character "
                    "cell limit. Shorten the cell and upload again."
                ),
            ) from exc
        raise HTTPException(
            status_code=422,
            detail=f"The CSV is malformed or has invalid quoting: {exc}",
        ) from exc
    return "\n".join(output)


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from supported PDF, Office, CSV, RTF, Markdown, and text files."""
    suffix = filename.lower().rsplit(".", maxsplit=1)[-1] if "." in filename else ""

    if suffix == "pdf":
        try:
            if not content.lstrip().startswith(b"%PDF-"):
                raise HTTPException(status_code=422, detail="Unsupported file format: the selected PDF has an invalid or missing PDF signature.")
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise HTTPException(status_code=422, detail="The PDF is password-protected. Upload an unencrypted PDF.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail="The PDF could not be extracted. It may be damaged, scanned, or unsupported. Upload a text-readable PDF.") from exc
    elif suffix == "docx":
        try:
            document = WordDocument(BytesIO(content))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        except (BadZipFile, PackageNotFoundError) as exc:
            raise HTTPException(
                status_code=422,
                detail="The DOCX file is damaged or is not a valid DOCX document.",
            ) from exc
    elif suffix == "doc":
        try:
            strings = re.findall(rb"[\x20-\x7e]{4,}", content)
            text = "\n".join(value.decode("ascii", errors="ignore") for value in strings[:3000])
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="The legacy DOC file could not be read. Verify that it is not damaged.",
            ) from exc
    elif suffix == "xls":
        try:
            import xlrd
            workbook = xlrd.open_workbook(file_contents=content)
            rows: list[str] = []
            for sheet in workbook.sheets():
                rows.append(f"Worksheet: {sheet.name}")
                for row_index in range(sheet.nrows):
                    values = ["" if value == "" else str(value) for value in sheet.row_values(row_index)]
                    if any(values):
                        rows.append("\t".join(values))
            text = "\n".join(rows)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="The XLS file could not be read. Verify that it is not damaged.",
            ) from exc
    elif suffix == "pptx":
        try:
            from pptx import Presentation
            presentation = Presentation(BytesIO(content))
            slides: list[str] = []
            for index, slide in enumerate(presentation.slides, start=1):
                slide_text = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
                if slide_text:
                    slides.append(f"Slide {index}:\n" + "\n".join(slide_text))
            text = "\n\n".join(slides)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="The PPTX file could not be read. Verify that it is not damaged.",
            ) from exc
    elif suffix in {"jpg", "jpeg", "png", "tiff", "tif"}:
        try:
            from PIL import Image
            import pytesseract
            image = Image.open(BytesIO(content))
            text = pytesseract.image_to_string(image)
        except ImportError as exc:
            raise HTTPException(status_code=422, detail="Image OCR is unavailable. Install the OCR runtime and Tesseract.") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="The image could not be read or OCR failed. Verify the image and Tesseract installation.",
            ) from exc
    elif suffix in {"dwg", "rvt", "ifc"}:
        strings = re.findall(rb"[\x20-\x7e]{5,}", content)
        printable = "\n".join(value.decode("ascii", errors="ignore") for value in strings[:2000])
        text = f"{suffix.upper()} property file metadata and embedded text:\n{printable}" if printable else f"{suffix.upper()} property file. No searchable text layer is embedded in this file."
    elif suffix == "zip":
        try:
            archive = ZipFile(BytesIO(content))
            entries = archive.infolist()
            total_uncompressed = sum(entry.file_size for entry in entries)
            if total_uncompressed > 20 * 1024 * 1024:
                raise HTTPException(status_code=422, detail="The ZIP contents exceed the 20 MB extraction safety limit.")
            names = [entry.filename for entry in entries if not entry.is_dir()][:500]
            text = "ZIP archive contents:\n" + "\n".join(names)
        except (BadZipFile, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail="The ZIP archive could not be read.") from exc
    elif suffix == "xlsx":
        try:
            from openpyxl import load_workbook
            workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
            rows: list[str] = []
            for worksheet in workbook.worksheets:
                rows.append(f"Worksheet: {worksheet.title}")
                for row in worksheet.iter_rows(values_only=True):
                    values = ["" if value is None else str(value) for value in row]
                    if any(values):
                        rows.append("\t".join(values))
            text = "\n".join(rows)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="The XLSX file could not be read. Verify that it is not damaged.",
            ) from exc
    elif suffix == "csv":
        text = _extract_csv_text(content)
    elif suffix == "rtf":
        try:
            decoded = content.decode("utf-8-sig", errors="replace")
            text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", decoded)
            text = re.sub(r"\\[a-zA-Z]+-?[0-9]* ?", " ", text)
            text = text.replace("{", " ").replace("}", " ").replace("\\\\", "\\")
        except Exception as exc:
            raise HTTPException(status_code=422, detail="The RTF file could not be read.") from exc
    elif suffix in {"txt", "md"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=422, detail="Text files must use UTF-8 encoding."
            ) from exc
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Check the active package's allowed file types.",
        )

    text = text.replace("\x00", "")
    if not text.strip():
        if suffix == "pdf":
            raise HTTPException(
                status_code=422,
                detail="No extractable text was found in the PDF. Scanned or image-only PDFs are not supported; upload a text-readable PDF.",
            )
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found in the document.",
        )
    return text
