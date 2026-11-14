from datetime import date
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from starlette.datastructures import UploadFile
import requests

from app.api.router import package_info, upload_document
from app.main import app
from app.core.config import settings
from app.core.package_client import fetch_package, load_package_state
from app.core.packages import (
    get_package,
    require_feature,
    validate_file_for_package,
)
from app.core.upload_errors import describe_request_error
from app.schemas import DocumentCreate
from app.services.ingestion_service import ingest_document
from app.services.file_storage import get_stored_file
from app.services import llm_service
from app.services.text_extraction import extract_text


class PackageTests(unittest.TestCase):
    def test_package_feature_hierarchy(self) -> None:
        basic = get_package("basic")
        standard = get_package("standard")
        premium = get_package("premium")
        self.assertIn("basic_qa", basic.features)
        self.assertNotIn("document_library", basic.features)
        self.assertIn("document_library", standard.features)
        self.assertNotIn("premium_export", standard.features)
        self.assertTrue(premium.features.issuperset(standard.features))

    def test_basic_features_remain_in_every_tier(self) -> None:
        for name in ("basic", "standard", "premium"):
            package = get_package(name)
            self.assertIn("document_upload", package.features)
            self.assertIn("basic_qa", package.features)
            self.assertIn("basic_retrieval", package.features)

    def test_premium_formats_and_premium_actions_are_explicit(self) -> None:
        premium = get_package("premium")
        expected = {".pdf", ".docx", ".doc", ".txt", ".rtf", ".csv", ".xlsx", ".xls", ".dwg", ".rvt", ".ifc", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".pptx", ".zip"}
        self.assertEqual(premium.allowed_extensions, expected)
        self.assertIn("premium_export", premium.features)
        self.assertNotIn("premium_export", get_package("basic").features)
        self.assertNotIn("premium_export", get_package("standard").features)

    def test_upload_validation_reports_package_extension_mime_and_size(self) -> None:
        basic = get_package("basic")
        self.assertIn("Basic package does not include XLSX", validate_file_for_package(basic, "report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 100))
        self.assertIn("MIME type", validate_file_for_package(basic, "lease.pdf", "text/plain", 100))
        self.assertIn("empty", validate_file_for_package(basic, "lease.pdf", "application/pdf", 0))
        self.assertIn("exceeds", validate_file_for_package(basic, "lease.pdf", "application/pdf", 11 * 1024 * 1024))
        self.assertEqual(validate_file_for_package(basic, "lease.pdf", "application/pdf", 1024), "")

    def test_feature_gating_rejects_lower_tiers(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            require_feature(get_package("basic"), "document_library")
        self.assertEqual(raised.exception.status_code, 403)

    def test_exact_basic_standard_and_premium_extension_sets(self) -> None:
        self.assertEqual(get_package("basic").allowed_extensions, {".pdf", ".docx", ".txt", ".md"})
        self.assertEqual(
            get_package("standard").allowed_extensions,
            {".pdf", ".docx", ".txt", ".md", ".rtf", ".csv", ".xlsx"},
        )

    def test_empty_unknown_extension_package_mime_and_size_errors_are_distinct(self) -> None:
        basic = get_package("basic")
        self.assertIn("empty", validate_file_for_package(basic, "lease.pdf", "application/pdf", 0))
        self.assertIn("Unsupported file format", validate_file_for_package(basic, "lease.exe", "application/octet-stream", 100))
        self.assertIn("package does not include", validate_file_for_package(basic, "sheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 100))
        self.assertIn("MIME type", validate_file_for_package(basic, "lease.pdf", "text/plain", 100))
        self.assertEqual(validate_file_for_package(basic, "LEASE.PDF", "application/pdf", 1024), "")
        self.assertIn("exceeds", validate_file_for_package(basic, "lease.pdf", "application/pdf", 11 * 1024 * 1024, 10))

    def test_allowed_formats_match_package_and_actual_backend_rules(self) -> None:
        supported = {
            "basic": ("lease.txt", "text/plain"),
            "standard": ("rent.csv", "text/csv"),
            "premium": (
                "presentation.pptx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
        }
        for package_name, (filename, mime) in supported.items():
            with self.subTest(package=package_name):
                package = get_package(package_name)
                self.assertEqual(
                    validate_file_for_package(package, filename, mime, 100),
                    "",
                )
                with patch.object(settings, "package", package_name):
                    package_response = package_info()
                self.assertEqual(
                    package_response["allowed_extensions"],
                    sorted(package.allowed_extensions),
                )

    def test_package_restriction_names_extension_and_next_eligible_package(self) -> None:
        blocked = validate_file_for_package(
            get_package("standard"),
            "presentation.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            100,
        )
        self.assertEqual(
            blocked,
            "Your Standard package does not include PPTX uploads. Choose Premium to upload this file.",
        )
        self.assertIn(
            "Choose Standard",
            validate_file_for_package(
                get_package("basic"),
                "rent.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                100,
            ),
        )

    def test_damaged_supported_docx_has_actionable_extraction_error(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            extract_text("lease.docx", b"not a docx archive")
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("damaged", raised.exception.detail)

    def test_pdf_extraction_removes_database_unsupported_null_characters(self) -> None:
        page = Mock()
        page.extract_text.return_value = "Lease\x00 renewal terms"
        reader = Mock(is_encrypted=False, pages=[page])
        with patch("app.services.text_extraction.PdfReader", return_value=reader):
            self.assertEqual(
                extract_text("lease.pdf", b"%PDF-test"),
                "Lease renewal terms",
            )

    def test_pdf_with_only_null_characters_has_readable_extraction_error(self) -> None:
        page = Mock()
        page.extract_text.return_value = "\x00\x00"
        reader = Mock(is_encrypted=False, pages=[page])
        with patch("app.services.text_extraction.PdfReader", return_value=reader):
            with self.assertRaises(HTTPException) as raised:
                extract_text("lease.pdf", b"%PDF-test")
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("No extractable text", raised.exception.detail)


class FileSupportTests(unittest.TestCase):
    def test_csv_and_zip_text_can_be_extracted(self) -> None:
        csv_text = extract_text("unit-costs.csv", b"unit,rent\nA1,1200\n")
        self.assertIn("unit", csv_text)

        from io import BytesIO
        from zipfile import ZipFile
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("lease.pdf", b"not needed")
            archive.writestr("notes.txt", b"inspection completed")
        zip_text = extract_text("property-files.zip", buffer.getvalue())
        self.assertIn("lease.pdf", zip_text)
        self.assertIn("notes.txt", zip_text)

    def test_txt_markdown_and_docx_text_can_be_extracted(self) -> None:
        self.assertIn("Roof inspection", extract_text("maintenance.TXT", b"Roof inspection completed"))
        self.assertIn("Lease terms", extract_text("lease.md", b"# Lease terms\nRent is due monthly"))

        from io import BytesIO
        from docx import Document
        buffer = BytesIO()
        document = Document()
        document.add_paragraph("Lease renewal for Pine Court")
        document.save(buffer)
        self.assertIn("Lease renewal", extract_text("lease.docx", buffer.getvalue()))

    def test_pptx_text_can_be_extracted(self) -> None:
        from pptx import Presentation

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.shapes.add_textbox(0, 0, 5000000, 500000).text = "Property inspection"
        buffer = BytesIO()
        presentation.save(buffer)
        self.assertIn("Property inspection", extract_text("inspection.pptx", buffer.getvalue()))

    def test_empty_document_extraction_is_readable(self) -> None:
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as raised:
            extract_text("notes.txt", b"   ")
        self.assertIn("No extractable text", raised.exception.detail)


class CsvExtractionTests(unittest.TestCase):
    def test_small_csv_keeps_headers_and_row_context_without_empty_cells(self) -> None:
        extracted = extract_text(
            "units.csv",
            b"unit,rent,notes\nA1,1200,\nA2,,Roof repair\n",
        )
        self.assertIn("CSV columns: unit | rent | notes", extracted)
        self.assertIn("Row 2: unit: A1 | rent: 1200", extracted)
        self.assertIn("Row 3: unit: A2 | notes: Roof repair", extracted)
        self.assertNotIn("Row 3: unit: A2 | rent:", extracted)

    def test_moderate_csv_extracts_data_rows(self) -> None:
        rows = ["unit,rent,status"]
        rows.extend(f"A{index},{1000 + index},Active" for index in range(300))
        extracted = extract_text("units.csv", "\n".join(rows).encode())
        self.assertIn("Row 301: unit: A299 | rent: 1299 | status: Active", extracted)

    def test_empty_csv_returns_specific_no_text_error(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            extract_text("empty.csv", b"")
        self.assertIn("No extractable text", raised.exception.detail)

    def test_malformed_csv_returns_actionable_error(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            extract_text("malformed.csv", b"name,amount\n\"unclosed,10\n")
        self.assertIn("malformed or has invalid quoting", raised.exception.detail)

    def test_wide_csv_is_rejected_with_column_limit(self) -> None:
        headers = [f"field{index}" for index in range(101)]
        content = (",".join(headers) + "\n" + ",".join(["value"] * 101)).encode()
        with self.assertRaises(HTTPException) as raised:
            extract_text("wide.csv", content)
        self.assertIn("more than 100 columns", raised.exception.detail)

    def test_oversized_cell_is_rejected_with_cell_limit(self) -> None:
        content = ("notes\n" + "x" * 2049).encode()
        with self.assertRaises(HTTPException) as raised:
            extract_text("long-cell.csv", content)
        self.assertIn("cell longer than 2,048 characters", raised.exception.detail)

    def test_extracted_text_limit_rejects_large_csv_instead_of_truncating(self) -> None:
        row = ",".join(["x" * 100] * 30).encode() + b"\n"
        content = b"c" + b"0," * 29 + b"c29\n" + row * 1200
        self.assertGreater(len(content), 3 * 1024 * 1024)
        with self.assertRaises(HTTPException) as raised:
            extract_text("large.csv", content)
        self.assertIn("50,000-character extraction limit", raised.exception.detail)
        self.assertIn("No partial document was indexed", raised.exception.detail)

    def test_row_limit_is_reported_for_many_empty_records(self) -> None:
        content = b"id\n" + b"\n" * 10_001
        with self.assertRaises(HTTPException) as raised:
            extract_text("many-rows.csv", content)
        self.assertIn("10,000-row extraction limit", raised.exception.detail)


class UploadRequestErrorTests(unittest.TestCase):
    def test_connect_and_processing_timeouts_are_distinguished(self) -> None:
        connect_message = describe_request_error(
            requests.ConnectTimeout("connect timed out"),
            "http://api:8000/api",
        )
        processing_message = describe_request_error(
            requests.ReadTimeout("read timed out"),
            "http://api:8000/api",
        )
        wrapped_timeout_message = describe_request_error(
            requests.ConnectionError("Connection aborted.", TimeoutError("timed out")),
            "http://api:8000/api",
        )
        self.assertIn("Could not connect", connect_message)
        self.assertIn("connection was established", processing_message)
        self.assertIn("Check the document library before retrying", processing_message)
        self.assertIn("waiting for upload processing", wrapped_timeout_message)

    def test_api_status_and_detail_are_preserved_for_the_ui(self) -> None:
        response = requests.Response()
        response.status_code = 422
        response.url = "http://api:8000/api/documents"
        response._content = b'{"detail":"No extractable text was found in the PDF."}'
        message = describe_request_error(
            requests.HTTPError("422 Client Error", response=response),
            "http://api:8000/api",
        )
        self.assertEqual(
            message,
            "API error (422): No extractable text was found in the PDF.",
        )


class UploadRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_api_openapi_schema_builds_with_date_routes(self) -> None:
        paths = app.openapi()["paths"]
        self.assertIn("/health", paths)
        self.assertIn("/api/documents", paths)
        self.assertIn("/api/documents/search-options", paths)
        query_schema = app.openapi()["components"]["schemas"]["QueryRequest"]
        self.assertIn("document_id", query_schema["properties"])

    async def test_optional_form_metadata_defaults_without_property_or_date(self) -> None:
        upload = UploadFile(
            filename="lease.txt",
            file=BytesIO(b"Lease document text"),
        )
        with patch("app.api.router.ingest_document", new_callable=AsyncMock) as ingest:
            await upload_document(
                file=upload,
                property_name="",
                unit="",
                document_type="",
                category="General",
                document_name=None,
                note="Renewal copy",
                issue_type="",
                portfolio_name="",
                property_code="",
                tags="",
                workflow_status="Active",
                date=None,
                db=object(),
            )
        metadata = ingest.await_args.args[1]
        self.assertEqual(metadata.document_type, "Other")
        self.assertEqual(metadata.note, "Renewal copy")
        self.assertEqual(metadata.date, date.today())


class PackageClientTests(unittest.TestCase):
    def test_unavailable_package_raises_instead_of_falling_back(self) -> None:
        with patch(
            "app.core.package_client.requests.get",
            side_effect=requests.ConnectionError("API is down"),
        ):
            with self.assertRaises(requests.ConnectionError):
                fetch_package("http://api:8000/api")

    def test_package_is_refetched_and_recovers_after_api_returns(self) -> None:
        response = Mock()
        response.json.return_value = {
            "name": "premium",
            "features": ["document_library"],
            "allowed_extensions": [".csv", ".pdf"],
        }
        with patch(
            "app.core.package_client.requests.get",
            side_effect=[requests.ConnectionError("API is down"), response],
        ) as get:
            with self.assertRaises(requests.ConnectionError):
                fetch_package("http://api:8000/api")
            package = fetch_package("http://api:8000/api")

        self.assertEqual(package["name"], "premium")
        self.assertEqual(get.call_count, 2)

    def test_package_unavailable_is_not_mislabeled_and_recovery_uses_backend_tier(self) -> None:
        response = Mock()
        response.json.return_value = {
            "name": "standard",
            "features": ["document_library"],
            "allowed_extensions": [".csv", ".pdf"],
        }
        with patch(
            "app.core.package_client.requests.get",
            side_effect=[requests.ConnectionError("API is down"), response],
        ):
            unavailable, unavailable_error = load_package_state("http://api:8000/api")
            recovered, recovered_error = load_package_state("http://api:8000/api")

        self.assertIsNone(unavailable)
        self.assertIsInstance(unavailable_error, requests.ConnectionError)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["name"], "standard")
        self.assertIsNone(recovered_error)

    def test_api_status_detail_is_retained_when_package_cannot_be_loaded(self) -> None:
        response = requests.Response()
        response.status_code = 503
        response.url = "http://api:8000/api/package"
        response._content = b'{"detail":"Database unavailable"}'
        with patch(
            "app.core.package_client.requests.get",
            return_value=response,
        ):
            package, error = load_package_state("http://api:8000/api")

        self.assertIsNone(package)
        self.assertIsInstance(error, requests.HTTPError)
        self.assertIn("503", describe_request_error(error, "http://api:8000/api"))
        self.assertIn("Database unavailable", describe_request_error(error, "http://api:8000/api"))

    def test_incomplete_package_response_is_not_trusted(self) -> None:
        response = Mock()
        response.json.return_value = {"name": "basic"}
        with patch("app.core.package_client.requests.get", return_value=response):
            with self.assertRaisesRegex(ValueError, "incomplete package configuration"):
                fetch_package("http://api:8000/api")


class IngestionPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_package_representative_files_are_extracted_embedded_stored_and_committed(self) -> None:
        class FakeSession:
            document = None

            def scalar(self, _statement):
                return 0

            def add(self, document):
                self.document = document

            def flush(self):
                pass

            def commit(self):
                pass

            def refresh(self, _document):
                pass

            def rollback(self):
                pass

        from pptx import Presentation

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.shapes.add_textbox(0, 0, 5000000, 500000).text = "Property inspection"
        pptx_buffer = BytesIO()
        presentation.save(pptx_buffer)
        examples = (
            (
                "basic",
                "lease.txt",
                b"Lease renewal\x00 for Pine Court",
                "text/plain",
                "Lease renewal",
            ),
            ("standard", "rent.csv", b"unit,rent\nA1,1200\n", "text/csv", "unit"),
            (
                "premium",
                "inspection.pptx",
                pptx_buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "Property inspection",
            ),
        )
        for package_name, filename, content, mime, extracted_text in examples:
            with self.subTest(package=package_name):
                with TemporaryDirectory() as upload_dir:
                    with (
                        patch.object(settings, "package", package_name),
                        patch.object(settings, "upload_dir", Path(upload_dir)),
                        patch(
                            "app.services.ingestion_service.create_embeddings",
                            return_value=[[0.0] * settings.embedding_dimensions],
                        ),
                    ):
                        upload = UploadFile(
                            filename=filename,
                            file=BytesIO(content),
                            headers={"content-type": mime},
                        )
                        metadata = DocumentCreate(
                            document_name=filename,
                            note="Fixture upload",
                            source_file=filename,
                            date=date.today(),
                        )
                        session = FakeSession()
                        document = await ingest_document(upload, metadata, session)

                        self.assertIs(document, session.document)
                        self.assertEqual(document.note, "Fixture upload")
                        self.assertIn(extracted_text, document.chunks[0].content)
                        self.assertNotIn("\x00", document.chunks[0].content)
                        self.assertEqual(
                            get_stored_file(document.storage_path).read_bytes(),
                            content,
                        )

    async def test_validation_failures_return_specific_api_errors(self) -> None:
        class FakeSession:
                    def scalar(self, _statement):
                        return 0

        examples = (
                    ("basic", "report.exe", b"data", "application/octet-stream", 415, "Unsupported file format"),
                    ("basic", "lease.pdf", b"", "application/pdf", 422, "file is empty"),
                    ("basic", "lease.pdf", b"data", "text/plain", 415, "MIME type"),
                    (
                        "standard",
                        "slides.pptx",
                        b"data",
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        403,
                        "Your Standard package does not include PPTX",
                    ),
        )
        for package_name, filename, content, mime, status, message in examples:
                    with self.subTest(filename=filename, package=package_name):
                        with patch.object(settings, "package", package_name):
                            upload = UploadFile(
                                filename=filename,
                                file=BytesIO(content),
                                headers={"content-type": mime},
                            )
                            with self.assertRaises(HTTPException) as raised:
                                await ingest_document(
                                    upload,
                                    DocumentCreate(source_file=filename, date=date.today()),
                                    FakeSession(),
                                )
                        self.assertEqual(raised.exception.status_code, status)
                        self.assertIn(message, raised.exception.detail)

    async def test_embedding_failure_does_not_save_file_or_create_record(self) -> None:
        class FakeSession:
                    added = False

                    def scalar(self, _statement):
                        return 0

                    def add(self, _document):
                        self.added = True

        with TemporaryDirectory() as upload_dir:
                    with (
                        patch.object(settings, "package", "basic"),
                        patch.object(settings, "upload_dir", Path(upload_dir)),
                        patch(
                            "app.services.ingestion_service.create_embeddings",
                            side_effect=HTTPException(status_code=503, detail="Model unavailable"),
                        ),
                    ):
                        upload = UploadFile(
                            filename="lease.txt",
                            file=BytesIO(b"Lease renewal for Pine Court"),
                            headers={"content-type": "text/plain"},
                        )
                        session = FakeSession()
                        with self.assertRaises(HTTPException) as raised:
                            await ingest_document(
                                upload,
                                DocumentCreate(source_file="lease.txt", date=date.today()),
                                session,
                            )

                    self.assertEqual(raised.exception.status_code, 503)
                    self.assertFalse(session.added)
                    self.assertEqual(list(Path(upload_dir).iterdir()), [])

    async def test_csv_limit_error_does_not_embed_or_store_partial_document(self) -> None:
        class FakeSession:
                    def scalar(self, _statement):
                        return 0

                    def add(self, _document):
                        raise AssertionError("A CSV beyond extraction limits must not create a document.")

        content = b"notes\n" + (b"x" * 100 + b"\n") * 1000
        with TemporaryDirectory() as upload_dir:
                    with (
                        patch.object(settings, "package", "standard"),
                        patch.object(settings, "upload_dir", Path(upload_dir)),
                        patch("app.services.ingestion_service.create_embeddings") as embed,
                    ):
                        upload = UploadFile(
                            filename="large.csv",
                            file=BytesIO(content),
                            headers={"content-type": "text/csv"},
                        )
                        with self.assertRaises(HTTPException) as raised:
                            await ingest_document(
                                upload,
                                DocumentCreate(source_file="large.csv", date=date.today()),
                                FakeSession(),
                            )

                    self.assertIn("extraction limit", raised.exception.detail)
                    embed.assert_not_called()
                    self.assertEqual(list(Path(upload_dir).iterdir()), [])

    async def test_database_failure_rolls_back_and_removes_original_file(self) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        class FakeSession:
            def scalar(self, _statement):
                return 0

            def add(self, _document):
                pass

            def flush(self):
                pass

            def refresh(self, _document):
                pass

            def commit(self):
                raise SQLAlchemyError("write failed")

            def rollback(self):
                self.rolled_back = True

        with TemporaryDirectory() as upload_dir:
            with (
                patch.object(settings, "package", "basic"),
                patch.object(settings, "upload_dir", Path(upload_dir)),
                patch(
                    "app.services.ingestion_service.create_embeddings",
                    return_value=[[0.0] * settings.embedding_dimensions],
                ),
            ):
                upload = UploadFile(
                    filename="lease.txt",
                    file=BytesIO(b"Lease renewal for Pine Court"),
                    headers={"content-type": "text/plain"},
                )
                metadata = DocumentCreate(
                    document_name="Lease",
                    source_file="lease.txt",
                    date=date.today(),
                )
                session = FakeSession()
                with self.assertRaises(HTTPException) as raised:
                    await ingest_document(upload, metadata, session)

            self.assertEqual(raised.exception.status_code, 500)
            self.assertTrue(session.rolled_back)
            self.assertEqual(list(Path(upload_dir).iterdir()), [])

    async def test_nul_persistence_error_is_actionable_and_cleans_up(self) -> None:
        class FakeSession:
            def scalar(self, _statement):
                return 0

            def add(self, _document):
                pass

            def flush(self):
                raise ValueError("A string literal cannot contain NUL (0x00) characters.")

            def rollback(self):
                pass

        with TemporaryDirectory() as upload_dir:
            with (
                patch.object(settings, "package", "basic"),
                patch.object(settings, "upload_dir", Path(upload_dir)),
                patch(
                    "app.services.ingestion_service.create_embeddings",
                    return_value=[[0.0] * settings.embedding_dimensions],
                ),
            ):
                upload = UploadFile(
                    filename="lease.txt",
                    file=BytesIO(b"Readable lease content"),
                    headers={"content-type": "text/plain"},
                )
                with self.assertRaises(HTTPException) as raised:
                    await ingest_document(
                        upload,
                        DocumentCreate(source_file="lease.txt", date=date.today()),
                        FakeSession(),
                    )

            self.assertEqual(raised.exception.status_code, 422)
            self.assertIn("cannot be stored", raised.exception.detail)
            self.assertEqual(list(Path(upload_dir).iterdir()), [])


class EmbeddingBackendTests(unittest.TestCase):
    def test_local_backend_dispatches_to_local_service(self) -> None:
        with patch.object(settings, "embedding_backend", "local"):
            with patch("app.services.local_embedding_service.create_local_embeddings", return_value=[[1.0]]) as local:
                result = llm_service.create_embeddings(["lease"])
        self.assertEqual(result, [[1.0]])
        local.assert_called_once_with(["lease"])

    def test_provider_backend_dispatches_cleanly(self) -> None:
        with patch.object(settings, "embedding_backend", "provider"):
            with patch.object(llm_service, "_create_provider_embeddings", return_value=[[2.0]]) as provider:
                result = llm_service.create_embeddings(["record"])
        self.assertEqual(result, [[2.0]])
        provider.assert_called_once_with(["record"])


if __name__ == "__main__":
    unittest.main()
