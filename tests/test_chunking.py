import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.core.config import settings
from app.services.chunking import chunk_text
from app.services.local_embedding_service import create_local_embeddings


class ChunkTextTests(unittest.TestCase):
    def test_returns_overlapping_chunks(self) -> None:
        chunks = chunk_text("abcdefghijkl", chunk_size=5, overlap=2)

        self.assertEqual(chunks, ["abcde", "defgh", "ghijk", "jkl"])

    def test_normalizes_whitespace_and_skips_empty_text(self) -> None:
        self.assertEqual(chunk_text(" first \n second ", 20, 2), ["first second"])
        self.assertEqual(chunk_text(" \n\t ", 20, 2), [])

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("some text", chunk_size=5, overlap=5)

    def test_local_embeddings_can_be_disabled_cleanly(self) -> None:
        with patch.object(settings, "local_embeddings_enabled", False):
            with self.assertRaises(HTTPException) as raised:
                create_local_embeddings(["a lease"])
        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
