# DocuPilot AI

DocuPilot AI is a document assistant for property-management and real-estate project records. It indexes uploaded files, answers questions with retrieved source excerpts, and drafts structured project reviews for a human to verify.

[View the demo video](./DocuPilot.mp4)

## Capabilities

- Upload multiple documents and search the full selected project or one document.
- Ask follow-up questions with bounded conversation context and source citations.
- Draft risk, cost and timeline, or compliance reviews with evidence excerpts.
- Generate structured review drafts and download the completed report as Markdown. Premium library CSV export is a separate feature in the Document library tab.
- Configure Basic, Standard, or Premium access in the backend.

## Technology

Streamlit, FastAPI, PostgreSQL with pgvector, SQLAlchemy, and Sentence Transformers for local embeddings by default. Chat generation uses the configured OpenAI-compatible provider.

## Run with Docker Compose

Requirements: Docker Desktop with Compose.

```powershell
Copy-Item .env.example .env
# Add a backend provider key to .env; never commit this file.
docker compose up -d --build
docker compose ps
```

The Streamlit interface is at <http://localhost:8501>; the API health endpoint is <http://localhost:8000/health>. Set `DEBUG=true` to enable API documentation at <http://localhost:8000/docs>.

To inspect application logs or stop the services:

```powershell
docker compose logs -f api frontend
docker compose stop
```

For local development, use Python 3.12 and PostgreSQL 16 with pgvector. Copy `.env.example` to `.env`, configure `DATABASE_URL` and a backend chat-provider key, then install and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal, run `streamlit run streamlit_app.py`. The frontend defaults to `http://localhost:8000/api`; Compose sets `API_URL` to `http://api:8000/api`.

## Configuration and limits

Copy `.env.example` to `.env` for configuration names and safe local placeholders. Compose reads package, provider, embedding, PostgreSQL, and upload settings from `.env`; provider keys are used by the API only. Set a real key locally for the chosen provider. Do not commit `.env`.

Key settings include `PACKAGE`, `MAX_UPLOAD_SIZE_MB`, `LLM_PROVIDER`, `LLM_MODEL`, `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GROQ_API_KEY`, `EMBEDDING_BACKEND`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `LOCAL_EMBEDDINGS_ENABLED`, `LOCAL_EMBEDDING_MODEL`, `LOCAL_EMBEDDING_DIMENSIONS`, `EMBEDDING_FALLBACK_LOCAL`, `DATABASE_URL`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `UPLOAD_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, and `DEBUG`. The example file also includes Compose cache paths and the local frontend `API_URL`.

`PACKAGE` selects Basic, Standard, or Premium access. Upload-size caps are 10 MB, 20 MB, and 50 MB respectively, limited further by `MAX_UPLOAD_SIZE_MB`. Formats are enforced by the backend package configuration:

| Package | Accepted extensions |
|---|---|
| Basic | PDF, DOCX, TXT, Markdown |
| Standard | Basic plus RTF, CSV, XLSX |
| Premium | Standard plus DOC, XLS, DWG, RVT, IFC, JPG/JPEG, PNG, TIFF/TIF, PPTX, ZIP |

CSV extraction is capped at 10,000 rows, 100 columns, 2,048 characters per cell, and 50,000 extracted characters; over-limit files are rejected whole. Accepted extensions do not guarantee useful text: scanned PDFs are not OCR-processed, ZIP files contribute filenames only, and image, legacy DOC, and CAD extraction may be limited. Unreadable or text-empty files are rejected.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app tests streamlit_app.py
docker compose config --quiet
```

Unit tests mock expensive provider and embedding calls where appropriate. They are not a live-service test or a guarantee of extraction quality.

## Limitations

Reviews use retrieved excerpts and are not an exhaustive audit. Precise page, section, slide, or row locations are not retained by the current index. Compliance review needs a project-specific checklist; without one, the app cannot determine compliance. AI-generated answers and reports are drafts, not verified facts, final decisions, legal advice, or proof of compliance. Check all findings and citations against the original documents.
