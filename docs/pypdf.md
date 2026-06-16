# pypdf

Free and open source pure-Python PDF library for splitting, merging, cropping, transforming, and extracting content from PDF files.

## Core Capabilities

- **Manipulation**: Split, merge, crop, transform pages
- **Content extraction**: Text, images, metadata
- **Security**: Passwords, encryption/decryption
- **Enhancement**: Annotations, watermarks, JavaScript, viewer preferences
- **Forms**: Interact with PDF form fields

## Key Classes

- `PdfReader` — read and extract from existing PDFs
- `PdfWriter` — create and modify PDFs
- `PageObject` — individual page manipulation
- `Transformation` — geometric transforms

## Usage in poldercheck

Used in `src/ingest/chunk.py` via `langchain_community.document_loaders.PyPDFLoader` (which wraps pypdf internally) to load party manifesto PDFs (2017–2025) and CPB/PBL reports into text chunks for embedding.

Corrupt PDFs (e.g. HTML error pages saved as .pdf) raise `PdfStreamError` — caught and skipped in the ingestion loop.

## Links
- Docs: https://pypdf.readthedocs.io
- GitHub: https://github.com/py-pdf/pypdf
