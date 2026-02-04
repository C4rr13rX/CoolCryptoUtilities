# Textbook Pipeline (c0d3r)

This folder contains the textbook ingestion + text extraction pipeline copied from StateOfLoci.

## Scripts (meta commands)
- fetch-libretexts.mjs: Download LibreTexts PDFs
- segment-textbook.mjs: Extract + segment text into JSON/PNG outputs
- build-seg-dataset.mjs: Build segments.ndjson dataset
- build-tiles.mjs: Build tiles.ndjson dataset
- fix-pages-json.mjs: Repair pages.json counts
- verify-pages-count.mjs: Verify processed pages
- reprocess-incomplete.ps1: Retry incomplete books (Windows)
- import-manifest.py: Import downloads into Django DB + APA citations
- ocr-pages.py: OCR page PNGs locally into Django DB
- prepare-textbook-qa.py: Generate QA candidates locally
- import-qa.py: Import QA candidates into Django DB
- build-knowledge-docs.py: Build KnowledgeDocument rows + queue items

## Dependencies
Install with:
```
::textbook_deps
```

## Usage (from c0d3r tool loop)
```
::textbook_fetch C:/path/to/workspace
::textbook_import C:/path/to/workspace
::textbook_segment C:/path/to/workspace
::textbook_ocr C:/path/to/workspace
::textbook_prepare_qa C:/path/to/workspace
::textbook_import_qa C:/path/to/workspace
::textbook_knowledge C:/path/to/workspace
```

## Database tracking
- Textbooks are stored in Django DB (TextbookSource) with deterministic APA citations.
- OCR text is stored in TextbookPage per page.
- QA candidates are stored in TextbookQACandidate.
- Knowledge docs are stored in KnowledgeDocument + KnowledgeQueueItem for downstream labeling.

Workspace must contain a `textbooks/` directory; scripts will create it if missing.
