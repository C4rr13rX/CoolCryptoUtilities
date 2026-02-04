#!/usr/bin/env python3
"""
Prepare textbook PDFs into page images + text and generate Q&A candidates.
Outputs under <output_root>:
  - pages/<book_id>/page_0001.png
  - text/<book_id>/page_0001.txt
  - page_manifest.jsonl
  - qa_candidates.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class Tools:
    renderer_cmd: str
    renderer_kind: str
    ocr_cmd: str
    ocr_kind: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_payload(payload: bytes) -> str:
    return hashlib.blake2s(payload).hexdigest()


def normalize_book_id(name: str) -> str:
    cleaned = []
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_", "."}:
            cleaned.append("_")
    collapsed = re.sub(r"_+", "_", "".join(cleaned)).strip("_")
    return collapsed or "book"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_tool(preferred: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if preferred and preferred != "auto":
        if shutil.which(preferred):
            return preferred
        raise SystemExit(f"Requested tool not found on PATH: {preferred}")
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None


def choose_tools(renderer: str, ocr: str) -> Tools:
    renderer_kind = None
    renderer_cmd = None
    if renderer != "auto":
        renderer_cmd = find_tool(renderer, [renderer])
        renderer_kind = renderer
    else:
        for candidate in ["pdftoppm", "mutool"]:
            cmd = find_tool(None, [candidate])
            if cmd:
                renderer_cmd = cmd
                renderer_kind = candidate
                break
    if not renderer_cmd or not renderer_kind:
        raise SystemExit("Missing PDF renderer. Install pdftoppm or mutool.")
    ocr_kind = None
    ocr_cmd = None
    if ocr != "auto":
        ocr_cmd = find_tool(ocr, [ocr])
        ocr_kind = ocr
    else:
        for candidate in ["pdftotext", "tesseract"]:
            cmd = find_tool(None, [candidate])
            if cmd:
                ocr_cmd = cmd
                ocr_kind = candidate
                break
    if not ocr_cmd or not ocr_kind:
        raise SystemExit("Missing OCR/text tool. Install pdftotext or tesseract.")
    return Tools(renderer_cmd=renderer_cmd, renderer_kind=renderer_kind, ocr_cmd=ocr_cmd, ocr_kind=ocr_kind)


def run_command(cmd: Sequence[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return result.stdout


def render_pages(pdf_path: Path, output_dir: Path, renderer_cmd: str, renderer_kind: str, dpi: int, skip_existing: bool) -> List[Path]:
    existing = sorted(output_dir.glob("page_*.png"))
    if existing and skip_existing:
        return existing
    ensure_dir(output_dir)
    if renderer_kind == "pdftoppm":
        prefix = output_dir / "page"
        run_command([renderer_cmd, "-png", "-r", str(dpi), str(pdf_path), str(prefix)])
        pages = []
        for path in output_dir.glob("page-*.png"):
            try:
                suffix = path.stem.split("-")[-1]
                index = int(suffix)
            except ValueError:
                continue
            target = output_dir / f"page_{index:04d}.png"
            if target.exists():
                target.unlink()
            path.rename(target)
            pages.append((index, target))
        return [item[1] for item in sorted(pages, key=lambda item: item[0])]
    if renderer_kind == "mutool":
        pattern = str(output_dir / "page_%04d.png")
        run_command([renderer_cmd, "draw", "-r", str(dpi), "-o", pattern, str(pdf_path)])
        return sorted(output_dir.glob("page_*.png"))
    raise RuntimeError(f"Unsupported renderer: {renderer_kind}")


def extract_text_pdftotext(pdf_path: Path, ocr_cmd: str) -> List[str]:
    output = run_command([ocr_cmd, "-layout", "-enc", "UTF-8", str(pdf_path), "-"])
    pages = output.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def extract_text_tesseract(image_path: Path, ocr_cmd: str, dpi: int) -> str:
    return run_command([ocr_cmd, str(image_path), "stdout", "--dpi", str(dpi)])


def estimate_text_confidence(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    printable = sum(1 for ch in stripped if ch.isprintable())
    letters = sum(1 for ch in stripped if ch.isalpha())
    length_score = min(1.0, len(stripped) / 1200.0)
    ratio = letters / max(1, printable)
    return max(0.0, min(1.0, 0.15 + 0.85 * ratio * length_score))


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sent.strip() for sent in sentences if sent.strip()]


def generate_qa_pairs(text: str, max_pairs: int, include_summary: bool) -> List[dict]:
    normalized = normalize_text(text)
    sentences = split_sentences(normalized)
    pairs: List[dict] = []
    patterns = [
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+is\s+(?P<definition>[^.]{5,220})", re.I), "is"),
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+are\s+(?P<definition>[^.]{5,220})", re.I), "are"),
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+refers to\s+(?P<definition>[^.]{5,220})", re.I), "is"),
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+is defined as\s+(?P<definition>[^.]{5,220})", re.I), "is"),
    ]
    for sentence in sentences:
        if len(pairs) >= max_pairs:
            break
        for pattern, verb in patterns:
            match = pattern.match(sentence)
            if not match:
                continue
            subject = match.group("subject").strip().rstrip(",")
            definition = match.group("definition").strip()
            if len(subject) < 3 or len(definition) < 6:
                continue
            question = f"What {verb} {subject}?"
            pairs.append({"question": question, "answer": definition, "source_sentence": sentence})
            break
    if include_summary and len(pairs) < max_pairs and sentences:
        summary = " ".join(sentences[:3])
        pairs.append({"question": "Summarize the section.", "answer": summary, "source_sentence": summary})
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", help="Folder containing textbook PDFs")
    parser.add_argument("--output-root", default="data/textbooks", help="Output root")
    parser.add_argument("--renderer", default="auto", help="auto, pdftoppm, mutool")
    parser.add_argument("--ocr", default="auto", help="auto, pdftotext, tesseract")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root or os.getenv("TEXTBOOKS_DIR", "textbooks")).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    pages_root = output_root / "pages"
    text_root = output_root / "text"
    ensure_dir(pages_root)
    ensure_dir(text_root)

    tools = choose_tools(args.renderer, args.ocr)

    page_manifest = []
    qa_candidates = []

    pdfs = sorted(input_root.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in input root")
        return 1

    for pdf_path in pdfs:
        book_id = normalize_book_id(pdf_path.stem)
        book_pages_dir = pages_root / book_id
        book_text_dir = text_root / book_id
        ensure_dir(book_pages_dir)
        ensure_dir(book_text_dir)
        pages = render_pages(pdf_path, book_pages_dir, tools.renderer_cmd, tools.renderer_kind, args.dpi, args.skip_existing)
        if tools.ocr_kind == "pdftotext":
            text_pages = extract_text_pdftotext(pdf_path, tools.ocr_cmd)
        else:
            text_pages = []
        for idx, page_img in enumerate(pages, start=1):
            if tools.ocr_kind == "tesseract":
                text = extract_text_tesseract(page_img, tools.ocr_cmd, args.dpi)
                text_source = "tesseract"
            else:
                text = text_pages[idx - 1] if idx - 1 < len(text_pages) else ""
                text_source = "pdftotext"
            confidence = estimate_text_confidence(text)
            txt_path = book_text_dir / f"page_{idx:04d}.txt"
            txt_path.write_text(text, encoding="utf-8")
            page_manifest.append({
                "book_id": book_id,
                "page": idx,
                "image_path": str(page_img),
                "text_path": str(txt_path),
                "text_source": text_source,
                "text_confidence": confidence,
                "sha": hash_payload(text.encode("utf-8")),
                "timestamp": now_iso(),
            })
            for qa in generate_qa_pairs(text, args.max_pairs, include_summary=True):
                qa_candidates.append({
                    "book_id": book_id,
                    "page": idx,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "confidence": confidence,
                    "source_sentence": qa.get("source_sentence", ""),
                    "timestamp": now_iso(),
                })

    (output_root / "page_manifest.jsonl").write_text("\n".join(json.dumps(x) for x in page_manifest), encoding="utf-8")
    (output_root / "qa_candidates.jsonl").write_text("\n".join(json.dumps(x) for x in qa_candidates), encoding="utf-8")
    print(f"Wrote {len(page_manifest)} pages and {len(qa_candidates)} QA candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
