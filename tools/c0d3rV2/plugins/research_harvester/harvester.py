from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import sqlite3
import time
import urllib.parse
import urllib.request
from io import BytesIO
import urllib.robotparser
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class HarvestConfig:
    max_depth: int = 1
    max_pages: int = 12
    max_bytes_per_page: int = 1_000_000
    same_origin: bool = True
    allowed_domains: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    delay_seconds: float = 0.35
    respect_robots: bool = True

    def bounded(self) -> "HarvestConfig":
        return HarvestConfig(
            max_depth=max(0, min(4, int(self.max_depth))),
            max_pages=max(1, min(200, int(self.max_pages))),
            max_bytes_per_page=max(16_384, min(4_000_000, int(self.max_bytes_per_page))),
            same_origin=bool(self.same_origin),
            allowed_domains=tuple(str(item).lower() for item in self.allowed_domains[:20]),
            include_patterns=tuple(str(item) for item in self.include_patterns[:20]),
            exclude_patterns=tuple(str(item) for item in self.exclude_patterns[:20]),
            delay_seconds=max(0.1, min(10.0, float(self.delay_seconds))),
            respect_robots=bool(self.respect_robots),
        )


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text: list[str] = []
        self.title: list[str] = []
        self._ignored = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        clean = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if clean:
            self.text.append(clean)
            if self._in_title:
                self.title.append(clean)


def extract_pdf_text(raw: bytes, fallback_title: str) -> tuple[str, str]:
    """Extract bounded text and title metadata from an archival PDF."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages[:500]:
        value = page.extract_text() or ""
        if value:
            parts.append(value)
        if sum(len(item) for item in parts) >= 2_000_000:
            break
    metadata = reader.metadata or {}
    title = str(getattr(metadata, "title", "") or fallback_title)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:2_000_000], title


class ResearchHarvester:
    """Crawl small, explicit web surfaces and search them locally.

    This is intentionally not a general mirror.  Every crawl has hard page,
    depth, byte, host, timing, and robots.txt bounds.  Stored passages retain
    URL, retrieval time, content hash, depth, and HTTP content type.
    """

    USER_AGENT = "C0d3rV2-ResearchHarvester/1.0 (+local archival research)"

    def __init__(self, runtime_root: str | Path) -> None:
        self.root = Path(runtime_root) / "research_harvester"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "knowledge.sqlite3"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=20)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
                content TEXT NOT NULL, content_hash TEXT NOT NULL, content_type TEXT NOT NULL,
                retrieved_at REAL NOT NULL, depth INTEGER NOT NULL, source_query TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS project_policies (
                project_key TEXT PRIMARY KEY, query TEXT NOT NULL, seeds_json TEXT NOT NULL,
                config_json TEXT NOT NULL, coverage_target REAL NOT NULL,
                refresh_seconds INTEGER NOT NULL, max_rounds INTEGER NOT NULL,
                last_coverage REAL NOT NULL DEFAULT 0, last_run REAL,
                next_refresh REAL, status TEXT NOT NULL DEFAULT 'configured',
                stop_reason TEXT NOT NULL DEFAULT ''
            )""")
            try:
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(title, content, url UNINDEXED, content='documents', content_rowid='id')")
                db.executescript("""
                    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                      INSERT INTO documents_fts(rowid,title,content,url) VALUES(new.id,new.title,new.content,new.url);
                    END;
                    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                      INSERT INTO documents_fts(documents_fts,rowid,title,content,url) VALUES('delete',old.id,old.title,old.content,old.url);
                    END;
                    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                      INSERT INTO documents_fts(documents_fts,rowid,title,content,url) VALUES('delete',old.id,old.title,old.content,old.url);
                      INSERT INTO documents_fts(rowid,title,content,url) VALUES(new.id,new.title,new.content,new.url);
                    END;
                """)
            except sqlite3.OperationalError:
                pass

    @staticmethod
    def _public_host(url: str) -> tuple[bool, str]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False, "Only HTTP(S) public hosts are allowed"
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))}
            if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
                return False, "Host resolves to a private or non-global address"
        except Exception as exc:
            return False, f"DNS validation failed: {exc}"
        return True, ""

    @staticmethod
    def _domain_allowed(host: str, domains: Iterable[str]) -> bool:
        domains = tuple(domains)
        return not domains or any(host == item or host.endswith("." + item) for item in domains)

    def crawl(self, seeds: list[str], *, query: str = "", config: HarvestConfig | None = None) -> dict[str, Any]:
        config = (config or HarvestConfig()).bounded()
        queue = [(url, 0, urllib.parse.urlparse(url).hostname or "") for url in dict.fromkeys(seeds)]
        visited: set[str] = set()
        stored: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        while queue and len(visited) < config.max_pages:
            url, depth, seed_host = queue.pop(0)
            canonical = urllib.parse.urldefrag(url)[0]
            if canonical in visited:
                continue
            visited.add(canonical)
            parsed = urllib.parse.urlparse(canonical)
            host = (parsed.hostname or "").lower()
            valid, reason = self._public_host(canonical)
            if not valid or not self._domain_allowed(host, config.allowed_domains):
                errors.append({"url": canonical, "error": reason or "domain not allowed"})
                continue
            if config.same_origin and seed_host and host != seed_host:
                continue
            if config.include_patterns and not any(re.search(pattern, canonical) for pattern in config.include_patterns):
                continue
            if any(re.search(pattern, canonical) for pattern in config.exclude_patterns):
                continue
            if config.respect_robots:
                origin = f"{parsed.scheme}://{parsed.netloc}"
                robot = robots.get(origin)
                if robot is None:
                    robot = urllib.robotparser.RobotFileParser(origin + "/robots.txt")
                    try:
                        robot.read()
                    except Exception:
                        pass
                    robots[origin] = robot
                if not robot.can_fetch(self.USER_AGENT, canonical):
                    errors.append({"url": canonical, "error": "robots.txt disallows crawl"})
                    continue
            try:
                request = urllib.request.Request(canonical, headers={"User-Agent": self.USER_AGENT, "Accept": "text/html,text/plain,application/json;q=0.8"})
                with urllib.request.urlopen(request, timeout=15) as response:
                    final_url = response.geturl()
                    final_valid, final_reason = self._public_host(final_url)
                    if not final_valid:
                        raise ValueError(final_reason)
                    content_type = str(response.headers.get("Content-Type") or "").lower()
                    raw = response.read(config.max_bytes_per_page + 1)
                if len(raw) > config.max_bytes_per_page:
                    errors.append({"url": canonical, "error": "page exceeded byte limit"})
                    continue
                decoded = raw.decode("utf-8", errors="ignore")
                parser = _PageParser()
                if "html" in content_type:
                    parser.feed(decoded)
                    content = re.sub(r"\s+", " ", " ".join(parser.text)).strip()
                    title = " ".join(parser.title).strip() or final_url
                elif "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
                    content, title = extract_pdf_text(raw, final_url)
                elif any(kind in content_type for kind in ("text/", "json")):
                    content, title = re.sub(r"\s+", " ", decoded).strip(), final_url
                else:
                    continue
                if len(content) < 120:
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                now = time.time()
                with self._connect() as db:
                    db.execute("""INSERT INTO documents(url,title,content,content_hash,content_type,retrieved_at,depth,source_query)
                        VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET title=excluded.title,content=excluded.content,
                        content_hash=excluded.content_hash,content_type=excluded.content_type,retrieved_at=excluded.retrieved_at,
                        depth=excluded.depth,source_query=excluded.source_query""",
                        (final_url, title[:500], content, digest, content_type[:200], now, depth, query[:1000]))
                stored.append({"url": final_url, "title": title[:200], "sha256": digest, "depth": depth, "bytes": len(raw)})
                if depth < config.max_depth and "html" in content_type:
                    for link in parser.links:
                        absolute = urllib.parse.urljoin(final_url, link)
                        if urllib.parse.urlparse(absolute).scheme in {"http", "https"}:
                            queue.append((absolute, depth + 1, seed_host))
                time.sleep(config.delay_seconds)
            except Exception as exc:
                errors.append({"url": canonical, "error": str(exc)[:500]})
        result = {"stored": stored, "errors": errors[-30:], "visited": len(visited), "config": asdict(config)}
        result["coverage"] = self.coverage(query) if query else {}
        return result

    def ingest(self, *, url: str, title: str, content: str, query: str = "", depth: int = 0) -> dict[str, Any]:
        """Store already-fetched verified text from another C0d3rV2 source."""
        normalized = re.sub(r"\s+", " ", content).strip()
        if len(normalized) < 80:
            return {"error": "content is too short to index"}
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        with self._connect() as db:
            db.execute("""INSERT INTO documents(url,title,content,content_hash,content_type,retrieved_at,depth,source_query)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET title=excluded.title,content=excluded.content,
                content_hash=excluded.content_hash,retrieved_at=excluded.retrieved_at,source_query=excluded.source_query""",
                (url, title[:500], normalized, digest, "verified/text", time.time(), depth, query[:1000]))
        return {"status": "indexed", "url": url, "sha256": digest}

    def search(self, query: str, *, limit: int = 6) -> dict[str, Any]:
        stop = {"src", "core", "app", "lib", "tests", "with", "into", "from", "this", "that", "only", "step", "setup", "project", "file", "class", "function"}
        tokens = list(dict.fromkeys(
            token for token in re.findall(r"[a-z0-9_]+", query.lower())
            if len(token) > 2 and token not in stop and not token.isdigit()
        ))[:24]
        if not tokens:
            return {"query": query, "results": [], "coverage": 0.0}
        rows: list[sqlite3.Row] = []
        with self._connect() as db:
            try:
                expression = " OR ".join(f'"{token}"' for token in tokens)
                rows = list(db.execute("""SELECT d.*, bm25(documents_fts) AS rank FROM documents_fts
                    JOIN documents d ON d.id=documents_fts.rowid WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?""",
                    (expression, max(1, min(20, limit)))))
            except sqlite3.OperationalError:
                candidates = list(db.execute("SELECT * FROM documents ORDER BY retrieved_at DESC LIMIT 500"))
                rows = sorted(candidates, key=lambda row: -sum(token in row["content"].lower() for token in tokens))[:limit]
        results = []
        matched_union: set[str] = set()
        for row in rows:
            lower = row["content"].lower()
            matched = [token for token in tokens if token in lower]
            matched_union.update(matched)
            anchors = [lower.find(token) for token in matched if lower.find(token) >= 0]
            center = min(anchors) if anchors else 0
            passage = row["content"][max(0, center - 250):center + 1750]
            results.append({
                "title": row["title"], "url": row["url"], "passage": passage,
                "content_sha256": row["content_hash"], "retrieved_at": row["retrieved_at"],
                "matched_terms": matched,
                "authority_score": self._authority_score(row["url"]),
            })
        results.sort(key=lambda item: (-item["authority_score"], -len(item["matched_terms"])))
        results = results[:limit]
        coverage = len(matched_union) / max(1, len(tokens))
        return {"query": query, "results": results, "coverage": round(coverage, 4), "needs_expansion": coverage < 0.45 or len(results) < 2}

    def document(self, url: str) -> dict[str, Any] | None:
        """Return the exact stored document for provenance verification."""
        with self._connect() as db:
            row = db.execute(
                "SELECT url,title,content,content_hash,content_type,retrieved_at,"
                "depth,source_query FROM documents WHERE url=?",
                (str(url),),
            ).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _authority_score(url: str) -> int:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        official = (
            "threejs.org", "typescriptlang.org", "developer.mozilla.org", "docs.python.org",
            "docs.djangoproject.com", "doc.rust-lang.org", "angular.dev", "react.dev",
            "w3.org", "rfc-editor.org", "nist.gov", "nasa.gov", "openstax.org",
        )
        if any(host == domain or host.endswith("." + domain) for domain in official):
            return 10
        if host.endswith((".gov", ".edu", ".ac.uk")):
            return 8
        if host in {"github.com", "gitlab.com"}:
            return 5
        if any(name in host for name in ("stackoverflow", "stackexchange", "forum", "deepwiki")):
            return 3
        return 4

    def coverage(self, query: str) -> dict[str, Any]:
        result = self.search(query, limit=8)
        return {"coverage": result["coverage"], "documents": len(result["results"]), "needs_expansion": result["needs_expansion"]}

    def status(self) -> dict[str, Any]:
        with self._connect() as db:
            count, newest = db.execute("SELECT COUNT(*), MAX(retrieved_at) FROM documents").fetchone()
            projects = db.execute("SELECT COUNT(*) FROM project_policies").fetchone()[0]
        return {"database": str(self.db_path), "documents": count, "projects": projects, "newest_retrieved_at": newest}

    def configure_project(
        self, project_key: str, *, query: str, seeds: list[str], config: HarvestConfig,
        coverage_target: float = 0.7, refresh_seconds: int = 86_400, max_rounds: int = 2,
    ) -> dict[str, Any]:
        """Persist bounded AI-configurable research controls for one project."""
        key = re.sub(r"[^A-Za-z0-9_.:-]+", "-", project_key.strip())[:240]
        if not key or not query.strip():
            return {"error": "project_key and query are required"}
        bounded = config.bounded()
        target = max(0.25, min(1.0, float(coverage_target)))
        refresh = max(300, min(31_536_000, int(refresh_seconds)))
        rounds = max(1, min(4, int(max_rounds)))
        with self._connect() as db:
            db.execute("""INSERT INTO project_policies(
                project_key,query,seeds_json,config_json,coverage_target,refresh_seconds,max_rounds
            ) VALUES(?,?,?,?,?,?,?) ON CONFLICT(project_key) DO UPDATE SET
                query=excluded.query,seeds_json=excluded.seeds_json,config_json=excluded.config_json,
                coverage_target=excluded.coverage_target,refresh_seconds=excluded.refresh_seconds,
                max_rounds=excluded.max_rounds,
                next_refresh=CASE WHEN project_policies.query<>excluded.query THEN NULL ELSE project_policies.next_refresh END,
                status=CASE WHEN project_policies.query<>excluded.query THEN 'configured' ELSE project_policies.status END,
                stop_reason=CASE WHEN project_policies.query<>excluded.query THEN '' ELSE project_policies.stop_reason END""", (
                key, query.strip()[:2000], json.dumps(list(dict.fromkeys(seeds))[:40]),
                json.dumps(asdict(bounded)), target, refresh, rounds,
            ))
        return self.project_policy(key) or {"error": "policy persistence failed"}

    def project_policy(self, project_key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM project_policies WHERE project_key=?", (project_key,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["seeds"] = json.loads(value.pop("seeds_json") or "[]")
        value["config"] = json.loads(value.pop("config_json") or "{}")
        value["due"] = value.get("next_refresh") is None or float(value["next_refresh"]) <= time.time()
        return value

    def record_project_refresh(self, project_key: str, *, coverage: float, status: str, reason: str) -> dict[str, Any]:
        now = time.time()
        policy = self.project_policy(project_key)
        if policy is None:
            return {"error": "unknown project policy"}
        next_refresh = now + int(policy["refresh_seconds"])
        with self._connect() as db:
            db.execute("""UPDATE project_policies SET last_coverage=?,last_run=?,next_refresh=?,
                status=?,stop_reason=? WHERE project_key=?""", (
                max(0.0, min(1.0, float(coverage))), now, next_refresh,
                status[:80], reason[:500], project_key,
            ))
        return self.project_policy(project_key) or {}
