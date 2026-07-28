from __future__ import annotations

import json
import datetime as dt
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class LongTermMemory:
    """
    Main LT Memory Module: stores all session user requests and model responses,
    full transcripts, and code, organised by date and searchable by context.

    Every turn is appended to a JSONL file.  Retrieval is keyword-based
    (simple for now; future: semantic search via embeddings or Kuzu).
    """

    MAX_ENTRIES: int = 10_000

    def __init__(self, runtime_root: Path) -> None:
        self._path = runtime_root / "lt_memory.jsonl"
        self._db_path = runtime_root / "lt_memory.sqlite3"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS memory_turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts_epoch REAL NOT NULL,
                ts_text TEXT NOT NULL, session_id TEXT NOT NULL DEFAULT '',
                workdir TEXT NOT NULL DEFAULT '', model_id TEXT NOT NULL DEFAULT '',
                user_text TEXT NOT NULL, model_text TEXT NOT NULL,
                context_text TEXT NOT NULL DEFAULT '')""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_ts ON memory_turns(ts_epoch DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_turns(session_id, ts_epoch DESC)")
            count = int(connection.execute("SELECT COUNT(*) FROM memory_turns").fetchone()[0])
            if count == 0 and self._path.exists():
                self._migrate_jsonl(connection)

    def _migrate_jsonl(self, connection: sqlite3.Connection) -> None:
        try:
            lines = self._path.read_text(encoding="utf-8", errors="ignore").splitlines()[-self.MAX_ENTRIES:]
        except Exception:
            return
        for line in lines:
            try:
                record = json.loads(line); stamp = str(record.get("ts") or "")
                epoch = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp() if stamp else time.time()
                connection.execute(
                    "INSERT INTO memory_turns(ts_epoch,ts_text,session_id,workdir,model_id,user_text,model_text,context_text) VALUES(?,?,?,?,?,?,?,?)",
                    (epoch, stamp, str(record.get("session_id") or ""), str(record.get("workdir") or ""), str(record.get("model_id") or ""), str(record.get("user") or ""), str(record.get("model") or ""), str(record.get("context_snippet") or "")),
                )
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(
        self,
        user_input: str,
        model_output: str,
        *,
        context: str = "",
        workdir: str = "",
        model_id: str = "",
        session_id: str = "",
    ) -> None:
        """Append one conversation turn to the long-term store."""
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "workdir": workdir,
            "model_id": model_id,
            "user": user_input[:8000],
            "model": model_output[:8000],
            "context_snippet": context[:1000],
        }
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO memory_turns(ts_epoch,ts_text,session_id,workdir,model_id,user_text,model_text,context_text) VALUES(?,?,?,?,?,?,?,?)",
                    (time.time(), record["ts"], session_id, workdir, model_id, record["user"], record["model"], record["context_snippet"]),
                )
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 20) -> list[dict]:
        """
        Keyword search over stored entries.
        Returns up to `limit` matching records, newest first.
        """
        if not query:
            return []
        tokens = self._query_tokens(query)
        start, end = self._date_range(query)
        clauses: list[str] = []; params: list[object] = []
        if tokens:
            term_clauses=[]
            for token in tokens[:12]:
                term_clauses.append("lower(user_text||' '||model_text||' '||context_text||' '||workdir) LIKE ?")
                params.append(f"%{token}%")
            clauses.append("("+" OR ".join(term_clauses)+")")
        if start is not None:
            clauses.append("ts_epoch >= ?"); params.append(start)
        if end is not None:
            clauses.append("ts_epoch < ?"); params.append(end)
        where = " WHERE "+" AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows=connection.execute("SELECT * FROM memory_turns"+where+" ORDER BY ts_epoch DESC LIMIT 500",params).fetchall()
        ranked=[]
        for row in rows:
            record=self._row_dict(row); blob=f"{record['user']} {record['model']} {record['context_snippet']} {record['workdir']}".lower()
            score=sum(1 for token in tokens if token in blob)
            ranked.append((score,float(row["ts_epoch"]),record))
        ranked.sort(key=lambda item:(item[0],item[1]),reverse=True)
        return [item[2] for item in ranked[:max(1,limit)]]

    def recent(self, *, limit: int = 10, session_id: str = "") -> list[dict]:
        """Return the most recent `limit` entries, optionally filtered by session."""
        with self._connect() as connection:
            if session_id:
                rows=connection.execute("SELECT * FROM memory_turns WHERE session_id=? ORDER BY ts_epoch DESC LIMIT ?",(session_id,max(1,limit))).fetchall()
            else:
                rows=connection.execute("SELECT * FROM memory_turns ORDER BY ts_epoch DESC LIMIT ?",(max(1,limit),)).fetchall()
        return [self._row_dict(row) for row in rows]

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> dict:
        return {"ts":row["ts_text"],"session_id":row["session_id"],"workdir":row["workdir"],"model_id":row["model_id"],"user":row["user_text"],"model":row["model_text"],"context_snippet":row["context_text"]}

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        stop={"do","you","remember","when","we","were","working","on","the","a","an","at","in","from","about","what","did","last","time","please","find","recall"}
        return list(dict.fromkeys(token for token in re.findall(r"[a-z0-9_.-]{2,}",query.lower()) if token not in stop and not re.fullmatch(r"\d{4}-\d{2}-\d{2}",token)))

    @staticmethod
    def _date_range(query: str) -> tuple[float|None,float|None]:
        now=dt.datetime.now().astimezone(); lower=query.lower(); day=None
        match=re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b",lower)
        if match:
            try: day=dt.datetime(int(match[1]),int(match[2]),int(match[3]),tzinfo=now.tzinfo)
            except ValueError: day=None
        if day is None:
            slash=re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b",lower)
            if slash:
                try: day=dt.datetime(int(slash[3]),int(slash[1]),int(slash[2]),tzinfo=now.tzinfo)
                except ValueError: day=None
        if day is None:
            named=re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(20\d{2})\b",lower)
            if named:
                try: day=dt.datetime(int(named[3]),dt.datetime.strptime(named[1].title(),"%B").month,int(named[2]),tzinfo=now.tzinfo)
                except ValueError: day=None
        if day is None and "yesterday" in lower:
            day=(now-dt.timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0)
        elif day is None and "today" in lower:
            day=now.replace(hour=0,minute=0,second=0,microsecond=0)
        elif day is None and "last week" in lower:
            return (now-dt.timedelta(days=7)).timestamp(),now.timestamp()
        elif day is None and "last month" in lower:
            return (now-dt.timedelta(days=31)).timestamp(),now.timestamp()
        if day is None: return None,None
        return day.timestamp(),(day+dt.timedelta(days=1)).timestamp()

    # ------------------------------------------------------------------
    # Efficient tail reading
    # ------------------------------------------------------------------

    def _tail_lines(self, max_lines: int = 200) -> list[str]:
        """
        Read the last `max_lines` lines from the JSONL file without loading
        the entire file into memory.  Returns lines in reverse order
        (newest first).
        """
        if not self._path.exists():
            return []
        try:
            size = self._path.stat().st_size
            if size == 0:
                return []
            # Read at most 2 MB from the tail — enough for ~200 entries.
            read_size = min(size, 2 * 1024 * 1024)
            with self._path.open("rb") as fh:
                fh.seek(max(0, size - read_size))
                chunk = fh.read().decode("utf-8", errors="ignore")
            lines = chunk.splitlines()
            # The first line may be partial if we seeked mid-line; drop it
            # unless we read from the start.
            if size > read_size and lines:
                lines = lines[1:]
            # Return newest first, capped at max_lines.
            lines.reverse()
            return lines[:max_lines]
        except Exception:
            return []
