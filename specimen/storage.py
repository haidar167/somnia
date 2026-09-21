"""SQLite persistence layer for THE SPECIMEN state machine."""

import sqlite3
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from somnia.utils import project_root


from contextlib import contextmanager


class SpecimenStorage:
    """Persistent SQLite database manager for THE SPECIMEN."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = project_root() / "data" / "specimen_state.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        """Create database tables if they don't exist."""
        with self._get_conn() as conn:
            cur = conn.cursor()

            # Metadata table for lifecycle counters
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Event log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time_str TEXT,
                    timestamp REAL,
                    message TEXT,
                    event_type TEXT
                )
            """)

            # Dream archive table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dreams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    b64 TEXT,
                    target_class INTEGER,
                    pseudo_label INTEGER,
                    weight REAL,
                    p_error REAL,
                    is_nightmare INTEGER,
                    created_at REAL
                )
            """)

            # Anonymous Visitor Memory table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    visitor_id TEXT PRIMARY KEY,
                    first_seen REAL,
                    last_seen REAL,
                    total_feeds INTEGER DEFAULT 0,
                    correct_reveals INTEGER DEFAULT 0,
                    total_reveals INTEGER DEFAULT 0,
                    max_p_error REAL DEFAULT 0.0,
                    last_pred INTEGER,
                    last_mood TEXT
                )
            """)

            # Memory buffer table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_id INTEGER,
                    image_json TEXT,
                    b64 TEXT,
                    pred INTEGER,
                    conf REAL,
                    p_error REAL,
                    true_label INTEGER,
                    timestamp REAL
                )
            """)

            # Taught memories table (The Exchange)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS taught_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_json TEXT,
                    true_label INTEGER,
                    what_the_organism_said INTEGER,
                    p_error_at_feed REAL,
                    visitor_id TEXT,
                    was_correct INTEGER,
                    timestamp REAL,
                    image_hash TEXT,
                    times_taught INTEGER DEFAULT 1,
                    weight REAL DEFAULT 1.0,
                    verdict TEXT DEFAULT ''
                )
            """)
            # Safe schema migrations for existing SQLite databases
            for col, col_type in [
                ("image_hash", "TEXT"),
                ("times_taught", "INTEGER DEFAULT 1"),
                ("weight", "REAL DEFAULT 1.0"),
                ("verdict", "TEXT DEFAULT ''"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE taught_memories ADD COLUMN {col} {col_type}")
                except Exception:
                    pass
            conn.commit()

    # Metadata operations
    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cur.fetchone()
            if row is not None:
                try:
                    return json.loads(row["value"])
                except Exception:
                    return row["value"]
            return default

    def set_meta(self, key: str, value: Any):
        with self._get_conn() as conn:
            cur = conn.cursor()
            val_str = json.dumps(value)
            cur.execute("""
                INSERT INTO metadata (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (key, val_str))
            conn.commit()

    # Event log operations
    def log_event(self, time_str: str, message: str, event_type: str = "info", timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO events (time_str, timestamp, message, event_type)
                VALUES (?, ?, ?, ?)
            """, (time_str, timestamp, message, event_type))
            conn.commit()

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT time_str, message, event_type, timestamp FROM events
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            return [
                {
                    "time": r["time_str"],
                    "message": r["message"],
                    "type": r["event_type"],
                    "timestamp": r["timestamp"]
                }
                for r in rows
            ]

    # Dream archive operations
    def save_dreams(self, dreams: List[Dict[str, Any]]):
        now = time.time()
        with self._get_conn() as conn:
            cur = conn.cursor()
            for d in dreams:
                cur.execute("""
                    INSERT INTO dreams (b64, target_class, pseudo_label, weight, p_error, is_nightmare, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    d["b64"],
                    int(d["target_class"]),
                    int(d["pseudo_label"]),
                    float(d["weight"]),
                    float(d["p_error"]),
                    1 if d["is_nightmare"] else 0,
                    now
                ))
            conn.commit()

    def get_recent_dreams(self, limit: int = 8) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT b64, target_class, pseudo_label, weight, p_error, is_nightmare
                FROM dreams ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            return [
                {
                    "b64": r["b64"],
                    "target_class": r["target_class"],
                    "pseudo_label": r["pseudo_label"],
                    "weight": r["weight"],
                    "p_error": r["p_error"],
                    "is_nightmare": bool(r["is_nightmare"])
                }
                for r in rows
            ]

    # Memory buffer operations
    def add_memory(self, feed_id: int, image_flat: List[float], b64: str, pred: int, conf: float, p_error: float):
        now = time.time()
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO memories (feed_id, image_json, b64, pred, conf, p_error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (feed_id, json.dumps(image_flat), b64, pred, conf, p_error, now))
            conn.commit()

    def update_last_memory_truth(self, true_label: int):
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE memories SET true_label = ?
                WHERE id = (SELECT MAX(id) FROM memories)
            """, (true_label,))
            conn.commit()

    def get_memories(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT feed_id, image_json, b64, pred, conf, p_error, true_label, timestamp
                FROM memories ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            return [
                {
                    "id": r["feed_id"],
                    "image": json.loads(r["image_json"]),
                    "b64": r["b64"],
                    "pred": r["pred"],
                    "conf": r["conf"],
                    "p_error": r["p_error"],
                    "true_label": r["true_label"],
                    "timestamp": r["timestamp"]
                }
                for r in rows
            ]

    # Taught memory operations (The Exchange)
    def add_taught_memory(self, image_flat: List[float], true_label: int,
                          what_the_organism_said: int, p_error_at_feed: float,
                          visitor_id: str, was_correct: bool,
                          image_hash: str = "", times_taught: int = 1,
                          weight: float = 1.0, verdict: str = ""):
        now = time.time()
        with self._get_conn() as conn:
            cur = conn.cursor()
            if image_hash:
                cur.execute("SELECT id, times_taught FROM taught_memories WHERE image_hash = ?", (image_hash,))
                row = cur.fetchone()
                if row is not None:
                    new_count = row["times_taught"] + 1
                    cur.execute("""
                        UPDATE taught_memories
                        SET times_taught = ?, true_label = ?, what_the_organism_said = ?,
                            p_error_at_feed = ?, visitor_id = ?, was_correct = ?, weight = ?, verdict = ?, timestamp = ?
                        WHERE id = ?
                    """, (new_count, true_label, what_the_organism_said, p_error_at_feed, visitor_id, 1 if was_correct else 0, weight, verdict, now, row["id"]))
                    conn.commit()
                    return new_count

            cur.execute("""
                INSERT INTO taught_memories (image_json, true_label, what_the_organism_said, p_error_at_feed, visitor_id, was_correct, timestamp, image_hash, times_taught, weight, verdict)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (json.dumps(image_flat), true_label, what_the_organism_said, p_error_at_feed, visitor_id, 1 if was_correct else 0, now, image_hash, times_taught, weight, verdict))
            conn.commit()
            return times_taught

    def get_taught_memories(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, image_json, true_label, what_the_organism_said, p_error_at_feed, visitor_id, was_correct, timestamp, image_hash, times_taught, weight, verdict
                FROM taught_memories ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "image": json.loads(r["image_json"]),
                    "true_label": r["true_label"],
                    "what_the_organism_said": r["what_the_organism_said"],
                    "p_error": r["p_error_at_feed"],
                    "visitor_id": r["visitor_id"],
                    "was_correct": bool(r["was_correct"]),
                    "timestamp": r["timestamp"],
                    "image_hash": r["image_hash"] if "image_hash" in r.keys() and r["image_hash"] else "",
                    "times_taught": r["times_taught"] if "times_taught" in r.keys() and r["times_taught"] else 1,
                    "weight": r["weight"] if "weight" in r.keys() and r["weight"] is not None else 1.0,
                    "verdict": r["verdict"] if "verdict" in r.keys() and r["verdict"] else "",
                }
                for r in rows
            ]

    def get_taught_count(self) -> int:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as count FROM taught_memories")
            return cur.fetchone()["count"]

    # Visitor memory operations
    def record_visitor_interaction(self, visitor_id: str, p_error: float, pred: int, mood: str) -> Dict[str, Any]:
        """Record or update visitor profile and return their history."""
        now = time.time()
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM visitors WHERE visitor_id = ?", (visitor_id,))
            row = cur.fetchone()

            if row is None:
                # First visit
                cur.execute("""
                    INSERT INTO visitors (visitor_id, first_seen, last_seen, total_feeds, max_p_error, last_pred, last_mood)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                """, (visitor_id, now, now, p_error, pred, mood))
                conn.commit()
                return {
                    "is_returning": False,
                    "total_feeds": 1,
                    "max_p_error": p_error,
                    "previous_mood": mood,
                    "previous_pred": pred,
                    "taught_count": 0,
                    "fooled_count": 0,
                }
            else:
                # Returning visitor
                new_total = row["total_feeds"] + 1
                max_p = max(row["max_p_error"], p_error)
                prev_pred = row["last_pred"]
                prev_mood = row["last_mood"]
                total_rev = row["total_reveals"]
                corr_rev = row["correct_reveals"]

                cur.execute("""
                    UPDATE visitors
                    SET last_seen = ?, total_feeds = ?, max_p_error = ?, last_pred = ?, last_mood = ?
                    WHERE visitor_id = ?
                """, (now, new_total, max_p, pred, mood, visitor_id))
                conn.commit()

                return {
                    "is_returning": True,
                    "total_feeds": new_total,
                    "max_p_error": max_p,
                    "previous_mood": prev_mood,
                    "previous_pred": prev_pred,
                    "taught_count": total_rev,
                    "fooled_count": total_rev - corr_rev,
                }

    def record_visitor_reveal(self, visitor_id: str, was_correct: bool):
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE visitors
                SET total_reveals = total_reveals + 1,
                    correct_reveals = correct_reveals + ?
                WHERE visitor_id = ?
            """, (1 if was_correct else 0, visitor_id))
            conn.commit()

    def get_visitor_profile(self, visitor_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM visitors WHERE visitor_id = ?", (visitor_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "visitor_id": row["visitor_id"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "total_feeds": row["total_feeds"],
                "total_reveals": row["total_reveals"],
                "correct_reveals": row["correct_reveals"],
                "fooled_count": row["total_reveals"] - row["correct_reveals"],
                "max_p_error": row["max_p_error"],
                "last_pred": row["last_pred"],
                "last_mood": row["last_mood"],
            }

    def get_visitor_count(self) -> int:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as count FROM visitors")
            return cur.fetchone()["count"]
