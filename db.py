"""
CodeSense - Database Layer
Uses Supabase (PostgreSQL) when SUPABASE_URL is set (production/cloud),
falls back to SQLite for local development.
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from logger import get_logger

logger = get_logger(__name__)


def _use_supabase() -> bool:
    """Return True if Supabase credentials are available AND package is installed."""
    # First check if supabase package is actually importable
    try:
        from supabase import create_client  # noqa: F401
    except ImportError:
        logger.warning("supabase package not installed — using SQLite fallback")
        return False
    # Then check if credentials exist
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            url = st.secrets.get("SUPABASE_URL", "")
            key = st.secrets.get("SUPABASE_KEY", "")
            if url and key:
                return True
    except Exception:
        pass
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    return bool(url and key)


def _get_supabase():
    """Return a Supabase client."""
    from supabase import create_client
    url, key = "", ""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "") or os.getenv("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "") or os.getenv("SUPABASE_KEY", "")
    except Exception:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


# ── SQLite fallback (local dev) ───────────────────────────────────────────────

import sqlite3
import threading
_local = threading.local()

def _sqlite_conn(db_path: str) -> sqlite3.Connection:
    if not hasattr(_local, "conns"):
        _local.conns = {}
    if db_path not in _local.conns:
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conns[db_path] = conn
    return _local.conns[db_path]


class Database:
    """
    Unified database interface.
    Automatically uses Supabase in production, SQLite locally.
    """

    def __init__(self, db_path: str = "codesense.db") -> None:
        self.db_path    = db_path
        self.use_remote = _use_supabase()
        if self.use_remote:
            self._sb = _get_supabase()
            self._init_remote_schema()
            logger.info("Database: Supabase (PostgreSQL)")
        else:
            self._init_sqlite_schema()
            logger.info("Database: SQLite at %s", db_path)

    # ─── Schema init ─────────────────────────────────────────────────────────

    def _init_sqlite_schema(self) -> None:
        conn = _sqlite_conn(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name     TEXT DEFAULT '',
                is_verified   INTEGER DEFAULT 1,
                is_active     INTEGER DEFAULT 1,
                otp_code      TEXT,
                otp_expiry    TEXT,
                login_attempts INTEGER DEFAULT 0,
                locked_until  TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER,
                language      TEXT NOT NULL,
                filename      TEXT DEFAULT '',
                code_hash     TEXT NOT NULL,
                score         REAL NOT NULL,
                grade         TEXT NOT NULL,
                confidence    REAL DEFAULT 0,
                ml_score      REAL NOT NULL,
                features      TEXT DEFAULT '{}',
                results       TEXT DEFAULT '{}',
                analysis_level TEXT DEFAULT 'standard',
                processing_ms INTEGER DEFAULT 0,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS achievements (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,
                achievement_key TEXT NOT NULL,
                title           TEXT NOT NULL,
                description     TEXT NOT NULL,
                icon            TEXT DEFAULT '🏆',
                earned_at       TEXT NOT NULL,
                UNIQUE(user_id, achievement_key)
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER,
                token      TEXT UNIQUE NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
        """)
        conn.commit()

    def _init_remote_schema(self) -> None:
        """
        Supabase schema is created via the Supabase dashboard SQL editor.
        This method just verifies the connection works.
        """
        try:
            self._sb.table("users").select("id").limit(1).execute()
        except Exception as e:
            logger.warning("Supabase connection check: %s", e)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _q(self, table: str):
        """Return a Supabase table query builder."""
        return self._sb.table(table)

    def _sqlite(self):
        return _sqlite_conn(self.db_path)

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    # ─── Users ───────────────────────────────────────────────────────────────

    def create_user(self, username: str, email: str, password_hash: str,
                    full_name: str = "") -> Optional[int]:
        now = self._now()
        if self.use_remote:
            try:
                res = self._q("users").insert({
                    "username": username, "email": email.lower(),
                    "password_hash": password_hash, "full_name": full_name,
                    "is_verified": True, "is_active": True,
                    "login_attempts": 0,
                    "created_at": now, "updated_at": now,
                }).execute()
                return res.data[0]["id"] if res.data else None
            except Exception as e:
                logger.warning("create_user error: %s", e)
                return None
        else:
            try:
                conn = self._sqlite()
                cur  = conn.execute(
                    "INSERT INTO users (username,email,password_hash,full_name,is_verified,created_at,updated_at) VALUES (?,?,?,?,1,?,?)",
                    (username, email.lower(), password_hash, full_name, now, now),
                )
                conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        if self.use_remote:
            # Don't filter by is_active — boolean comparison varies across
            # supabase-py versions; filter in Python instead
            res = self._q("users").select("*").eq("email", email.lower()).limit(1).execute()
            if res.data:
                u = res.data[0]
                # Accept active users (True, 1, "true", "1", or missing key)
                active = u.get("is_active", True)
                if active not in (False, 0, "false", "0"):
                    return u
            return None
        else:
            row = self._sqlite().execute(
                "SELECT * FROM users WHERE email=? AND is_active=1", (email.lower(),)
            ).fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        if self.use_remote:
            res = self._q("users").select("*").eq("id", user_id).limit(1).execute()
            if res.data:
                u = res.data[0]
                active = u.get("is_active", True)
                if active not in (False, 0, "false", "0"):
                    return u
            return None
        else:
            row = self._sqlite().execute(
                "SELECT * FROM users WHERE id=? AND is_active=1", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_user(self, user_id: int, **kwargs) -> bool:
        kwargs["updated_at"] = self._now()
        if self.use_remote:
            try:
                self._q("users").update(kwargs).eq("id", user_id).execute()
                return True
            except Exception:
                return False
        else:
            sets   = ", ".join(f"{k}=?" for k in kwargs)
            values = list(kwargs.values()) + [user_id]
            conn   = self._sqlite()
            conn.execute(f"UPDATE users SET {sets} WHERE id=?", values)
            conn.commit()
            return True

    def verify_user(self, user_id: int) -> bool:
        return self.update_user(user_id, is_verified=True)

    def set_otp(self, email: str, otp: str, expiry: str) -> bool:
        if self.use_remote:
            try:
                self._q("users").update({"otp_code": otp, "otp_expiry": expiry}).eq("email", email.lower()).execute()
                return True
            except Exception:
                return False
        else:
            conn = self._sqlite()
            conn.execute("UPDATE users SET otp_code=?,otp_expiry=? WHERE email=?", (otp, expiry, email.lower()))
            conn.commit()
            return True

    def increment_login_attempts(self, user_id: int) -> int:
        user = self.get_user_by_id(user_id)
        attempts = (user.get("login_attempts") or 0) + 1
        self.update_user(user_id, login_attempts=attempts)
        return attempts

    def reset_login_attempts(self, user_id: int) -> None:
        self.update_user(user_id, login_attempts=0, locked_until=None)

    # ─── Sessions ────────────────────────────────────────────────────────────

    def create_session(self, user_id: int, token: str, expires_at: str,
                       ip: str = "", ua: str = "") -> bool:
        now = self._now()
        if self.use_remote:
            try:
                self._q("sessions").insert({
                    "user_id": user_id, "token": token,
                    "expires_at": expires_at, "created_at": now,
                }).execute()
                return True
            except Exception:
                return False
        else:
            try:
                conn = self._sqlite()
                conn.execute(
                    "INSERT INTO sessions (user_id,token,expires_at,created_at) VALUES (?,?,?,?)",
                    (user_id, token, expires_at, now),
                )
                conn.commit()
                return True
            except Exception:
                return False

    def get_session(self, token: str) -> Optional[Dict]:
        now = self._now()
        if self.use_remote:
            res = self._q("sessions").select("*").eq("token", token).gt("expires_at", now).limit(1).execute()
            return res.data[0] if res.data else None
        else:
            row = self._sqlite().execute(
                "SELECT * FROM sessions WHERE token=? AND expires_at>?", (token, now)
            ).fetchone()
            return dict(row) if row else None

    def delete_session(self, token: str) -> bool:
        if self.use_remote:
            self._q("sessions").delete().eq("token", token).execute()
        else:
            conn = self._sqlite()
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            conn.commit()
        return True

    # ─── Analyses ────────────────────────────────────────────────────────────

    def save_analysis(self, user_id: int, language: str, filename: str,
                      code: str, score: float, grade: str, confidence: float,
                      ml_score: float, features: dict, results: dict,
                      analysis_level: str = "standard",
                      processing_ms: int = 0) -> Optional[int]:
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        now = self._now()
        record = {
            "user_id": user_id, "language": language, "filename": filename,
            "code_hash": code_hash, "score": round(score, 2), "grade": grade,
            "confidence": round(confidence, 2), "ml_score": round(ml_score, 2),
            "features": json.dumps(features, default=str),
            "results":  json.dumps(results, default=str),
            "analysis_level": analysis_level,
            "processing_ms": processing_ms, "created_at": now,
        }
        if self.use_remote:
            try:
                res = self._q("analyses").insert(record).execute()
                return res.data[0]["id"] if res.data else None
            except Exception as e:
                logger.error("save_analysis error: %s", e)
                return None
        else:
            conn = self._sqlite()
            cur  = conn.execute(
                """INSERT INTO analyses (user_id,language,filename,code_hash,score,grade,
                   confidence,ml_score,features,results,analysis_level,processing_ms,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                list(record.values()),
            )
            conn.commit()
            return cur.lastrowid

    def get_user_analyses(self, user_id: int, limit: int = 50,
                          language: Optional[str] = None) -> List[Dict]:
        if self.use_remote:
            q = self._q("analyses").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit)
            if language:
                q = q.eq("language", language)
            res  = q.execute()
            rows = res.data or []
        else:
            query  = "SELECT * FROM analyses WHERE user_id=?"
            params: list = [user_id]
            if language:
                query += " AND language=?"
                params.append(language)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = [dict(r) for r in self._sqlite().execute(query, params).fetchall()]

        for row in rows:
            if isinstance(row.get("features"), str):
                row["features"] = json.loads(row["features"] or "{}")
            if isinstance(row.get("results"), str):
                row["results"]  = json.loads(row["results"] or "{}")
        return rows

    def get_analysis_stats(self, user_id: int) -> Dict:
        if self.use_remote:
            res = self._q("analyses").select("score,processing_ms").eq("user_id", user_id).order("created_at", desc=True).limit(100).execute()
            rows = res.data or []
        else:
            rows = [dict(r) for r in self._sqlite().execute(
                "SELECT score,processing_ms FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT 100",
                (user_id,)
            ).fetchall()]

        if not rows:
            return {"total": 0, "avg_score": 0, "max_score": 0, "min_score": 0, "avg_ms": 0, "recent_improvement": 0.0}

        scores = [r["score"] for r in rows]
        ms     = [r.get("processing_ms") or 0 for r in rows]
        recent_improvement = 0.0
        if len(scores) >= 10:
            recent_improvement = round(sum(scores[:5]) / 5 - sum(scores[5:10]) / 5, 2)

        return {
            "total":               len(scores),
            "avg_score":           round(sum(scores) / len(scores), 2),
            "max_score":           round(max(scores), 2),
            "min_score":           round(min(scores), 2),
            "avg_ms":              round(sum(ms) / len(ms), 0),
            "recent_improvement":  recent_improvement,
        }

    # ─── Achievements ────────────────────────────────────────────────────────

    def award_achievement(self, user_id: int, key: str, title: str,
                          description: str, icon: str = "🏆") -> bool:
        now = self._now()
        if self.use_remote:
            try:
                self._q("achievements").upsert({
                    "user_id": user_id, "achievement_key": key,
                    "title": title, "description": description,
                    "icon": icon, "earned_at": now,
                }, on_conflict="user_id,achievement_key").execute()
                return True
            except Exception:
                return False
        else:
            try:
                conn = self._sqlite()
                conn.execute(
                    "INSERT OR IGNORE INTO achievements (user_id,achievement_key,title,description,icon,earned_at) VALUES (?,?,?,?,?,?)",
                    (user_id, key, title, description, icon, now),
                )
                conn.commit()
                return True
            except Exception:
                return False

    def get_user_achievements(self, user_id: int) -> List[Dict]:
        if self.use_remote:
            res = self._q("achievements").select("*").eq("user_id", user_id).order("earned_at", desc=True).execute()
            return res.data or []
        else:
            rows = self._sqlite().execute(
                "SELECT * FROM achievements WHERE user_id=? ORDER BY earned_at DESC", (user_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def has_achievement(self, user_id: int, key: str) -> bool:
        if self.use_remote:
            res = self._q("achievements").select("id").eq("user_id", user_id).eq("achievement_key", key).limit(1).execute()
            return bool(res.data)
        else:
            row = self._sqlite().execute(
                "SELECT 1 FROM achievements WHERE user_id=? AND achievement_key=?", (user_id, key)
            ).fetchone()
            return row is not None