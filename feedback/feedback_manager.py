import sqlite3
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from utils.config import DATABASE_PATH, FEEDBACK_DIR
from utils.logger import logger

class FeedbackManager:
    """
    Manages user feedback storage in an ACID SQLite database.
    Stores user verified labels, comments, model predictions, and image references.
    """

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes database schema if not already present."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            file_path TEXT,
            predicted_verdict TEXT NOT NULL,
            predicted_ai_prob REAL NOT NULL,
            predicted_confidence REAL NOT NULL,
            user_label TEXT NOT NULL,
            user_agrees INTEGER NOT NULL,
            comment TEXT,
            verified_by_admin INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending_review',
            metadata_json TEXT
        )
        """)
        conn.commit()
        conn.close()

    def record_feedback(
        self,
        pil_img: Image.Image,
        image_hash: str,
        predicted_verdict: str,
        predicted_ai_prob: float,
        predicted_confidence: float,
        user_label: str,
        user_agrees: bool,
        comment: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Saves a feedback submission and stores image file for retraining dataset."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Save image file safely into feedback pool
        feedback_img_name = f"{image_hash[:16]}_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}.png"
        img_path = FEEDBACK_DIR / feedback_img_name
        try:
            pil_img.save(img_path, format="PNG")
        except Exception as e:
            logger.error(f"Failed to persist feedback image: {e}")
            img_path = None

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO feedback_entries (
            timestamp, image_hash, file_path, predicted_verdict,
            predicted_ai_prob, predicted_confidence, user_label,
            user_agrees, comment, verified_by_admin, status, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            image_hash,
            str(img_path) if img_path else "",
            predicted_verdict,
            float(predicted_ai_prob),
            float(predicted_confidence),
            user_label,
            1 if user_agrees else 0,
            comment,
            0,
            "staged_for_review",
            json.dumps(metadata or {})
        ))
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Recorded user feedback #{entry_id} for hash {image_hash[:8]} (User label: {user_label})")
        return entry_id

    def get_all_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves recent feedback submissions."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT * FROM feedback_entries ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Calculates summary statistics of received user feedback."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM feedback_entries")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback_entries WHERE user_agrees = 1")
        agreed = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback_entries WHERE verified_by_admin = 1")
        verified = cursor.fetchone()[0]

        conn.close()

        agreement_rate = (agreed / total * 100.0) if total > 0 else 100.0
        return {
            "total_submissions": total,
            "user_agreed_count": agreed,
            "user_disagreed_count": total - agreed,
            "user_agreement_rate": round(agreement_rate, 1),
            "verified_count": verified
        }

    def verify_entry(self, entry_id: int, verified: bool = True):
        """Marks a feedback entry as verified by admin for inclusion in retraining."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE feedback_entries
        SET verified_by_admin = ?, status = ?
        WHERE id = ?
        """, (1 if verified else 0, "verified_for_training" if verified else "rejected", entry_id))
        conn.commit()
        conn.close()
