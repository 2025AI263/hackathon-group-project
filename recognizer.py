
import os
import sqlite3
import numpy as np
from insightface.app import FaceAnalysis


class FaceDatabase:
    """Small SQLite adapter used by Person 2.

    The class intentionally exposes list_users() and count_samples()
    because Person 4's current app.py calls those methods.
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or os.getenv("FACE_DB_PATH", "face_recognition.db")
        self._create_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES face_users(id)
                )
                """
            )

    def add_embedding(self, name, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM face_users WHERE name = ?", (name,)
            ).fetchone()

            if row is None:
                cur = conn.execute(
                    "INSERT INTO face_users(name) VALUES (?)", (name,)
                )
                user_id = cur.lastrowid
            else:
                user_id = row[0]

            conn.execute(
                "INSERT INTO face_samples(user_id, embedding) VALUES (?, ?)",
                (user_id, sqlite3.Binary(embedding.tobytes())),
            )

    def get_embeddings(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT u.name, s.embedding
                FROM face_users u
                JOIN face_samples s ON u.id = s.user_id
                """
            ).fetchall()

        result = []
        for name, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            result.append((name, vector))
        return result

    def list_users(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM face_users ORDER BY name"
            ).fetchall()
        return [row[0] for row in rows]

    def count_samples(self, name):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM face_samples s
                JOIN face_users u ON u.id = s.user_id
                WHERE u.name = ?
                """,
                (name,),
            ).fetchone()
        return int(row[0])


class FaceRecognizer:
    """Face detection + enrollment + embedding-based recognition.

    Uses InsightFace's pretrained model. It does NOT make fake/random
    predictions and it does NOT use an LLM for identity decisions.

    Recognition distance = 1 - cosine similarity.
    Smaller distance means a better match.
    """

    def __init__(self):
        self.threshold = float(
            os.getenv("FACE_RECOGNITION_THRESHOLD", "0.45")
        )

        db_path = os.getenv("FACE_DB_PATH", "face_recognition.db")
        self.db = FaceDatabase(db_path)

        model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_s")

        self.model = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def _normalize(vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)

        if norm == 0:
            return None

        return vector / norm

    def _get_embedding(self, face):
        embedding = getattr(face, "normed_embedding", None)

        if embedding is None:
            embedding = getattr(face, "embedding", None)

        if embedding is None:
            return None

        return self._normalize(embedding)

    @staticmethod
    def _bbox(face):
        x1, y1, x2, y2 = face.bbox
        return int(x1), int(y1), int(x2), int(y2)

    def enroll(self, name, images):
        name = name.strip()

        if not name:
            return {
                "success": False,
                "message": "Name cannot be empty.",
                "samples_saved": 0,
                "samples_rejected": len(images),
            }

        saved = 0
        rejected = 0

        for image in images:
            try:
                if image is None:
                    rejected += 1
                    continue

                faces = self.model.get(image)

                # Exactly one face is required for enrollment.
                if len(faces) != 1:
                    rejected += 1
                    continue

                embedding = self._get_embedding(faces[0])

                if embedding is None:
                    rejected += 1
                    continue

                self.db.add_embedding(name, embedding)
                saved += 1

            except Exception:
                rejected += 1

        if saved == 0:
            return {
                "success": False,
                "message": (
                    "No valid face sample was saved. "
                    "Use a clear photo containing exactly one face."
                ),
                "samples_saved": 0,
                "samples_rejected": rejected,
            }

        message = f"User '{name}' registered successfully"

        return {
            "success": True,
            "message": message,
            "samples_saved": saved,
            "samples_rejected": rejected,
        }

    def recognize(self, frame):
        if frame is None:
            return {
                "face_detected": False,
                "face_count": 0,
                "message": "Invalid image.",
                "recognized": False,
                "name": "Unknown",
            }

        try:
            faces = self.model.get(frame)
        except Exception as exc:
            return {
                "face_detected": False,
                "face_count": 0,
                "message": f"Face detection failed: {exc}",
                "recognized": False,
                "name": "Unknown",
            }

        face_count = len(faces)

        if face_count == 0:
            return {
                "face_detected": False,
                "face_count": 0,
                "message": "No face detected.",
                "recognized": False,
                "name": "Unknown",
            }

        if face_count > 1:
            return {
                "face_detected": True,
                "face_count": face_count,
                "message": "Please show only one face.",
                "recognized": False,
                "name": "Unknown",
            }

        face = faces[0]
        bbox = self._bbox(face)
        query_embedding = self._get_embedding(face)

        if query_embedding is None:
            return {
                "face_detected": True,
                "face_count": 1,
                "bbox": bbox,
                "recognized": False,
                "name": "Unknown",
                "distance": None,
                "confidence": 0.0,
                "message": "Could not create a face embedding.",
            }

        stored_embeddings = self.db.get_embeddings()

        if not stored_embeddings:
            return {
                "face_detected": True,
                "face_count": 1,
                "bbox": bbox,
                "recognized": False,
                "name": "Unknown",
                "distance": None,
                "confidence": 0.0,
                "message": "No registered users yet",
            }

        best_name = "Unknown"
        best_similarity = -1.0

        for name, stored_embedding in stored_embeddings:
            stored_embedding = self._normalize(stored_embedding)

            if stored_embedding is None:
                continue

            # Both vectors are normalized, so dot product = cosine similarity.
            similarity = float(np.dot(query_embedding, stored_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        if best_similarity < -1:
            return {
                "face_detected": True,
                "face_count": 1,
                "bbox": bbox,
                "recognized": False,
                "name": "Unknown",
                "distance": None,
                "confidence": 0.0,
                "message": "No valid stored embeddings.",
            }

        distance = 1.0 - best_similarity
        recognized = distance <= self.threshold

        confidence = max(0.0, min(100.0, best_similarity * 100.0))

        return {
            "face_detected": True,
            "face_count": 1,
            "bbox": bbox,
            "recognized": recognized,
            "name": best_name if recognized else "Unknown",
            "distance": float(distance),
            "confidence": round(float(confidence), 2),
            "message": "Recognized" if recognized else "Unknown person",
        }
