"""Persistent embedding cache keyed by exact text hashes and encoder namespace.

The database stores no prompt-text column. A namespace must identify the encoder
revision, tokenizer, windowing and pooling settings used by the caller. Integrity
hashes detect damaged records; they are not authentication against a third party.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path

import numpy as np


def _reject_nonfinite_json(value: str):
    raise ValueError(f"Constante JSON no finita: {value}")


class EmbeddingCache:
    """Cache complete prompt representations without retaining plaintext inputs.

    Connections are short lived, so this object can be serialized as part of a
    local encoder without carrying an open SQLite handle between processes.
    """

    def __init__(self, path: str | Path, namespace: str, dimension: int = 384):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace debe ser una cadena no vacía.")
        if not isinstance(dimension, Integral) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension debe ser un entero positivo.")
        self.path = Path(path)
        self.namespace = namespace
        self.dimension = int(dimension)

    @staticmethod
    def text_key(text: str) -> str:
        """Hash the exact UTF-8 text, without deduplication normalization."""
        if not isinstance(text, str):
            raise TypeError("El texto de la caché debe ser una cadena.")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @contextmanager
    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = None
        try:
            connection = sqlite3.connect(str(self.path), timeout=30.0)
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                namespace TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                stats_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                PRIMARY KEY (namespace, text_sha256)
            )""")
            connection.commit()
            yield connection
        except sqlite3.DatabaseError as exc:
            raise ValueError("La caché SQLite no se puede leer o está dañada.") from exc
        finally:
            if connection is not None:
                connection.close()

    def _vector_bytes(self, vector: np.ndarray) -> bytes:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(vector, dtype="<f4")
        except (TypeError, ValueError) as exc:
            raise ValueError("El embedding debe ser numérico y convertible a float32.") from exc
        if array.shape != (self.dimension,) or not np.isfinite(array).all() or not np.any(array != 0):
            raise ValueError("El embedding debe tener la dimensión declarada, ser finito y no nulo.")
        return np.ascontiguousarray(array).tobytes()

    @staticmethod
    def _stats_json(stats: dict) -> str:
        if not isinstance(stats, dict):
            raise ValueError("Las estadísticas de caché deben ser un diccionario JSON.")
        try:
            return json.dumps(stats, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Las estadísticas deben ser JSON válido, sin NaN ni infinitos.") from exc

    @staticmethod
    def _record_digest(vector: bytes, stats_json: str) -> str:
        return hashlib.sha256(vector + stats_json.encode("utf-8")).hexdigest()

    def get_many(self, texts: Sequence[str]) -> dict[str, tuple[np.ndarray, dict]]:
        """Return hits by exact input string, preserving misses as absent keys."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts debe ser una secuencia de cadenas.")
        requested = {self.text_key(text): text for text in texts}
        if not requested:
            return {}
        hits = {}
        hashes = list(requested)
        with self._connection() as connection:
            # Keep each statement below SQLite's conservative variable limit.
            for offset in range(0, len(hashes), 400):
                batch = hashes[offset:offset + 400]
                placeholders = ",".join("?" for _ in batch)
                records = connection.execute(
                    "SELECT text_sha256, dimension, vector, stats_json, record_sha256 "
                    f"FROM embeddings WHERE namespace = ? AND text_sha256 IN ({placeholders})",
                    [self.namespace, *batch],
                ).fetchall()
                for text_hash, dimension, blob, stats_json, digest in records:
                    if dimension != self.dimension or not isinstance(blob, bytes) or len(blob) != self.dimension * 4:
                        raise ValueError("Registro de embedding incompatible o dañado en caché.")
                    if not isinstance(stats_json, str) or self._record_digest(blob, stats_json) != digest:
                        raise ValueError("La integridad de un registro de caché no coincide.")
                    vector = np.frombuffer(blob, dtype="<f4").copy()
                    if not np.isfinite(vector).all() or not np.any(vector != 0):
                        raise ValueError("El vector almacenado en caché no es válido.")
                    try:
                        stats = json.loads(stats_json, parse_constant=_reject_nonfinite_json)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Las estadísticas almacenadas en caché están dañadas.") from exc
                    if not isinstance(stats, dict):
                        raise ValueError("Las estadísticas almacenadas no son un diccionario.")
                    hits[requested[text_hash]] = (vector, stats)
        return hits

    def put_many(self, records: Sequence[tuple[str, np.ndarray, dict]]) -> None:
        """Atomically add or replace validated representations in this namespace."""
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records debe ser una secuencia de registros.")
        rows = []
        for text, vector, stats in records:
            text_hash = self.text_key(text)
            blob = self._vector_bytes(vector)
            stats_json = self._stats_json(stats)
            rows.append((self.namespace, text_hash, self.dimension, blob, stats_json, self._record_digest(blob, stats_json)))
        if not rows:
            return
        with self._connection() as connection:
            with connection:
                connection.executemany(
                    "INSERT INTO embeddings (namespace, text_sha256, dimension, vector, stats_json, record_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(namespace, text_sha256) DO UPDATE SET "
                    "dimension=excluded.dimension, vector=excluded.vector, stats_json=excluded.stats_json, record_sha256=excluded.record_sha256",
                    rows,
                )
