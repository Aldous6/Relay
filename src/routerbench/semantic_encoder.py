"""Frozen local ONNX E5 encoder with explicit full token-window coverage.

No automatic downloads or remote inference. Each window is contextualized
independently; weighted aggregation does not learn cross-window relationships.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .context import window_token_ids
from .data import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Los archivos del encoder deben permanecer dentro del proyecto.")
    return path


def aggregate_windows(embeddings: np.ndarray, new_token_weights: np.ndarray) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    weights = np.asarray(new_token_weights, dtype=float)
    if vectors.ndim != 2 or weights.shape != (len(vectors),) or not len(vectors):
        raise ValueError("Ventanas y pesos desalineados o vacíos.")
    if not np.isfinite(vectors).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Vectores o pesos inválidos.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Un embedding de ventana es nulo.")
    combined = np.average(vectors / norms, axis=0, weights=weights).astype(np.float32)
    norm = np.linalg.norm(combined)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("El embedding agregado es nulo o no finito.")
    return combined / norm


class WindowedE5Encoder:
    def __init__(self, spec: dict):
        self.spec = dict(spec)
        if self.spec.get("embedding_dimension") != 384:
            raise ValueError("Esta implementación requiere el encoder E5 de 384 dimensiones.")
        if self.spec.get("aggregation") != "new_token_weighted_mean_of_l2_window_embeddings_then_l2":
            raise ValueError("Agregación desconocida.")
        if self.spec.get("token_assembly_version") != "e5-joint-prefix-tokenization-v3" or self.spec.get("prefix") != "query: ":
            raise ValueError("Versión de ensamblado o prefijo E5 desconocidos.")
        for name in ("max_tokens", "batch_size", "threads"):
            if not isinstance(self.spec[name], int) or isinstance(self.spec[name], bool) or self.spec[name] < 1:
                raise ValueError(f"{name} debe ser entero positivo.")
        if self.spec["batch_size"] != 1:
            raise ValueError("Este experimento cuantizado requiere batch_size=1 para una representación independiente por texto.")
        if self.spec["max_tokens"] > 512:
            raise ValueError("El límite por ventana no puede superar el contexto de este encoder.")
        self.model_dir = _project_path(self.spec["model_dir"])
        self._session = None
        self._tokenizer = None
        self._verified = False

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_session"] = None
        state["_tokenizer"] = None
        state["_verified"] = False
        return state

    def _verify_assets(self):
        if self._verified:
            return
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Falta el encoder local. Ejecuta tools/setup_semantic_environment.py en el entorno autorizado.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model_id"] != self.spec["model_id"] or manifest["revision"] != self.spec["revision"]:
            raise ValueError("La revisión del encoder no coincide.")
        for name in (self.spec["model_file"], self.spec["tokenizer_file"]):
            target = (self.model_dir / name).resolve()
            if not target.is_relative_to(self.model_dir) or not target.is_file():
                raise FileNotFoundError(f"Falta un archivo del encoder: {name}")
            actual = sha256_file(target)
            if actual != manifest["files"][name]["sha256"] or (name == self.spec["model_file"] and actual != self.spec["model_sha256"]):
                raise ValueError(f"El archivo del encoder fue modificado: {name}")
        identity = {key: self.spec[key] for key in ("model_id", "revision", "model_sha256", "embedding_dimension", "prefix", "token_assembly_version", "max_tokens", "overlap", "aggregation", "packages", "batch_size", "threads")}
        identity["tokenizer_sha256"] = manifest["files"][self.spec["tokenizer_file"]]["sha256"]
        identity["dtype"] = "float32"
        identity["backend"] = "onnxruntime CPUExecutionProvider"
        self.cache_namespace_ = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.assets_manifest_ = manifest
        self._verified = True

    def _ensure_runtime(self):
        if self._session is not None and self._tokenizer is not None:
            return
        self._verify_assets()
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("Usa .venv-relay/Scripts/python.exe; faltan onnxruntime o tokenizers.") from exc
        for module, name in ((ort, "onnxruntime"),):
            if module.__version__ != self.spec["packages"][name]:
                raise ValueError(f"Versión de {name} distinta del experimento fijado.")
        import tokenizers
        if tokenizers.__version__ != self.spec["packages"]["tokenizers"]:
            raise ValueError("Versión de tokenizers distinta del experimento fijado.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.spec["threads"]
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(self.model_dir / self.spec["model_file"]), sess_options=options, providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(str(self.model_dir / self.spec["tokenizer_file"]))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        # XLM-R tokenizes a trailing isolated space as an extra metaspace token.
        # The payload already begins with the tokenizer's metaspace boundary.
        self._prefix_ids = tokenizer.encode(self.spec["prefix"].rstrip(), add_special_tokens=False).ids
        self._cls = tokenizer.token_to_id("<s>")
        self._sep = tokenizer.token_to_id("</s>")
        self._pad = tokenizer.token_to_id("<pad>")
        if any(value is None for value in (self._cls, self._sep, self._pad)):
            raise ValueError("No se encontraron los tokens especiales esperados.")
        self._capacity = self.spec["max_tokens"] - len(self._prefix_ids) - 2
        if self._capacity < 1 or not isinstance(self.spec["overlap"], int) or not 0 <= self.spec["overlap"] < self._capacity:
            raise ValueError("El solapamiento no cabe después de reservar prefijo y especiales.")
        self._input_names = {item.name for item in session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self._input_names) or self._input_names - {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError(f"Entradas ONNX desconocidas: {self._input_names}")
        outputs = session.get_outputs()
        self._output_name = next((item.name for item in outputs if item.name == "last_hidden_state"), outputs[0].name)
        self.runtime_description_ = {"providers": session.get_providers(), "inputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_inputs()],
                                     "output": self._output_name, "payload_capacity": self._capacity,
                                     "special_ids": {"cls": self._cls, "sep": self._sep, "pad": self._pad}, "prefix_ids": self._prefix_ids}
        self._session, self._tokenizer = session, tokenizer

    def _infer_windows(self, sequences):
        maximum = max(map(len, sequences))
        ids = np.full((len(sequences), maximum), self._pad, dtype=np.int64)
        attention = np.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = sequence
            attention[index, :len(sequence)] = 1
        feeds = {"input_ids": ids, "attention_mask": attention}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        output = self._session.run([self._output_name], feeds)[0]
        if output.shape != (len(sequences), maximum, self.spec["embedding_dimension"]):
            raise ValueError(f"Salida ONNX inesperada: {output.shape}; se esperaban estados por token.")
        pooled = (output * attention[..., None]).sum(axis=1) / attention.sum(axis=1, keepdims=True)
        if not np.isfinite(pooled).all():
            raise ValueError("El encoder produjo embeddings no finitos.")
        return pooled.astype(np.float32)

    def _encode_new(self, texts):
        self._ensure_runtime()
        encoding_start = time.perf_counter()
        token_lists = self._tokenizer.encode_batch([self.spec["prefix"] + text for text in texts], add_special_tokens=False)
        tokenization_seconds = time.perf_counter() - encoding_start
        pending, stats = [], []
        vectors, weights = [[] for _ in texts], [[] for _ in texts]
        for owner, tokens in enumerate(token_lists):
            if tokens.ids[:len(self._prefix_ids)] != self._prefix_ids:
                raise ValueError("El prefijo no tiene una segmentación estable para este texto.")
            payload_ids = tokens.ids[len(self._prefix_ids):]
            windows = window_token_ids(payload_ids, self._capacity, self.spec["overlap"])
            if not windows:
                raise ValueError("El prompt no contiene tokens de texto.")
            previous_end = 0
            for window in windows:
                new_tokens = window.end - previous_end
                if new_tokens < 1 or window.start > previous_end:
                    raise ValueError("La segmentación dejó tokens sin cubrir.")
                sequence = [self._cls, *self._prefix_ids, *window.token_ids, self._sep]
                if len(sequence) > self.spec["max_tokens"]:
                    raise ValueError("Una ventana excedió el límite del encoder.")
                pending.append((owner, sequence, new_tokens))
                previous_end = window.end
            if previous_end != len(payload_ids):
                raise ValueError("El final del prompt no fue cubierto.")
            stats.append({"input_tokens": len(payload_ids), "n_windows": len(windows), "covered_tokens": previous_end,
                          "truncated_tokens": 0, "source_characters": len(texts[owner])})
        # Length bucketing reduces padding work; owner indices preserve the
        # mapping back to original prompts, independently of batch ordering.
        pending.sort(key=lambda item: len(item[1]))
        inference_start = time.perf_counter()
        for start in range(0, len(pending), self.spec["batch_size"]):
            batch = pending[start:start+self.spec["batch_size"]]
            embeddings = self._infer_windows([item[1] for item in batch])
            for (owner, _, weight), vector in zip(batch, embeddings):
                vectors[owner].append(vector)
                weights[owner].append(weight)
        inference_seconds = time.perf_counter() - inference_start
        results = [aggregate_windows(np.asarray(value), np.asarray(weight)) for value, weight in zip(vectors, weights)]
        return results, stats, tokenization_seconds, inference_seconds

    def encode(self, texts, use_cache: bool = True):
        if isinstance(texts, (str, bytes)):
            raise TypeError("Se requiere una secuencia de prompts, no un string individual.")
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Se requieren prompts de texto no vacíos.")
        self._verify_assets()
        started = time.perf_counter()
        unique_texts = list(dict.fromkeys(texts))
        from .embedding_cache import EmbeddingCache
        cache = EmbeddingCache(_project_path(self.spec["cache_path"]), self.cache_namespace_, self.spec["embedding_dimension"]) if use_cache else None
        found = cache.get_many(unique_texts) if cache is not None else {}
        missing = [text for text in unique_texts if text not in found]
        tokenization_seconds = inference_seconds = 0.0
        for start in range(0, len(missing), 128):
            batch = missing[start:start+128]
            vectors, stats, tokenize_time, inference_time = self._encode_new(batch)
            tokenization_seconds += tokenize_time
            inference_seconds += inference_time
            records = [(text, vector, stat) for text, vector, stat in zip(batch, vectors, stats)]
            if cache is not None:
                cache.put_many(records)
            found.update({text: (vector, stat) for text, vector, stat in records})
            if len(missing) > 256 and ((start+len(batch)) % 1024 == 0 or start+len(batch) == len(missing)):
                print(f"E5: {start+len(batch)}/{len(missing)} prompts codificados con cobertura completa.", flush=True)
        self.last_stats_ = [dict(found[text][1]) for text in texts]
        self.last_call_ = {"prompts": len(texts), "unique_prompts": len(unique_texts), "cache_misses": len(missing),
                           "cache_hits": len(unique_texts)-len(missing), "tokenization_seconds": tokenization_seconds,
                           "window_inference_seconds": inference_seconds, "elapsed_seconds": time.perf_counter()-started,
                           "windows": sum(item["n_windows"] for item in self.last_stats_),
                           "input_tokens": sum(item["input_tokens"] for item in self.last_stats_), "truncated_tokens": 0}
        return np.stack([found[text][0] for text in texts]).astype(np.float32)
