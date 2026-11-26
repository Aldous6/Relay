"""Optional, resident RouteLLM-BERT scoring from a verified local checkpoint.

This implements the published three-logit preference aggregation. It is not
the official RouteLLM SDK or a probability that the selected model will pass
the user's verifier. No weights, prompts or generation clients are bundled.
Source: https://github.com/lm-sys/RouteLLM/tree/0b64fdafe049e596a3f5657c219329f24af24198
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import threading
import time


CHECKPOINT = "routellm/bert_gpt4_augmented"
REVISION = "86237e3df400762178ea98379477b8296e66d5e4"
FILES = {
    "model.safetensors": "e2a92240bad1c7fb9befbf021a62fad5feb1e2a3990d53a28d160a39bb1cac7b",
    "config.json": "937704625fe4fb59b9c08ec80292dfd759d5efc75788b45e35ea6e54a3181f91",
    "tokenizer.json": "8373f9cd3d27591e1924426bcc1c8799bc5a9affc4fc857982c5d66668dd1f41",
    "tokenizer_config.json": "f90024142df07163e5e6c5b9a6ad7c8c68b22a9112af11e3db4559a9ff90f737",
    "special_tokens_map.json": "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
    "LICENSE": "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
}


def verify_checkpoint(directory):
    """Verify all executable/model inputs before importing optional libraries."""
    directory = Path(directory)
    informational = {"manifest.json", "README.md", ".gitattributes", ".cache"}
    if any(path.name not in set(FILES) | informational for path in directory.iterdir()):
        raise ValueError("The checkpoint directory contains unverified auxiliary files; use only the pinned files.")
    for name, expected in FILES.items():
        digest = hashlib.sha256()
        with (directory / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Local RouteLLM checkpoint file differs from the pinned revision: {name}")
    return directory


class LocalBERTScorer:
    """Load once per process; score new text without API calls or reloading.

    Initialization verifies files and loads weights on CPU; load_seconds includes
    that work. Each later call exposes truncation so the selector can retain its
    strong baseline when the scorer cannot cover the request.
    """

    def __init__(self, checkpoint_dir, *, threads=4):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer between 1 and 64.")
        started = time.perf_counter()
        directory = verify_checkpoint(checkpoint_dir)
        for package, required in (("torch", "2.9.0"), ("transformers", "4.57.1")):
            if importlib.metadata.version(package).split("+")[0] != required:
                raise ValueError(f"This adapter was verified with {package}=={required}.")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._lock = threading.RLock()
        self._torch = torch
        torch.set_num_threads(threads)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, num_labels=3).to("cpu").eval()
        self.load_seconds = time.perf_counter() - started
        self.provenance = {"implementation": "local_published_routellm_bert_scoring",
                           "checkpoint": CHECKPOINT, "revision": REVISION,
                           "weights_sha256": FILES["model.safetensors"],
                           "input_contract": "caller_supplied_text; pilot used unchanged last user message",
                           "max_input_tokens": 512, "source_supervision": "GPT-4/Mixtral preferences",
                           "success_probability": False, "cpu_threads": threads,
                           "load_seconds": self.load_seconds, "router_cost_usd": None}

    def __call__(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("The scorer requires nonempty routing text.")
        with self._lock:
            tokens = self._tokenizer([text], truncation=False, padding=False,
                                     add_special_tokens=True)["input_ids"][0]
            encoded = self._tokenizer([text], padding=True, truncation=True,
                                      max_length=512, return_tensors="pt")
            retained = int(encoded["attention_mask"].sum().item())
            with self._torch.inference_mode():
                values = self._model(**encoded).logits.detach().cpu().numpy().astype(float)
            # Keep float64 stable softmax, as in the separately audited reference.
            import numpy as np
            if values.shape != (1, 3) or not np.isfinite(values).all():
                raise ValueError("BERT did not return three finite preference logits.")
            probabilities = np.exp(values - values.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            score = float(np.clip(1 - probabilities[:, -2:].sum(axis=1), 0, 1)[0])
            return {"score": score, "input_tokens": len(tokens),
                    "truncated_tokens": max(0, len(tokens) - retained)}


def load_groq_pilot_selector(checkpoint_dir):
    """Explicit experimental profile, threshold frozen on development at 0.58.

    This is not a policy for arbitrary models or Ollama aliases. The historical
    held-out observations establish no net-compute or production savings.
    """
    from ..prompt_selection import PairwisePromptSelector

    scorer = LocalBERTScorer(checkpoint_dir)
    selector = PairwisePromptSelector(
        weak_model="groq/openai/gpt-oss-20b", strong_model="groq/openai/gpt-oss-120b",
        threshold=0.58, scorer=scorer, selector_id="routellm-bert-groq-pilot-v2")
    return selector, dict(scorer.provenance)
