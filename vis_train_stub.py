#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_verl_batches.py

Visualise Verl DataProto pickles (e.g. gen_batch.pkl / test_gen_batch.pkl) to JSONL.
The output file shares the same basename, but with `.jsonl` extension.

Key features
------------
1. Automatically decodes both `input_ids` and `raw_prompt_ids`.
2. If the batch size equals 32 and `SELECT_FIRST_IF_BATCH_32` is True,
   only the first record (index 0) is exported.
3. You can override which records to keep via `SAMPLE_INDICES`.
4. Robust `decode_ids()` that copes with nested list / torch.Tensor / numpy arrays.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import AutoTokenizer
import numpy as np

CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)


def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        # strip NUL (\x00) and other C0 control chars
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj


# ---------------------------------------------------------------------
# User-configurable section
# ---------------------------------------------------------------------
TOKENIZER_NAME: str = "Qwen/Qwen2.5-7B-Instruct"

PKL_FILES: Sequence[str] = [
    "./gen_batch.pkl",
    "./chat_scheduler_generate_sequences_output_batch_1.pkl",
    "./chat_scheduler_generate_sequences_output_repeated_batch_1.pkl",
    "./chat_scheduler_generate_sequences_output_chunk_batch_1.pkl",
    "./chat_scheduler_generate_sequences_output_chunk_batch_2.pkl",
    "./run_llm_loop_async_batch1.pkl",
]

# If True and the file contains exactly 32 samples, keep only record 0
SELECT_FIRST_IF_BATCH_32: bool = True

# Explicit indices to keep (overrides the 32-check if not None)
# Example: SAMPLE_INDICES = [0, 5, 9]
SAMPLE_INDICES: Sequence[int] | None = None
# ---------------------------------------------------------------------


def parse_chat_messages(decoded_text: str) -> List[Dict[str, str]]:
    """
    Extract messages of form <|im_start|>role\ncontent<|im_end|>
    Returns a list of {"role": ..., "message": ...}
    """
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    return [
        {"role": role, "message": content.strip()}
        for role, content in re.findall(pattern, decoded_text, flags=re.DOTALL)
    ]


def to_list(x):
    """
    Convert torch.Tensor / numpy.ndarray / list-like to a Python list.
    Keeps nested structure intact.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    try:
        # covers numpy arrays
        import numpy as np  # lazy import

        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:  # numpy might not be installed
        pass
        
    return list(x)


def decode_ids(tokenizer, ids: Iterable[int] | Sequence):
    """
    Decode tokens to string; robust against nested list / Tensor.
    Recursively unwraps until reaching a flat list[int].
    """
    ids = to_list(ids)  # ensure Python list
    while ids and isinstance(ids[0], (list, tuple, torch.Tensor)):
        ids = to_list(ids[0])
    ids = [int(i) for i in ids if i != tokenizer.pad_token_id]
    return tokenizer.decode(ids, skip_special_tokens=False)


def get_sample_indices(num_samples: int) -> List[int]:
    """Decide which sample indices to keep for a given file."""
    if SAMPLE_INDICES is not None:
        # Explicit override – filter invalid indices silently
        return [i for i in SAMPLE_INDICES if 0 <= i < num_samples]

    # if SELECT_FIRST_IF_BATCH_32 and num_samples == 32:
    #     return [0

    return list(range(num_samples))


def process_one_file(path: str, tokenizer):
    """Read a DataProto pickle → dump one JSONL per file."""
    if not os.path.isfile(path):
        print(f"[Skip] File not found: {path}")
        return

    print(f"\n[Processing] {path}")
    with open(path, "rb") as f:
        dp = pickle.load(f)

    batch = dp.batch
    non_tensor = dp.non_tensor_batch

    input_ids = batch["input_ids"]
    raw_prompt_ids = non_tensor.get("raw_prompt_ids", None)
    extra_info = non_tensor.get("extra_info", None)

    # Handle both 1-D object arrays and 2-D tensors for prompts
    if input_ids.dim() == 1:  # shape = [seq_len]
        num_samples = 1
    else:                     # shape = [batch, seq_len, ...]
        num_samples = input_ids.shape[0]
    
    keep_indices = get_sample_indices(num_samples)
    print(f"Samples in file: {num_samples}  →  exporting indices: {keep_indices}")

    records = []
    for idx in keep_indices:
        # ------------- Decode input -------------
        decoded_input = decode_ids(tokenizer, input_ids[idx] if num_samples > 1 else input_ids)

        # ------------- Decode prompt (optional) -------------
        decoded_prompt = None
        if raw_prompt_ids is not None:
            decoded_prompt = decode_ids(tokenizer, raw_prompt_ids[idx] if num_samples > 1 else raw_prompt_ids)

        # ------------- Build record -------------
        rec: Dict[str, Any] = {
            "id": int(idx),
            "messages": parse_chat_messages(decoded_input),
            "decoded_input": decoded_input,
            "decoded_prompt": decoded_prompt,
        }

        # extra_info may be None or shorter than num_samples
        if extra_info is not None and idx < len(extra_info):
            rec["extra_info"] = extra_info[idx] if num_samples > 1 else extra_info

        # Copy any additional optional non-tensor fields (traj_ids, tools_kwargs, etc.)
        for k in ("traj_ids", "tools_kwargs"):
            if k in non_tensor and idx < len(non_tensor[k]):
                rec[k] = non_tensor[k][idx] if num_samples > 1 else non_tensor[k]

        records.append(rec)
    print(len(records))

    # ------------- Write JSONL -------------
    out_path = os.path.splitext(path)[0] + ".jsonl"
    with open(out_path, "w", encoding="utf-8") as fw:
        for rec in records:
            fw.write(json.dumps(sanitize_request(rec), ensure_ascii=False) + "\n")

    print(f"✅ Saved → {out_path}")


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for pkl in PKL_FILES:
        process_one_file(pkl, tokenizer)


if __name__ == "__main__":
    main()
