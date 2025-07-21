import os
import json
import pickle
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from tqdm import tqdm


REWARD_PAT = "<<< Finished >>><reward>1.0</reward>"
# 你也可以把这个正则化，但对固定字符串直接包含判断最快

# TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_MODEL = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
verl_tool_base_path = "/data/minimax-dialogue/users/ruobai/rl_r2e"


def parse_chat_messages(decoded_text: str):
    """
    把 <|im_start|>role\ncontent<|im_end|> … 这种文本拆成 list[dict]
    返回 [{"role": xxx, "content": yyy}, …] ，用于 OpenAI 格式
    """
    import re

    msgs = []
    pat = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    for role, content in re.findall(pat, decoded_text, flags=re.S):
        msgs.append({"role": role, "content": content.strip()})
    return msgs


def iter_pkl_files(dirs: List[str]):
    """递归遍历目录，yield 每个 .pkl 路径"""
    for d in dirs:
        dpath = Path(d)
        if not dpath.exists():
            continue
        for p in dpath.rglob("*.pkl"):
            yield p


def build_from_verl_tool(input_dirs: List[str], output_path: str):
    """
    把 verl_tool 生成的 step‑*.pkl 扫一遍，
    筛出 last_obs 含目标 reward_pat 的样本，
    以 OpenAI messages 结构写入 JSONL
    """
    n_total, n_kept = 0, 0
    input_dirs = [os.path.join(verl_tool_base_path, d) for d in input_dirs]
    with open(output_path, "w", encoding="utf‑8") as fout:
        for pkl_file in tqdm(list(iter_pkl_files(input_dirs)), desc="scanning pkl"):
            try:
                step_obj = pickle.load(open(pkl_file, "rb"))
            except Exception as e:
                print(f"[warn] skip {pkl_file}: {e}")
                continue

            try:
                batch = step_obj.batch
                non_tensor = step_obj.non_tensor_batch
            except AttributeError:
                # 不是 verl 的 StepRecord，忽略
                continue

            input_ids = batch["input_ids"]
            last_obs = non_tensor["last_obs"]

            for i in range(len(input_ids)):
                n_total += 1
                if REWARD_PAT not in last_obs[i]:
                    continue  # 只关心 reward==0.0 的对话

                tokens = input_ids[i]
                tokens = tokens[tokens != tokenizer.pad_token_id]
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                msgs = parse_chat_messages(text)
                if not msgs:
                    continue  # 数据坏了

                # OpenAI D/S 格式：每个样本是 {"messages":[...]}
                fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                n_kept += 1

    print(f"Done. kept {n_kept}/{n_total} samples -> {output_path}")

def build_from_sweswe_xiancai():
    pass


if __name__ == "__main__":
    build_from_verl_tool(["verl_step_records/qwen3-8b-r2e_lite_user-0721-main-vllm-2025-07-21-12-35-47"], "./verl_tool_data.jsonl")