import os
import json
import pickle
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from tqdm import tqdm
import random


REWARD_PAT = "<<< Finished >>><reward>1.0</reward>"
# 你也可以把这个正则化，但对固定字符串直接包含判断最快

TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# TOKENIZER_MODEL = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
verl_tool_base_path = "/data/minimax-dialogue/users/ruobai/rl_r2e"

ROLE_ALIAS = {"system": "system", "user": "human", "assistant": "gpt"}

def _sanitize_for_sharegpt(messages: list[dict]) -> list[dict]:
    """过滤并重排，满足 ShareGPT 要求；无合法对话返回 []"""
    system_msgs, dialog = [], []
    for m in messages:
        role = m.get("role", "").lower()
        content = m.get("content", "").strip()
        if not content:
            continue
        mapped = ROLE_ALIAS.get(role, "gpt")
        if mapped == "system" and not system_msgs:          # 仅保留首条 system
            system_msgs.append({"role": "system", "content": content})
        else:
            dialog.append({"role": mapped, "content": content})
    pairs, i = [], 0
    while i + 1 < len(dialog):
        a, b = dialog[i], dialog[i + 1]
        if a["role"] == "human" and b["role"] == "gpt":
            pairs.extend([a, b]); i += 2
        else:
            i += 1
    return system_msgs + pairs if pairs else []
# ---------------------------------------

def parse_chat_messages(decoded_text: str, remove_think=False):
    """
    把 <|im_start|>role\ncontent<|im_end|> … 这种文本拆成 list[dict]
    返回 [{"role": xxx, "content": yyy}, …] ，用于 OpenAI 格式
    """
    import re

    if remove_think:
        decoded_text = re.sub(r"<think>.*?</think>", "", decoded_text, flags=re.S)

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


def build_from_verl_tool(
    input_dirs: List[str],
    output_path: str,
    remove_think: bool = False,
    output_format: str = "default",  # "default" or "sharegpt"
):
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
                msgs = parse_chat_messages(text, remove_think=remove_think)
                if not msgs:
                    continue  # 数据坏了

                # OpenAI D/S 格式：每个样本是 {"messages":[...]}
                # fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                # n_kept += 1
                if output_format == "sharegpt":
                    msgs = _sanitize_for_sharegpt(msgs)
                    if not msgs:
                        continue
                fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                n_kept += 1


    print(f"Done. kept {n_kept}/{n_total} samples -> {output_path}")

# ---------------- 1. build_from_sweswe_xiancai (仅小修，核心逻辑不变) -------------
def build_from_sweswe_xiancai(input_dirs: list[str], output_path: str):
    """
    将默认格式 ({'messages':[...], ...}) 转成 ShareGPT 格式并落盘。
    不再做任何 dedup / 抽样；只是格式转换 + 基础合法性过滤。
    """
    n_total, n_kept = 0, 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for d in input_dirs:
            for fp in sorted(Path(d).rglob("*.jsonl")):
                for line in Path(fp).read_text("utf-8").splitlines():
                    if not line.strip():
                        continue
                    n_total += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msgs = obj.get("messages") or obj.get("dialog") or []
                    msgs = _sanitize_for_sharegpt(msgs)
                    if not msgs:
                        continue                     # 被过滤
                    fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                    n_kept += 1

    print(f"[build_from_sweswe_xiancai] kept {n_kept}/{n_total} samples -> {output_path}")

# ---------------------------  2. ShareGPT 子抽样  --------------------------- #
# ---------------- 2. subsample_dataset_sharegpt (改动较大，符合“不可重复”语义) -----
def subsample_dataset_sharegpt(
    input_path: str,
    output_path: str,
    n_samples: int,
    strategy: str = "random",   # random / sequential / reduplicate_random / reduplicate_sequential
):
    """
    从 ShareGPT 数据集中抽 n_samples 条：
    1⃣️ 先依据第一条 human 消息内容去重。
    2⃣️ 再抽样，**绝不重复**。若 n > unique_count，则只输出 unique_count 条。
    3⃣️ strategy == 'random' 时随机洗牌；'sequential' 时保持文件顺序。
    """
    # 读取并去重
    if strategy == "reduplicate_random" or strategy == "reduplicate_sequential":
        uniq = {}
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line)
                first = obj["messages"][1]["content"].strip() if obj["messages"] else ""
                # 保留出现的第一条即可
                uniq.setdefault(first, line.rstrip("\n"))
        strategy = "random" if strategy == "reduplicate_random" else "sequential"

    pool = list(uniq.values())
    unique_cnt = len(pool)
    if unique_cnt == 0:
        raise RuntimeError("输入数据为空或全部非法")

    # 取样数量 = min(n_samples, unique_cnt)
    take = min(n_samples, unique_cnt)
    if strategy not in ("random", "sequential"):
        raise ValueError(f"未知 strategy: {strategy}")

    if strategy == "random":
        random.shuffle(pool)
    # 截断
    sample_lines = pool[:take]

    # 写盘
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(sample_lines) + "\n")

    print(f"[subsample_dataset_sharegpt] unique={unique_cnt}, "
          f"written={take} -> {output_path}")


if __name__ == "__main__":
    # build_from_verl_tool(["verl_step_records/qwen3-32b-r2e_swe_extra_user-0723-rollout-vllm-2025-07-24-00-33-25/"], "./verl_tool_data_swe_extra.jsonl")
    # build_from_verl_tool(["verl_step_records/qwen3-32b-r2e_sync_extra_user-0723-rollout-vllm-2025-07-24-04-55-34/"], "./verl_tool_data_sync_extra.jsonl")
    # build_from_verl_tool(["verl_step_records/deepswe-preview-r2e_sync_extra_user-0723-rollout-vllm-2025-07-24-20-52-36/"], "./verl_tool_data_sync_extra_deepswe.jsonl")
    # build_from_verl_tool(["verl_step_records/deepswe-preview-r2e_swe_extra_user-0723-rollout-vllm-2025-07-24-16-27-36/"], "./verl_tool_data_swe_extra_deepswe.jsonl")
    # build_from_verl_tool(["verl_step_records/deepswe-preview-r2e_sync_extra_user-0723-rollout-vllm-2025-07-24-20-52-36/"], "./sync_extra_deepswe_remove_think_sharegpt.jsonl", remove_think=True, output_format="sharegpt")
    # build_from_verl_tool(["verl_step_records/deepswe-preview-r2e_swe_extra_user-0723-rollout-vllm-2025-07-24-16-27-36/"], "./swe_extra_deepswe_remove_think_sharegpt.jsonl", remove_think=True, output_format="sharegpt")
    # verl_step_records/xiancai-32b-r2e_swe_extra_user-0723-rollout-vllm-2025-07-25-09-10-52/
    # build_from_verl_tool(["verl_step_records_before0728/xiancai-32b-r2e_swe_extra_user-0723-rollout-vllm-2025-07-25-09-10-52/"], "data/xiancai_swe_extra.jsonl", remove_think=True, output_format="default")
    # build_from_verl_tool(["verl_step_records_before0728/xiancai-32b-r2e_swe_extra_user-0723-rollout-vllm-2025-07-25-09-10-52/"], "data/xiancai_swe_extra_sharegpt.jsonl", remove_think=True, output_format="sharegpt")
    # build_from_verl_tool(["verl_step_records_before0728/xiancai-32b-r2e_sync_extra_user-0723-rollout-vllm-2025-07-25-11-02-56/"], "data/xiancai_sync_extra.jsonl", remove_think=True, output_format="default")
    # build_from_verl_tool(["verl_step_records_before0728/xiancai-32b-r2e_sync_extra_user-0723-rollout-vllm-2025-07-25-11-02-56/"], "data/xiancai_sync_extra_sharegpt.jsonl", remove_think=True, output_format="sharegpt")
    # build_from_sweswe_xiancai(["/data/minimax-dialogue/users/xiancai/verl/rollout_samples"], "data/xiancai_rollout_samples_sharegpt.jsonl")
    subsample_dataset_sharegpt("data/xiancai_rollout_samples_sharegpt.jsonl", "data/xiancai_rollout_samples_sharegpt_subsampled.jsonl", 1000, strategy="reduplicate_random")
    subsample_dataset_sharegpt("data/xiancai_rollout_samples_sharegpt.jsonl", "data/xiancai_rollout_samples_sharegpt_reduplicated.jsonl", 999999, strategy="reduplicate_random")