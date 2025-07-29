#!/usr/bin/env python3
"""
把 verl_tool_data.jsonl ➜ verl_tool_sharegpt_clean.json
确保:
  • 仅保留 (user,human) ↔ (assistant,gpt) 成对消息
  • 至少一对，否则丢弃该样本
  • 保留首条 system（若存在）作为 'system' 角色
  • 字段命名严格符合 ShareGPT:   role / content
"""

import argparse, json, sys, pathlib

ROLE_ALIAS = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
    # 任何奇怪的角色统一并入 gpt
}

def sanitize(messages: list[dict]) -> list[dict]:
    """过滤+重排，返回合法 ShareGPT 消息列表；不合法则返回 []"""
    system_msgs, dialog = [], []

    # 1. 拆分 system 与其余
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if not content.strip():
            continue
        mapped = ROLE_ALIAS.get(role, "gpt")
        if mapped == "system" and not system_msgs:          # 只取首条 system
            system_msgs.append({"role": "system", "content": content})
        else:
            dialog.append({"role": mapped, "content": content})

    # 2. 只保留成对  human → gpt
    pairs, i = [], 0
    while i + 1 < len(dialog):
        first, second = dialog[i], dialog[i+1]
        if first["role"] == "human" and second["role"] == "gpt":
            pairs.extend([first, second])
            i += 2
        else:               # 遇到脏数据直接跳过此条，继续找下一个 human
            i += 1

    # 3. 至少要有一对才算有效
    if not pairs:
        return []
    return system_msgs + pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  required=True, dest="in_file")
    ap.add_argument("--out", required=True, dest="out_file")
    args = ap.parse_args()

    in_path  = pathlib.Path(args.in_file)
    out_path = pathlib.Path(args.out_file)
    kept, skipped = 0, 0

    with in_path.open(encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for ln, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                clean = sanitize(rec.get("messages", []))
                if clean:
                    fout.write(json.dumps({"messages": clean}, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                print(f"[WARN] 第 {ln} 行解析失败: {e}", file=sys.stderr)

    print(f"✅ 转换完成  保留 {kept} 条  跳过 {skipped} 条")

if __name__ == "__main__":
    main()
