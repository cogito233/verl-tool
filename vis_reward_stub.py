import pickle
import json
import re
import argparse
import os

def parse_chat_messages(decoded_text):
    """解析decoded文本，提取chat messages"""
    messages = []
    
    # 使用正则表达式匹配 <|im_start|>role\ncontent<|im_end|> 格式
    pattern = r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern, decoded_text, re.DOTALL)
    
    for role, content in matches:
        messages.append({
            "role": role,
            "message": content.strip()
        })
    
    return messages

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='解码pkl文件中的token数据为jsonl格式的聊天记录')
    # parser.add_argument('--input', '-i', default="verl_step_records/r2egym-32b-agent-r2e_lite-0630-bs256-2025-07-01-16-29-40/step-31.pkl", help='输入pkl文件路径')
    parser.add_argument('--input', '-i', default="verl_step_records/r2egym-7b-agent-r2e_swe_verified_user-0704-magic-2025-07-07-07-47-51/step-val-0.pkl", help='输入pkl文件路径')
    # parser.add_argument('--input', '-i', default="verl_step_records/r2egym-7b-agent-r2e_swe_verified_user-0704-magic-2025-07-07-09-35-20/step-1.pkl", help='输入pkl文件路径')
    parser.add_argument('--preview', '-p', type=int, default=2, help='预览样本数量 (默认: 2)')
    parser.add_argument('--model', '-m', default="Qwen/Qwen2.5-7B-Instruct", help='tokenizer模型名称')
    
    args = parser.parse_args()
    
    # 自动生成输出文件路径：同目录同名，扩展名为.jsonl
    input_path = args.input
    input_dir = os.path.dirname(input_path)
    input_base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(input_dir, input_base + ".jsonl")
    output_path = output_path.replace("verl_step_records", "verl_step_records_jsonl")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"输入文件: {args.input}")
    print(f"输出文件: {output_path}")
    print(f"使用模型: {args.model}")
    print("=" * 50)
    
    # 加载pkl文件
    try:
        with open(args.input, "rb") as f:
            dummy_data = pickle.load(f)
        # print("成功加载pkl文件")
        # print(dummy_data)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {args.input}")
        return
    except Exception as e:
        print(f"错误: 加载文件失败 - {e}")
        return
    
    # 加载tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print("成功加载tokenizer")
    except Exception as e:
        print(f"错误: 加载tokenizer失败 - {e}")
        return
    
    # 获取batch数据
    batch_data = dummy_data.batch
    non_tensor_batch_data = dummy_data.non_tensor_batch
    input_ids = batch_data['input_ids']
    
    print(f"Input_ids shape: {input_ids.shape}")
    print(f"数据包含 {input_ids.shape[0]} 个样本")
    
    # 解码并保存为jsonl格式
    output_data = []
    
    for i in range(input_ids.shape[0]):
        # 直接解码input_ids，去除padding token
        tokens = input_ids[i]
        tokens = tokens[tokens != tokenizer.pad_token_id]
        
        # 直接解码，让tokenizer处理im_start等特殊token
        decoded_text = tokenizer.decode(tokens, skip_special_tokens=False)

        # ---------------- 计算 decoded_text_with_lossmask ----------------
        # loss_mask 与 input_ids 形状一致（1 表示要算 loss，0 表示忽略）
        # print(f"input_ids shape: {input_ids[i].shape}")
        # print(f"loss_mask shape: {batch_data['loss_mask'][i].shape}")
        # print(f"input_ids: {input_ids[i]}")
        # print(f"loss_mask: {batch_data['loss_mask'][i]}")
        # print(f"attention_mask: {batch_data['attention_mask'][i]}")
        # print(f"response_mask: {batch_data['response_mask'][i]}")
        # print(f"len(tokens): {len(tokens)}")
        # print(f"sum(loss_mask): {sum(batch_data['loss_mask'][i])}")
        # print(f"sum(attention_mask): {sum(batch_data['attention_mask'][i])}")
        # print(f"sum(response_mask): {sum(batch_data['response_mask'][i])}")
        # print(f"len(input_ids): {len(input_ids[i])}")
        # print(f"sum(loss_mask or attention_mask or response_mask): {sum()}")
        # exit(1)
        loss_mask = batch_data['loss_mask'][i]
        # loss_mask = batch_data['response_mask'][i]

        # 先把 pad 裁掉，保持与 tokens 对齐
        # loss_mask = loss_mask[: len(tokens)]

        # # ---------------- 计算 decoded_text_with_lossmask ----------------
        # loss_mask = batch_data['loss_mask'][i][: len(tokens)]          # 裁掉 pad

        pieces = []
        for tid, m in zip(input_ids[i].tolist(), loss_mask.tolist()):
            if tid == tokenizer.pad_token_id:
                continue
            # 单 token decode，保证空格 / 特殊 token 不丢
            piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
            if m == 0:
                piece = f"#"
            pieces.append(piece)

        decoded_text_with_lossmask = "".join(pieces)
        # -----------------------------------------------------------------

        # 解析chat messages
        chat_messages = parse_chat_messages(decoded_text)
        
        # 构建数据记录
        print(non_tensor_batch_data.keys())
        chat_record = {
            "id": i,
            "messages": chat_messages,
            "raw_text": decoded_text,
            "raw_text_with_lossmask": decoded_text_with_lossmask,
            "last_obs": non_tensor_batch_data['last_obs'][i],
            "extra_info": non_tensor_batch_data['extra_info'][i],
        }
        
        output_data.append(chat_record)
        
        # 打印预览样本
        # if i < args.preview:
        #     print(f"\n样本 {i+1}:")
        #     for msg in chat_messages:
        #         print(f"Role: {msg['role']}")
        #         print(f"Message: {msg['message'][:200]}...")
        #         print("-" * 30)
    
    # 保存为jsonl文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"\n成功解码 {len(output_data)} 条聊天记录并保存到 {output_path}")
        print("文件格式: JSONL (每行一个JSON对象)")
        print("每个记录包含: id, messages (role + message 格式)")
        
    except Exception as e:
        print(f"错误: 保存文件失败 - {e}")

if __name__ == "__main__":
    main()

