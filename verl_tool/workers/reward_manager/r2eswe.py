import json
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import os
import time
import asyncio
import regex as re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from r2egym.agenthub.trajectory.swebench_utils import make_test_spec
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from verl.workers.reward_manager import register

def clean_text(text):
    # 删除控制字符 & 非打印字符
    return re.sub(r'[\x00-\x1F\x7F-\x9F\u200b-\u200f\u2028-\u202f\u2060-\u206f]', '', text)

@register("r2eswe")
class R2ESWERewardManager:
    """
    Reward Manager for the WikiRL dataset.

    This class computes a combined reward for each predicted answer by comparing it with
    the ground truth answers. The final reward is a weighted combination of a fuzzy matching
    score and a structure score.
    # """
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, reward_fn_key='data_source'):
        """
        Initialize the WikiRLRewardManager.

        Parameters:
        - fuzzy_weight: The weight applied to the fuzzy matching score.
        - structure_weight: The weight applied to the structure score.
        """
        if tokenizer is None:
            # Simply use QWen2.5-7B tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key  # 新增，兼容主流程

    # def reward_single(self, output, ds): # Aborted
    #     test_spec = make_test_spec(ds)
    #     out, _ = self.runtime.run_tests()
    #     eval_status_map, found = self.runtime.get_logs_eval(test_spec, out)
    #     eval_ref = {
    #         KEY_INSTANCE_ID: self.test_spec.instance_id,
    #         FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
    #         PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
    #     }
    #     report = get_eval_tests_report(
    #         eval_status_map, eval_ref, eval_type=get_eval_type(self.test_spec)
    #     )
    #     success = get_resolution_status(report) == ResolvedStatus.FULL.value
    #     return int(success), report

    def parse_last_response(self, data: DataProto):
        last_responses = []
        for i in range(len(data)):
            pass
        return last_responses

    def __call__(self, data: DataProto, return_dict=False):
        # print("")
        # print(data)
        # print(len(data))
        # import pickle
        # with open("data_stub_withReward.pkl", "wb") as f:
        #     pickle.dump(data, f)
        # exit(1)
        import re

        print(f"Processing {len(data)} samples for reward extraction")
        
        # 获取input_ids
        batch_data = data.batch
        input_ids = batch_data['input_ids']
        
        rewards = []
        reports = []
        
        for i in range(input_ids.shape[0]):
            try:
                # 解码input_ids，去除padding token
                tokens = input_ids[i]
                tokens = tokens[tokens != self.tokenizer.pad_token_id]
                
                # 解码为raw text
                raw_text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                
                # 使用正则表达式查找<reward>reward_str</reward>
                reward_pattern = r'<reward>(.*?)</reward>'
                reward_match = re.search(reward_pattern, raw_text, re.DOTALL)
                
                if reward_match:
                    reward_str = reward_match.group(1).strip()
                    try:
                        # 尝试转换为float
                        reward_value = float(reward_str)
                        rewards.append(reward_value)
                        reports.append(f"Found reward: {reward_value}")
                    except ValueError:
                        # 如果无法转换为float，设为0
                        rewards.append(0.0)
                        reports.append(f"Invalid reward format: '{reward_str}', set to 0.0")
                else:
                    # 如果找不到reward标签，设为0
                    rewards.append(0.0)
                    reports.append("No reward tag found, set to 0.0")
                    
            except Exception as e:
                # 处理任何解码错误
                rewards.append(0.0)
                reports.append(f"Error processing sample {i}: {str(e)}")
        
        # print( f"Extracted rewards: {rewards}")
        # print(f"Extracted reports: {reports}")
        # responses_id = data.batch["responses"]
        # reward_tensor = torch.torch.zeros_like(responses_id, dtype=torch.float32)   
        # print(f"Reward tensor shape: {reward_tensor.shape}")
        # print(f"Reward tensor: {reward_tensor}")
        # exit(1)
        responses_id = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses_id, dtype=torch.float32)

        pad_id = self.tokenizer.pad_token_id
        for i, reward_val in enumerate(rewards):
            # find last non-pad token position in this response sequence
            seq = responses_id[i]
            valid_mask = seq != pad_id
            if valid_mask.any():
                last_token_idx = int(valid_mask.nonzero(as_tuple=False)[-1])
                reward_tensor[i, last_token_idx] = reward_val
            else:
                # rare edge case: all pads – keep zeros
                reports[i] += " | response sequence empty"

        print(f"Extracted rewards: {rewards}")
        print(f"Reward tensor shape: {reward_tensor.shape}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                # "reward_extra_info": {"reports": reports},
                "reward_extra_info": None,
            }
        else:
            return reward_tensor



if __name__ == '__main__':
    import pickle

    # Load the saved data object from disk
    with open("data_stub_withReward.pkl", "rb") as f:
        dummy_data = pickle.load(f)
    print(dummy_data)

    # Instantiate the WikiRLRewardManager (you can pass in config if needed)
    reward_manager = R2ESWERewardManager()

    # Compute rewards for the loaded data
    rewards = reward_manager(dummy_data, return_dict=True)
    print("Rewards:", rewards["reward_tensor"])
    print("Reports:", rewards["reward_extra_info"])

