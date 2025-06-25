import json
import torch
import pickle

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import os
import time
import asyncio
import regex as re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import numpy as np

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
        self.step = None

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


    def __call__(self, data: DataProto, return_dict=False):
        # 初始化record_dir（参考torl实现）
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"{self.run_id}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"r2eswe-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查last step index（参考torl实现）
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.pkl", file):
                        step_idx = int(file[:-len(".pkl")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.pkl", file):
                        step_idx = int(file[:-len(".pkl")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

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
        reward_extra_info = defaultdict(list)
        
        for i in range(input_ids.shape[0]):
            try:
                # 解码input_ids，去除padding token
                tokens = input_ids[i]
                tokens = tokens[tokens != self.tokenizer.pad_token_id]
                
                # 解码为raw text
                # raw_text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                data_item = data[i]
                last_obs_raw_text = data_item.non_tensor_batch["last_obs"]
                
                # 使用正则表达式查找<reward>reward_str</reward>
                reward_pattern = r'<reward>(.*?)</reward>'
                # reward_match = re.search(reward_pattern, raw_text, re.DOTALL)
                last_obs_raw_text = data_item.non_tensor_batch["last_obs"]
                reward_match = re.search(reward_pattern, last_obs_raw_text, re.DOTALL)
                print(f"last_obs_raw_text of {i}: {last_obs_raw_text}")
                
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
        
        # 为每个样本计算统计信息（参考torl的实现方式）
        for i in range(len(data)):
            data_item = data[i]
            
            # 计算当前样本的obs长度统计
            if 'obs_lengths' in data_item.non_tensor_batch:
                obs_lengths = data_item.non_tensor_batch['obs_lengths']
                # 过滤掉0值（padding）
                valid_obs_lengths = [length for length in obs_lengths if length > 0]
                if valid_obs_lengths:
                    reward_extra_info['average_obs_length'].append(float(np.mean(valid_obs_lengths)))
                    reward_extra_info['max_obs_length'].append(int(np.max(valid_obs_lengths)))
                else:
                    reward_extra_info['average_obs_length'].append(0.0)
                    reward_extra_info['max_obs_length'].append(0)
            else:
                reward_extra_info['average_obs_length'].append(0.0)
                reward_extra_info['max_obs_length'].append(0)
            
            # 计算当前样本的action长度统计
            if 'action_lengths' in data_item.non_tensor_batch:
                action_lengths = data_item.non_tensor_batch['action_lengths']
                # 过滤掉0值（padding）
                valid_action_lengths = [length for length in action_lengths if length > 0]
                if valid_action_lengths:
                    reward_extra_info['average_action_length'].append(float(np.mean(valid_action_lengths)))
                    reward_extra_info['max_action_length'].append(int(np.max(valid_action_lengths)))
                else:
                    reward_extra_info['average_action_length'].append(0.0)
                    reward_extra_info['max_action_length'].append(0)
                
                # 统计非0元素的个数作为action round
                action_round = sum(1 for length in action_lengths if length > 0)
                reward_extra_info['action_round'].append(action_round)
            else:
                reward_extra_info['average_action_length'].append(0.0)
                reward_extra_info['max_action_length'].append(0)
                reward_extra_info['action_round'].append(0)
        
        print( f"Extracted rewards: {rewards}")
        print(f"Extracted reports: {reports}")
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

        # print(f"Extracted rewards: {rewards}")
        print(f"Reward tensor shape: {reward_tensor.shape}")
        print(f"Statistics: avg_obs_len={reward_extra_info['average_obs_length']}, max_obs_len={reward_extra_info['max_obs_length']}, avg_action_len={reward_extra_info['average_action_length']}, max_action_len={reward_extra_info['max_action_length']}, action_rounds={reward_extra_info['action_round']}")

        # 保存完整的data为pkl文件
        # if save_record:
        if True: # Always save
            if self.num_examine == 1:
                temp_file = self.record_dir / f"step-val-{self.step}.pkl"
            else:
                temp_file = self.record_dir / f"step-{self.step}.pkl"
            self.step += 1
            
            # 保存完整的data对象为pkl
            with open(temp_file, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved complete data to {temp_file}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor



if __name__ == '__main__':
    import pickle

    # Load the saved data object from disk
    with open("verl_step_records/r2egym-7b-agent-r2e_sync_extra-baseline-0623-bs32-2025-06-25-05-35-58/step-1.pkl", "rb") as f:
        dummy_data = pickle.load(f)
    # print(dummy_data)

    # Instantiate the WikiRLRewardManager (you can pass in config if needed)
    reward_manager = R2ESWERewardManager()

    # Compute rewards for the loaded data
    rewards = reward_manager(dummy_data, return_dict=True)
    # print("Rewards:", rewards["reward_tensor"])
    # print("Reports:", rewards["reward_extra_info"])

