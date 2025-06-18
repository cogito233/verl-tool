import nltk
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

def clean_text(text):
    # 删除控制字符 & 非打印字符
    return re.sub(r'[\x00-\x1F\x7F-\x9F\u200b-\u200f\u2028-\u202f\u2060-\u206f]', '', text)

class R2ESWERewardManager:
    """
    Reward Manager for the WikiRL dataset.

    This class computes a combined reward for each predicted answer by comparing it with
    the ground truth answers. The final reward is a weighted combination of a fuzzy matching
    score and a structure score.
    # """
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None) -> None:
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
        # self.fuzzy_weight = 0.7
        # self.structure_weight = 0.3

    def reward_single(self, output, ds):
        test_spec = make_test_spec(ds)
        out, _ = self.runtime.run_tests()
        eval_status_map, found = self.runtime.get_logs_eval(test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=get_eval_type(self.test_spec)
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        return int(success), report

    def parse_last_response(self, data: DataProto):
        last_responses = []
        for i in range(len(data)):
            pass
        return last_responses

    def __call__(self, data: DataProto, return_dict=False):
        print("")
        print(data)
        print(len(data))
        import pickle
        with open("data_stub_new_qwq.pkl", "wb") as f:
            pickle.dump(data, f)
        exit(1)
        dses = [datapoint["extra_fields"] for datapoint in data.non_tensor_batch]
        print(test_spec)
        outputs = self.parse_last_response(data)
        rewards, reports = [], []
        for output, ds in zip(outputs, dses):
            success, report = self.reward_single(output, ds)
            rewards.append(success)
            reports.append(report)
        # exit(1)
        if return_dict:
            return {
                "reward_tensor": rewards,
                "reward_extra_info": reports,
            }
        else:
            return rewards



if __name__ == '__main__':
    import pickle

    # Load the saved data object from disk
    with open("data_stub_r2e.pkl", "rb") as f:
        dummy_data = pickle.load(f)

    # Instantiate the WikiRLRewardManager (you can pass in config if needed)
    reward_manager = R2ESWERewardManager()

    # Compute rewards for the loaded data
    rewards = reward_manager(dummy_data)
    print("Rewards:", rewards)


"""
(TaskRunner pid=2019847) ==== Call WikiRLRewardManager ====
(TaskRunner pid=2019847) DataProto(batch=TensorDict(
(TaskRunner pid=2019847)     fields={
(TaskRunner pid=2019847)         attention_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         info_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         input_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         old_log_probs: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         position_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         prompts: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         ref_log_prob: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         responses: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         responses_with_info_mask: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False)},
(TaskRunner pid=2019847)     batch_size=torch.Size([4]),
(TaskRunner pid=2019847)     device=None,
(TaskRunner pid=2019847)     is_shared=False), non_tensor_batch={'data_source': array(['wiki_qa', 'wiki_qa', 'wiki_qa', 'wiki_qa'], dtype=object), 'ability': array(['wiki', 'wiki', 'wiki', 'wiki'], dtype=object), 'reward_model': array([{'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'}],
(TaskRunner pid=2019847)       dtype=object), 'index': array([0, 0, 0, 0], dtype=object), 'uid': array(['ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b'], dtype=object)}, meta_info={'turns_stats': [4, 4], 'active_mask': [True, True], 'valid_action_stats': [4, 4], 'global_token_num': [5541, 5541, 3697, 5542], 'temperature': 0.9})
"""
