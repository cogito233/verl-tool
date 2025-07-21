"""
Multi-turn rollout implementation for per-action reward.
Based on verl-agent implementation with adaptations for current verl-tool architecture.
"""
import torch
import numpy as np
import uuid
from typing import Dict, List, Tuple, Any, Optional
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import asyncio
import json
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MultiTurnRolloutCollector:
    """
    Multi-turn rollout collector that implements per-action reward collection.
    
    This class collects trajectories through multi-step agent-environment interactions,
    where each step (action) gets its own reward and forms an independent training sample.
    """
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        """
        Initialize the MultiTurnRolloutCollector.
        
        Args:
            config: Configuration object containing rollout settings
            tokenizer: Tokenizer for text processing
        """
        self.config = config
        self.tokenizer = tokenizer
        
    def preprocess_single_sample(self, item: int, gen_batch: DataProto, obs: Dict) -> Dict:
        """
        Process a single observation sample for multi-turn rollout.
        
        Args:
            item: Sample index in the batch
            gen_batch: Batch data containing original prompts
            obs: Environment observation
            
        Returns:
            Dict containing processed input data
        """
        # Get the raw prompt from the original batch
        raw_prompt = gen_batch.non_tensor_batch.get('raw_prompt', [None])[item]
        data_source = gen_batch.non_tensor_batch.get('data_source', [None])[item]
        
        # Get observation text
        obs_text = obs.get('text', [None])[item] if obs.get('text') else None
        
        if obs_text is None:
            logger.warning(f"No text observation found for item {item}")
            obs_text = ""
        
        # Build input prompt for current step
        # This should include the task description and current observation
        if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
            # Use the original prompt as base
            chat_messages = raw_prompt.copy()
            # Update the last message with current observation
            if chat_messages and chat_messages[-1].get('role') == 'user':
                chat_messages[-1]['content'] = obs_text
            else:
                chat_messages.append({
                    "content": obs_text,
                    "role": "user"
                })
        else:
            # Create new chat structure
            chat_messages = [{
                "content": obs_text,
                "role": "user"
            }]
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize the prompt
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        
        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build return dict
        row_dict = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt': chat_messages,
            'anchor_obs': obs_text,  # For compatibility with GiGPO
            'index': item,
            'data_source': data_source,
        }
        
        return row_dict
    
    def preprocess_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        """
        Process a batch of observations for multi-turn rollout.
        
        Args:
            gen_batch: Batch data containing original prompts
            obs: Environment observations
            
        Returns:
            DataProto with processed batch data
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample
        for item in range(batch_size):
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs
            )
            processed_samples.append(processed)
        
        # Collate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )
        
        return new_batch
    
    def gather_step_data(self, batch: DataProto, step_rewards: List[float], 
                        step_dones: List[bool], step_infos: List[Dict],
                        traj_uids: List[str], step_idx: int) -> List[Dict]:
        """
        Gather data for a single step across all trajectories.
        
        Args:
            batch: Batch data for current step
            step_rewards: Rewards for current step
            step_dones: Done flags for current step
            step_infos: Info dicts for current step
            traj_uids: Trajectory UIDs
            step_idx: Current step index
            
        Returns:
            List of step data dictionaries
        """
        step_data_list = []
        
        batch_size = len(batch.batch['input_ids'])
        for i in range(batch_size):
            # Convert batch data to individual dict
            step_data = {}
            
            # Add tensor data
            for key, tensor in batch.batch.items():
                if isinstance(tensor, torch.Tensor):
                    step_data[key] = tensor[i]
            
            # Add non-tensor data
            for key, array in batch.non_tensor_batch.items():
                step_data[key] = array[i]
            
            # Add step-specific data
            step_data.update({
                'step_reward': step_rewards[i],
                'step_done': step_dones[i],
                'step_info': step_infos[i],
                'traj_uid': traj_uids[i],
                'step_idx': step_idx,
                'timestamp': time.time(),
            })
            
            step_data_list.append(step_data)
        
        return step_data_list
    
    async def multi_turn_rollout(self, gen_batch: DataProto, actor_rollout_wg,
                                tool_server, max_steps: int = 20) -> List[Dict]:
        """
        Execute multi-turn rollout with per-action rewards.
        
        Args:
            gen_batch: Initial batch data
            actor_rollout_wg: Actor rollout worker group
            tool_server: Tool server for environment interactions
            max_steps: Maximum number of steps per trajectory
            
        Returns:
            List of step data dictionaries for training
        """
        # Initialize trajectories
        batch_size = len(gen_batch.batch['input_ids'])
        traj_uids = [str(uuid.uuid4()) for _ in range(batch_size)]
        
        # Initialize observations - start with getting initial observations
        trajectory_ids = [f"traj_{i}_{traj_uids[i]}" for i in range(batch_size)]
        
        # Start environments and get initial observations
        actions = [""] * batch_size  # Empty action for initialization
        extra_fields = []
        
        for i in range(batch_size):
            # Get original data for environment initialization
            data_source = gen_batch.non_tensor_batch.get('data_source', [None])[i]
            extra_fields.append({
                'ds': data_source,
                'trajectory_id': trajectory_ids[i],
                'is_last_step': False
            })
        
        # Get initial observations
        obs_texts, dones, valids = await tool_server.aget_observations(
            trajectory_ids, actions, extra_fields
        )
        
        # Convert to obs dict format
        obs = {'text': obs_texts}
        
        # Initialize trajectory collection
        all_step_data = []
        is_done = np.array(dones, dtype=bool)
        
        # Multi-turn rollout loop
        for step_idx in range(max_steps):
            # Check if all trajectories are done
            if is_done.all():
                break
                
            # Get active trajectories
            active_mask = ~is_done
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) == 0:
                break
            
            # Preprocess observations for active trajectories
            active_gen_batch = gen_batch.select(active_indices)
            active_obs = {'text': [obs['text'][i] for i in active_indices]}
            
            # Process batch for current step
            batch = self.preprocess_batch(gen_batch=active_gen_batch, obs=active_obs)
            
            # Prepare batch for generation
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt"]
            
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            
            # Generate responses
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            
            # Combine input and output
            batch = batch.union(batch_output)
            
            # Decode actions
            text_actions = self.tokenizer.batch_decode(
                batch.batch['responses'], skip_special_tokens=True
            )
            
            # Prepare for environment step
            active_trajectory_ids = [trajectory_ids[i] for i in active_indices]
            active_extra_fields = []
            
            for i, global_i in enumerate(active_indices):
                active_extra_fields.append({
                    'ds': gen_batch.non_tensor_batch.get('data_source', [None])[global_i],
                    'trajectory_id': active_trajectory_ids[i],
                    'is_last_step': step_idx == max_steps - 1
                })
            
            # Step environment
            next_obs_texts, step_dones, step_valids = await tool_server.aget_observations(
                active_trajectory_ids, text_actions, active_extra_fields
            )
            
            # Extract rewards from observations
            step_rewards = []
            for obs_text in next_obs_texts:
                reward = self._extract_reward_from_obs(obs_text)
                step_rewards.append(reward)
            
            # Create step info
            step_infos = []
            for i in range(len(active_indices)):
                step_infos.append({
                    'action': text_actions[i],
                    'observation': next_obs_texts[i],
                    'valid': step_valids[i]
                })
            
            # Gather step data
            active_traj_uids = [traj_uids[i] for i in active_indices]
            step_data_list = self.gather_step_data(
                batch=batch,
                step_rewards=step_rewards,
                step_dones=step_dones,
                step_infos=step_infos,
                traj_uids=active_traj_uids,
                step_idx=step_idx
            )
            
            # Add to overall collection
            all_step_data.extend(step_data_list)
            
            # Update done status for all trajectories
            for i, global_i in enumerate(active_indices):
                is_done[global_i] = step_dones[i]
            
            # Update observations for next step
            for i, global_i in enumerate(active_indices):
                obs['text'][global_i] = next_obs_texts[i]
        
        return all_step_data
    
    def _extract_reward_from_obs(self, obs_text: str) -> float:
        """
        Extract reward from observation text.
        
        Args:
            obs_text: Observation text that may contain reward tags
            
        Returns:
            Extracted reward value
        """
        import re
        
        # Look for <reward>value</reward> pattern
        reward_pattern = r'<reward>(.*?)</reward>'
        reward_match = re.search(reward_pattern, obs_text, re.DOTALL)
        
        if reward_match:
            reward_str = reward_match.group(1).strip()
            try:
                return float(reward_str)
            except ValueError:
                logger.warning(f"Invalid reward format: '{reward_str}', using 0.0")
                return 0.0
        
        # If no reward found, return 0.0
        # This is normal for intermediate steps
        return 0.0
    
    def create_training_batch(self, step_data_list: List[Dict]) -> DataProto:
        """
        Create training batch from collected step data.
        
        Args:
            step_data_list: List of step data dictionaries
            
        Returns:
            DataProto ready for training
        """
        # Process each step data for training
        processed_data = []
        
        for step_data in step_data_list:
            # Create training sample from step data
            training_sample = {}
            
            # Copy tensor data
            for key, value in step_data.items():
                if isinstance(value, torch.Tensor):
                    training_sample[key] = value
                elif key in ['step_reward', 'step_done', 'step_info', 'traj_uid', 
                           'step_idx', 'timestamp', 'raw_prompt', 'anchor_obs']:
                    # These are handled separately in non_tensor_batch
                    continue
                else:
                    training_sample[key] = value
            
            # Create reward tensor - put reward at the last token position
            if 'responses' in training_sample:
                responses = training_sample['responses']
                reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
                
                # Find last non-pad token position
                pad_id = self.tokenizer.pad_token_id
                valid_mask = responses != pad_id
                
                if valid_mask.any():
                    last_token_idx = int(valid_mask.nonzero(as_tuple=False)[-1])
                    reward_tensor[last_token_idx] = step_data['step_reward']
                
                training_sample['rewards'] = reward_tensor
            
            # Add non-tensor data
            training_sample.update({
                'traj_uid': step_data['traj_uid'],
                'step_idx': step_data['step_idx'],
                'raw_prompt': step_data.get('raw_prompt', []),
                'anchor_obs': step_data.get('anchor_obs', ''),
                'last_obs': step_data['step_info']['observation'],
            })
            
            processed_data.append(training_sample)
        
        # Collate into batch
        batch = collate_fn(processed_data)
        
        # Create DataProto
        training_batch = DataProto.from_single_dict(
            data=batch,
            meta_info={'rollout_type': 'multi_turn', 'num_steps': len(step_data_list)}
        )
        
        return training_batch