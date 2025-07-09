import asyncio
import aiohttp
import time
import heapq
import torch
from tqdm.asyncio import tqdm
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, logger, DictConfig
from openai.types import Completion
from openai import AsyncOpenAI
from typing import Union, List, Dict, Any, Iterable
from pathlib import Path
from verl.protocol import DataProto
from verl_tool.llm_agent import AgentActorManager, AgentActorConfig
import os
import json
import pickle

class VerlToolChatCompletionScheduler(ChatCompletionScheduler):
    """A chat completion scheduler for verl-tool, which is a wrapper around the ChatCompletionScheduler."""

    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, server_addresses)

        self.agent_config = AgentActorConfig()
        # for key in get(self.config.agent_config
        rollout_config = config.actor_rollout_ref
        for key in getattr(rollout_config, 'agent', {}).keys():
            if key in self.agent_config.__dict__.keys():
                setattr(self.agent_config, key, rollout_config.agent[key])
        setattr(self.agent_config, 'n', rollout_config.rollout.n)
        setattr(self.agent_config, 'max_model_len', rollout_config.rollout.max_model_len)
        print(f"AgentAsyncActorRolloutRefWorker: {self.agent_config}")
        self.model_path = rollout_config.model.path
        self.agent_config.rollout_mode = "async"
        self.agent_actor_manager = AgentActorManager(self.model_path, self, self.agent_config)
        self.max_model_len = self.agent_actor_manager.max_model_len
        self.max_response_length = self.agent_config.max_response_length
        self.max_concurrent_trajectories = self.agent_config.max_concurrent_trajectories

        self.tokenizer = self.agent_actor_manager.tokenizer
        print(f"AgentActorManager initialized with config: {self.agent_config}")
    
    async def _completions_openai(self, address: str, **complete_request) -> Completion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        return await client.completions.create(**complete_request)

    async def _completions_aiohttp(self, address: str, **complete_request) -> Completion:
        try:
            extra_body = complete_request.pop("extra_body", {})
            complete_request.update(extra_body or {})
            extra_headers = complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=complete_request,
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    logger.error(f"Request failed with status address: {address}; headers: {extra_headers}; request: {complete_request}")
                    raise ValueError(f"Request failed with status {data['code']}: {data}; request: {complete_request}")
                return Completion(**data)
        finally:
            await session.close()

    async def _submit_completions(
        self,
        prompt: Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None],
        request_id: str,
        info: Dict[str, Any],
    ):
        """Submit chat completion request, wait request finish and do callback."""
        if not request_id:
            raise ValueError("request_id must be provided for chat completion requests.")
        
        # 保存原始的 request_id，用于路由选择
        original_raw_request_id = request_id.removeprefix("chatcmpl-")
        
        # 重试逻辑
        max_retries = 4
        timeout = 180  # seconds
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # ──────────────── ① 生成唯一 request_id ────────────────
                if attempt == 0:
                    raw_request_id = original_raw_request_id
                    request_id = f"{raw_request_id}"
                else:
                    # 重试时生成新的 raw_request_id
                    raw_request_id = f"{original_raw_request_id}_retry{attempt}"
                    request_id = f"{raw_request_id}_{int(time.time()*1e6)}"
                
                # request_id = f"{raw_request_id}_{int(time.time()*1e6)}"

                # ──────────────── ② 记录 old→new 映射 ────────────────
                try:
                    log_dir = Path("/minimax-dialogue/users/ruobai/rl_r2e/attempt_logs")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with (log_dir / "request_id_map.jsonl").open("a") as f:
                        f.write(json.dumps({
                            "ts": time.time(),
                            "old_id": raw_request_id,
                            "new_id": request_id,
                            "attempt": attempt
                        }) + "\n")
                except Exception as _e:
                    logger.warning(f"failed to log request_id map: {_e}")

                # ──────────────── ③ 选路由 ────────────────
                # 只在第一次尝试时选择路由，重试时使用相同路由
                if attempt == 0:
                    if original_raw_request_id not in self.request_id_to_address:
                        address = self.weighted_addresses[0][1]
                        self.weighted_addresses[0][0] += 1
                        heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
                        self.request_id_to_address[original_raw_request_id] = address
                    assert original_raw_request_id in self.request_id_to_address
                    address = self.request_id_to_address[original_raw_request_id]
                
                # use new request_id to avoid duplicate request_id problem
                self.request_id_to_address[request_id] = address
                
                openai_completion_allowed_keys = [
                    "model", "prompt", "best_of", "echo", "frequency_penalty",
                    "logit_bias", "logprobs", "max_tokens", "n", "presence_penalty",
                    "seed", "stop", "stream", "stream_options", "suffix", "temperature", "top_p", "user",
                    "extra_headers", "extra_query", "extra_body", "timeout"
                ]
                sampling_params = {k: v for k, v in info["__sampling_params__"].items() if k in openai_completion_allowed_keys}
                extra_body = {k: v for k, v in info["__sampling_params__"].items() if k not in openai_completion_allowed_keys}
                completion, exception = None, None
                sampling_params["max_tokens"] = 1536

                if "max_tokens" in sampling_params:
                    prompt_len = len(prompt)
                    if prompt_len + sampling_params["max_tokens"] > self.max_model_len:
                        sampling_params["max_tokens"] = self.max_model_len - prompt_len
                        logger.debug(f"Adjusted max_tokens to {sampling_params['max_tokens']} for prompt length {prompt_len} and max model length {self.max_model_len}.")
                    if sampling_params["max_tokens"] <= 0:
                        raise ValueError(f"max_tokens {sampling_params['max_tokens']} is too small for prompt length {prompt_len} and max model length {self.max_model_len}.")
                
                # ──────────────── ④ 记录完整请求 ────────────────
                try:
                    log_dir = Path("/minimax-dialogue/users/ruobai/rl_r2e/attempt_logs")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with (log_dir / "http_requests_id_log.jsonl").open("a") as f:
                        f.write(json.dumps({
                            "ts": time.time(),
                            "request_id": request_id,
                            "address": address,
                            "prompt_len": len(prompt) if hasattr(prompt, "__len__") else None,
                            "sampling_params": sampling_params,
                            "attempt": attempt
                        }) + "\n")
                except Exception as _e:
                    logger.warning(f"failed to log http request: {_e}")
                
                # 使用 asyncio.wait_for 实现超时控制
                completion = await asyncio.wait_for(
                    self._completions_aiohttp(
                        address,
                        prompt=prompt,
                        extra_body=extra_body,
                        extra_headers={"x-request-id": request_id},
                        **sampling_params,
                    ),
                    timeout=timeout
                )
                
                # 如果成功，清理路由地址并返回结果
                if attempt == max_retries - 1 or completion is not None:
                    self.request_id_to_address.pop(original_raw_request_id, None)
                
                info["__depth__"] -= 1
                if info["__depth__"] == 0:
                    info["__done__"].set()
                
                return completion.choices[0].text
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Request timeout after {timeout}s for attempt {attempt + 1}/{max_retries}, request_id: {request_id}")
                # 清理当前 request_id
                self.request_id_to_address.pop(request_id, None)
                if attempt < max_retries - 1:
                    timeout = timeout * 2
                    continue
                else:
                    # 最后一次重试也失败了，清理原始 request_id
                    self.request_id_to_address.pop(original_raw_request_id, None)
                    info["__depth__"] -= 1
                    if info["__depth__"] == 0:
                        info["__done__"].set()
                    raise TimeoutError(f"Request failed after {max_retries} attempts due to timeout") from e
                    
            except Exception as e:
                # 其他异常，记录并继续重试
                last_exception = e
                logger.exception(f"Request failed for attempt {attempt + 1}/{max_retries}, request_id: {request_id}, error: {e}")
                # 清理当前 request_id
                self.request_id_to_address.pop(request_id, None)
                if attempt < max_retries - 1:
                    continue
                else:
                    # 最后一次重试也失败了，清理原始 request_id
                    self.request_id_to_address.pop(original_raw_request_id, None)
                    info["__depth__"] -= 1
                    if info["__depth__"] == 0:
                        info["__done__"].set()
                    raise e
        
        # 不应该到达这里，但为了安全起见
        info["__depth__"] -= 1
        if info["__depth__"] == 0:
            info["__done__"].set()
        raise RuntimeError(f"Request failed after {max_retries} attempts, last error: {last_exception}")

    def simple_postprocess(self, batch: DataProto, responses: List[str]) -> DataProto:
        prompt_ids = batch.batch["input_ids"]
        prompt_attention_mask = batch.batch["attention_mask"]
        responses = self.tokenizer(responses, return_tensors="pt", padding="max_length", padding_side="right", max_length=self.max_response_length, truncation=True)

        input_ids = torch.cat([prompt_ids, responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch.batch['prompts'] = prompt_ids
        batch.batch['input_ids'] = input_ids
        batch.batch['attention_mask'] = attention_mask
        batch.batch['position_ids'] = position_ids
        batch.batch['responses'] = responses["input_ids"]
        batch.batch['response_mask'] = responses["attention_mask"]
        return batch
    
    async def simple_generate_sequences(
        self, batch: DataProto, **kwargs
    ) -> DataProto:
        # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_output_batch_0.pkl", "wb") as f:
        #     pickle.dump(batch, f)
        t_start = time.time()
        kwargs.update({
            "model": self.model_name,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        })
        to_remove_keys = ["max_new_tokens", "detokenize"]
        for key in to_remove_keys:
            if key in kwargs:
                kwargs.pop(key)

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        tasks = []
        for batch_index in range(len(batch)):
            prompt = batch.non_tensor_batch["raw_prompt_ids"][batch_index]
            prompt = list(prompt)
            request_id = batch.non_tensor_batch["traj_ids"][batch_index]
            info = {
                "__sampling_params__": kwargs,
                "__depth__": 1,
                "__done__": asyncio.Event(),
            }
            tasks.append(
                asyncio.create_task(
                    self._submit_completions(
                        prompt=prompt,
                        request_id=request_id,
                        info=info,
                    )
                )
            )
        responses = await tqdm.gather(*tasks, total=len(tasks), desc="Simple generating sequences", disable=len(tasks) < 10)
        # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_output_batch1.pkl", "wb") as f:
        #     pickle.dump(batch, f)
        output_batch = self.simple_postprocess(batch, responses)
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_output_batch2.pkl", "wb") as f:
        #     pickle.dump(output_batch, f)
        return output_batch

    async def generate_sequences(self, batch: DataProto, **kwargs) -> DataProto:
        # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_generate_sequences_output_batch_1.pkl", "wb") as f:
        #     pickle.dump(batch, f)
        logger.info("[VerlToolChatCompletionScheduler] generate_sequences start")
        t_start = time.time()
        kwargs.update({
            "model": self.model_name,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        })

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature
        repeated_batch = self.agent_actor_manager.repeat_inputs_by_n(batch)
        # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_generate_sequences_output_repeated_batch_1.pkl", "wb") as f:
        #     pickle.dump(repeated_batch, f)
        repeated_chunk_batch = repeated_batch.chunk(len(repeated_batch))
        # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_generate_sequences_output_chunk_batch_1.pkl", "wb") as f:
        #     pickle.dump(repeated_chunk_batch[-1], f)
        # repeated_batch = [repeated_batch] # for debug
        logger.info(f"[VerlToolChatCompletionScheduler] generate_sequences number of chunks: {len(repeated_chunk_batch)}")
        tasks = []
        if self.max_concurrent_trajectories is None or self.max_concurrent_trajectories <= 0:
            self.max_concurrent_trajectories = 256
            logger.warning(f"[VerlToolChatCompletionScheduler] max_concurrent_trajectories is not set, set to 256")

        if self.agent_config.enable_agent:
            if self.max_concurrent_trajectories is not None and self.max_concurrent_trajectories > 0:
                MAX_CONCURRENCY = self.max_concurrent_trajectories
                START_INTERVAL = getattr(self.config, "launch_interval_sec", 0.5)
                
                queue: asyncio.Queue = asyncio.Queue()
                sem = asyncio.Semaphore(MAX_CONCURRENCY)
                results = []
                
                async def producer() -> None:
                    """Feed chunks into the queue at a fixed rate."""
                    for chunk in repeated_chunk_batch:
                        await queue.put(chunk)
                        await asyncio.sleep(START_INTERVAL)
                    # poison pills to gracefully stop workers
                    for _ in range(MAX_CONCURRENCY):
                        await queue.put(None)
                        
                async def worker() -> None:
                    """Consume chunks and run LLM loop with concurrency guard."""
                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            break
                        async with sem:
                            # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_generate_sequences_output_chunk_batch_2.pkl", "wb") as f:
                            #     pickle.dump(chunk, f)
                            res = await self.agent_actor_manager.run_llm_loop_async(chunk, **kwargs)
                            results.append(res)
                
                # launch producer + N workers
                await tqdm.gather(
                    producer(),
                    *[asyncio.create_task(worker()) for _ in range(MAX_CONCURRENCY)],
                    desc="Starting producer-consumer pattern"
                )
                
                gen_outputs = results
            else:
                for batch_index in range(len(repeated_chunk_batch)):
                    tasks.append(
                        asyncio.create_task(
                            self.agent_actor_manager.run_llm_loop_async(
                                repeated_chunk_batch[batch_index],
                                **kwargs
                            )
                        )
                    )
                # gen_outputs = await asyncio.gather(*tasks)
                gen_outputs = await tqdm.gather(*tasks, total=len(tasks), desc="Async Generating sequences")
            output_batch = DataProto.concat(gen_outputs)
        else:
            kwargs["max_tokens"] = self.max_response_length
            output_batch = await self.simple_generate_sequences(
                repeated_batch,
                **kwargs
            )
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        logger.info(f"[VerlToolChatCompletionScheduler] generate_sequences for {len(repeated_batch)} number of trajectories done, took", output_batch.meta_info["timing"]["generate_sequences"], "seconds")
        # with open("/minimax-dialogue/users/ruobai/rl_r2e/chat_scheduler_generate_sequences_output_batch_2.pkl", "wb") as f:
        #     pickle.dump(output_batch, f)
        return output_batch