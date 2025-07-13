import ray
import re
import json
import time
import asyncio  
import contextlib
import io
import sys 
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError, as_completed

from .base import BaseTool, register_tool
from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from r2egym.agenthub.tools import (
    search_tool,
    file_editor,
    bash_execute_tool,
    finish_tool,
)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Temporarily redirect both stdout and stderr to /dev/null-like objects."""
    new_target = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_target, new_target
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err

@ray.remote
class R2EEnvActor:
    def __init__(self, ds: Dict[str, Any], command_files: Optional[list] = None):
        """
        Initialize R2E environment actor
        
        Args:
            ds: Dataset entry containing docker_image and other info
            command_files: Command files to add to environment
        """
        # print("Checkpoint 1, start init R2EEnvActor")
        self.ds = ds
        self.command_files = command_files or []
                
        # Initialize environment directly
        env_args = EnvArgs(ds=ds)
        self.env = RepoEnv(env_args)
        # print("Checkpoint 2, init RepoEnv")
        self.env.reset()
        
        # Add command files if provided
        if self.command_files:
            self.env.add_commands(self.command_files)
            print("add command files")

        # print("Checkpoint 3, init RepoEnv done")
        

    def start_env(self) -> str:
        """
        Start environment and return initial observation
        
        Returns:
            Initial problem statement
        """
        # print("Checkpoint 4, start get_task_instruction")
        # Get initial problem statement
        problem_statement = self.env.runtime.get_task_instruction()
        user_prompt = f"""Consider the following github issue:
  <github_issue>
  {problem_statement}
  </github_issue>

  Can you help me implement the necessary changes to the repository to fix the <github_issue>?
  I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
  Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

  IMPORTANT TIP:
  Follow these steps to resolve the issue:
  1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
  2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
  3. Edit the sourcecode of the repo to resolve the issue
  4. Rerun your reproduce script and confirm that the error is fixed!
  5. Think about edgecases and make sure your fix handles them as well
  6. When viewing large files, use specific line-ranges, usually within 50 to 100 lines) as required
  7. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files. 
  """
        return user_prompt

    def step_env(self, action_str: str) -> Tuple[str, bool, bool]:
        """
        Execute one action step
        
        Args:
            action_str: Action string from LLM (XML format)
            
        Returns:
            Tuple[observation, done, valid]:
            - observation: Environment observation as string
            - done: Whether task is complete
            - valid: Whether action was valid
        """
        # print("Checkpoint 5, start step_env")
        if "<compute_reward>sandbox_r2e</compute_reward>" in action_str:
            reward, valid, test_output = self.reward_env("<compute_reward>sandbox_r2e</compute_reward>")
            reward_str = f"[no_obs] <reward>{reward}</reward>" #<test_output>{test_output}</test_output>" # 暂时不返回test_output
            return reward_str, True, True

        try:
            # Parse action from string (similar to agent.py parse_response)
            action = Action.from_string(action_str)
            
            # Check if action parsing failed
            if action is None or not action.function_name:
                return "Failed to parse action. Invalid XML format or missing function name.", False, False
            
            # print("Checkpoint 6, start step_env, action: ", action)

            # Execute action (same pattern as agent.py)
            with suppress_stdout_stderr():
                obs, reward, done, info = self.env.step(action, timeout=120)
            # print("Checkpoint 7, start step_env done, obs: ", obs)

            # if done, then pad the reward into the observation
            if done:
                reward, valid, test_output = self.reward_env("<compute_reward>sandbox_r2e</compute_reward>")
                obs = str(obs)+f"<reward>{reward}</reward>" # Maybe Finished is Here
                # obs += f"<reward>{reward}</reward>"#<test_output>{test_output}</test_output>"
            
            # Return observation as string (following agent.py pattern)
            return str(obs), done, True
            
        except Exception as e:
            raise e
            # Handle exceptions like agent.py does - convert to observation string
            return str(e), False, False
    
    def reward_env(self, action_str: str) -> Tuple[float, bool, str]:
        """
        Reward the environment, return the reward, validity and test output
        """
        # print("Checkpoint 8, start reward_env")
        if action_str == "<compute_reward>sandbox_r2e</compute_reward>":
            with suppress_stdout_stderr():
                reward, test_output = self.env.runtime._calculate_reward(get_test_output=True)
            reward = float(reward)
            # print("Checkpoint 9, start reward_env done, reward: ", reward)
            return reward, True, test_output
        else:
            raise ValueError(f"Invalid reward action: {action_str}")

    def close_env(self):
        """关闭环境，带超时机制"""
        try:
            json_content = {
                "timestamp": time.time(),
                "job_name": getattr(self.env.runtime, 'job_name', 'unknown')
            }
            print(f"Deleting env, timestamp: {time.time()}, job_name: {json_content['job_name']}")
            
            # 使用 ray 的超时机制而不是自己实现
            with open("env_deleted.jsonl", "a") as f:
                f.write(json.dumps(json_content) + "\n")
            
            # 如果 env.close() 可能卡住，可以考虑跳过或使用简单的清理
            try:
                self.env.close()
            except Exception as e:
                print(f"Error in env.close(): {e}")
                
        except Exception as e:
            print(f"Error in close_env: {e}")

@register_tool
class SandboxR2ETool(BaseTool):
    """
    SandboxR2ETool uses Ray actors to manage R2E environment sessions.
    Each trajectory_id has a dedicated actor. It supports initial
    render (action=None) and step operations.
    """
    tool_type = "sandbox_r2e"

    def __init__(self, num_workers=512):
        super().__init__(num_workers)
        # Maps trajectory_id to Ray Actor
        self.env_actors = {}
        # Track creation order for cleanup
        self.actor_creation_order = []

    # -------------------------------------------------------------------------
    # BaseTool interface methods (some are no-ops here, but we must implement them)
    # -------------------------------------------------------------------------
    def get_usage_inst(self) -> str:
        """Return usage instructions."""
        return "SandboxR2ETool uses Ray actors to manage R2E environment sessions."

    def has_env(self, trajectory_id):
        return trajectory_id in self.env_actors

    def load_env(self, trajectory_id: str):
        """Return a live actor or `None` if the trajectory is unknown."""
        return self.env_actors.get(trajectory_id)

    def save_env(self, trajectory_id: str, actor):
        """Register / refresh an actor and update LRU ordering."""
        if self.env_actors.get(trajectory_id) is None:
            self.env_actors[trajectory_id] = actor
            self.actor_creation_order.append(trajectory_id)
            self._cleanup_actors_if_needed()
        else:
            # If it exists, check if it's the same actor, otherwise raise an error
            if self.env_actors[trajectory_id] != actor:
                raise RuntimeError(f"Actor with trajectory_id {trajectory_id} already exists.")
            if trajectory_id in self.actor_creation_order:
                self.actor_creation_order.remove(trajectory_id)
            self.actor_creation_order.append(trajectory_id)

    # ---------- async 版本 ----------
    async def asave_env(self, trajectory_id: str, actor):
        """Async register / refresh; awaits async cleanup."""
        if self.env_actors.get(trajectory_id) is None:
            self.env_actors[trajectory_id] = actor
            self.actor_creation_order.append(trajectory_id)
            await self._acleanup_actors_if_needed()
        else:
            if self.env_actors[trajectory_id] != actor:
                raise RuntimeError(
                    f"Actor with trajectory_id {trajectory_id} already exists."
                )
            if trajectory_id in self.actor_creation_order:
                self.actor_creation_order.remove(trajectory_id)
            self.actor_creation_order.append(trajectory_id)

    # ---- 兼容旧的同步调用路径 ----
    def delete_env(self, trajectory_id):
        """Sync wrapper kept for legacy code paths."""
        asyncio.run(self.adelete_env(trajectory_id))

    def parse_action(self, action):
        """
        检查action是否为有效的R2E动作格式。
        R2E动作应该是XML格式，如：<function=function_name>...</function>
        或者为空字符串（初始化）
        
        这个方法主要用于预检查，实际的解析由R2EEnvActor处理
        """
        if action == "" or action is None:  # 允许空动作用于初始化
            return action, True
        
        # 对于非空动作，我们采用宽松的策略
        # 只要包含基本的function标签就认为可能有效
        # 具体的解析和验证由环境Actor处理
        action_str = str(action).strip()
        
        if "<compute_reward>sandbox_r2e</compute_reward>" in action_str:
            return action, True
        
        # 基本格式检查
        if ("<function=" in action_str and "</function>" in action_str) or \
           ("function=" in action_str and "parameter=" in action_str):
            return action, True
        
        # 即使格式不标准，也给环境一个尝试的机会
        # 让具体的解析器来决定是否有效
        print(f"Action: {action}, maybe it is invalid")
        return action, True

    def _cleanup_actors_if_needed(self):
        """Remove oldest actors if count exceeds limit."""
        while len(self.env_actors) > 512:
            # 实际清理而不是抛出异常
            if not self.actor_creation_order:
                break
            oldest = self.actor_creation_order.pop(0)
            print(f"[INFO] Deleting actor {oldest} due to too many actors.")
            try:
                self.delete_env(oldest)
            except Exception as e:
                print(f"[ERROR] Failed to delete actor {oldest}: {e}")
                # 即使清理失败也要从字典中移除引用
                if oldest in self.env_actors:
                    del self.env_actors[oldest]

    async def _acleanup_actors_if_needed(self):
        """Remove oldest actors if count exceeds limit."""
        while len(self.env_actors) > 512:
            # 实际清理而不是抛出异常
            if not self.actor_creation_order:
                break
            oldest = self.actor_creation_order.pop(0)
            print(f"[INFO] Deleting actor {oldest} due to too many actors.")
            try:
                await self.adelete_env(oldest)
            except Exception as e:
                print(f"[ERROR] Failed to delete actor {oldest}: {e}")
                # 即使清理失败也要从字典中移除引用
                if oldest in self.env_actors:
                    del self.env_actors[oldest]

    async def aconduct_action(
        self, trajectory_id: str, action: str, extra_field: dict
    ):
        """完全异步，不阻塞事件循环。"""
        # print("Checkpoint 10, start aconduct_action")
        actor = self.load_env(trajectory_id)
        if actor is None:                        # 懒创建
            actor = await self._aspawn_actor(trajectory_id, extra_field)
            await self.asave_env(trajectory_id, actor)
        # print("Checkpoint 11, start save_env done")

        # === 把 Ray 调用也异步化 ===
        obj_ref = (
            actor.start_env.remote()
            if not action
            else actor.step_env.remote(action)
        )
        # print("Checkpoint 12, start obj_ref")
        try:
            # Ray≥2.10 支持 `await obj_ref`
            result = await asyncio.wait_for(obj_ref, timeout=300)
        except asyncio.TimeoutError:
            return "[TIMEOUT] (aconduct_action) <reward>0.0</reward>", True, True
        except Exception as e:
            return f"Error: {e}", False, False
        # print("Checkpoint 13, start obj_ref done")
        # 拆包
        if isinstance(result, tuple):
            obs, done, valid = result
        else:
            obs, done, valid = result, False, True
        # print("Checkpoint 14, start result done")
        # LRU 刷新
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)
        self.actor_creation_order.append(trajectory_id)
        # print("Checkpoint 15, start LRU done")          
        if not valid:
            obs = f"The action {action} is invalid, please retry, obs is {obs}"
        return obs, done, valid
    
    async def _aspawn_actor(self, trajectory_id: str, extra_field: dict):
        ds = extra_field.get("ds", extra_field)
        cmd_files = [
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/search.py"),
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/file_editor.py"),
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/execute_bash.py"),
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/finish.py")
            ]
        actor = R2EEnvActor.options(
            lifetime="detached"  # 或按需
        ).remote(ds, cmd_files)
        # 等待 actor.reset() 完成，确保就绪
        await (actor.ready.remote() if hasattr(actor, "ready") else asyncio.sleep(0))
        return actor
    
    async def adelete_env(self, trajectory_id):
        if trajectory_id in self.env_actors:
            actor = self.env_actors[trajectory_id]
            try:
                fut = actor.close_env.remote()
                try:
                    await asyncio.wait_for(fut, timeout=60)
                except asyncio.TimeoutError:
                    print(f"close_env timeout for trajectory_id: {trajectory_id}, forcing kill")
            except Exception as e:
                print(f"Error closing env for trajectory_id: {trajectory_id}: {e}")
            
            # 强制kill actor并清理引用
            try:
                ray.kill(actor, no_restart=True)
            except Exception as e:
                print(f"Error killing actor for trajectory_id: {trajectory_id}: {e}")
            
            # 无论如何都要清理引用
            if trajectory_id in self.env_actors:
                del self.env_actors[trajectory_id]
         
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)

    async def aget_observations(
        self, trajectory_ids, actions, extra_fields
    ):
        sem = asyncio.Semaphore(self.num_workers)

        async def _task(i):
            async with sem:
                act = ( "<compute_reward>sandbox_r2e</compute_reward>"
                        if extra_fields[i].get("is_last_step")
                        else actions[i] )
                try:
                    return i, *await self.aconduct_action(
                        trajectory_ids[i], act, extra_fields[i]
                    ), None
                except Exception as e:
                    # 异常情况下也要清理环境
                    try:
                        await self.adelete_env(trajectory_ids[i])
                    except Exception as cleanup_e:
                        print(f"[ERROR] Cleanup failed for {trajectory_ids[i]}: {cleanup_e}")
                    return i, "", False, False, e

        coros = [_task(i) for i in range(len(trajectory_ids))]
        results = await asyncio.gather(*coros, return_exceptions=False)

        # 初始化
        n = len(trajectory_ids)
        obs, dones, valids = [""] * n, [False] * n, [True] * n

        # 处理结果
        for i, o, d, v, err in results:
            obs[i], dones[i], valids[i] = o, d, v
            if err:
                print(f"[ERROR] trajectory_id={trajectory_ids[i]}: {err}")

        # 清理 last-step 或 done 的环境
        cleanups = [
            self.adelete_env(tid)
            for tid, done, extra in zip(trajectory_ids, dones, extra_fields)
            if done or extra.get("is_last_step")
        ]
        if cleanups:
            await asyncio.gather(*cleanups, return_exceptions=True)

        return obs, dones, valids
    

    def conduct_action(self, trajectory_id, action, extra_field):
        return asyncio.run(
            self.aconduct_action(trajectory_id, action, extra_field)
        )

    def get_observations(self, trajectory_ids, actions, extra_fields):
        return asyncio.run(
            self.aget_observations(trajectory_ids, actions, extra_fields)
        )