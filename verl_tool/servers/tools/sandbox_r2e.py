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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
        self.ds = ds
        self.command_files = command_files or []
        
        # Initialize environment directly
        env_args = EnvArgs(ds=ds)
        self.env = RepoEnv(env_args)
        # Reset environment
        self.env.reset()
        
        # Add command files if provided
        if self.command_files:
            self.env.add_commands(self.command_files)
            print("add command files")
        
        # print(self.env.commands)

    def start_env(self) -> str:
        """
        Start environment and return initial observation
        
        Returns:
            Initial problem statement
        """
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
        if "<compute_reward>sandbox_r2e</compute_reward>" in action_str:
            reward, valid, test_output = self.reward_env("<compute_reward>sandbox_r2e</compute_reward>")
            reward_str = f"<reward>{reward}</reward>"#<test_output>{test_output}</test_output>" # 暂时不返回test_output
            return reward_str, True, True

        try:
            # Parse action from string (similar to agent.py parse_response)
            action = Action.from_string(action_str)
            
            # Check if action parsing failed
            if action is None or not action.function_name:
                return "Failed to parse action. Invalid XML format or missing function name.", False, False
            
            # Execute action (same pattern as agent.py)
            with suppress_stdout_stderr():
                obs, reward, done, info = self.env.step(action, timeout=900)

            # if done, then pad the reward into the observation
            if done:
                reward, valid, test_output = self.reward_env("<compute_reward>sandbox_r2e</compute_reward>")
                obs = str(obs)+f"<reward>{reward}</reward>"
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
        if action_str == "<compute_reward>sandbox_r2e</compute_reward>":
            with suppress_stdout_stderr():
                reward, test_output = self.env.runtime._calculate_reward(get_test_output=True)
            reward = float(reward)
            return reward, True, test_output
        else:
            raise ValueError(f"Invalid reward action: {action_str}")
        
    def close_env(self):
        # print(f"in the close_env, timestamp: {time.time()}, job_name: {self.env.runtime.job_name}")
        # Write to env_deleted.jsonl the following information: trajectory_id, timestamp, job_name
        json_content = {
            "timestamp": time.time(),
            "job_name": self.env.runtime.job_name
        }
        print(f"Deleting env, timestamp: {time.time()}, job_name: {self.env.runtime.job_name}")
        with open("env_deleted.jsonl", "a") as f:
            f.write(json.dumps(json_content) + "\n")
        self.env.close()

@register_tool
class SandboxR2ETool(BaseTool):
    """
    SandboxR2ETool uses Ray actors to manage R2E environment sessions.
    Each trajectory_id has a dedicated actor. It supports initial
    render (action=None) and step operations.
    """
    tool_type = "sandbox_r2e"

    def __init__(self, num_workers=4):
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

    def delete_env(self, trajectory_id):
        """Kill and remove the actor."""
        # return
        #print(f"in the self deleting env, trajectory_id: {trajectory_id}, trajectory_id in self.env_actors = {trajectory_id in self.env_actors}")
        if trajectory_id in self.env_actors:
            future = self.env_actors[trajectory_id].close_env.remote()
            ray.get(future)
            # self.env_actors[trajectory_id].close_env.remote()
            ray.kill(self.env_actors[trajectory_id], no_restart=True)
            del self.env_actors[trajectory_id]
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)

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

    def conduct_action(self, trajectory_id: str, action: str, extra_field: dict):
        """
        Execute a *single* action on the environment for `trajectory_id`.

        Returns
        -------
        obs : str
            Environment observation (empty string if episode finished).
        done : bool
            Whether the episode ended with this step.
        valid : bool
            Whether the action itself was valid.
        """
#         print(
#     "--------------------------------\n"
#     "conduct_action\n"
#     f"trajectory_id {trajectory_id}\n"
#     f"action {action}\n"
#     f"extra_field {extra_field}\n"
#     "--------------------------------"
# )
#         exit(1)
        # 1) Ensure an actor exists (lazy creation).
        actor = self.load_env(trajectory_id)
        if actor is None:
            # Create a brand-new R2EEnvActor for this trajectory.
            ds = extra_field.get("ds", extra_field.get("extra_fields", extra_field))
            # Default command files
            command_files = [
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/search.py"),
                # Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/search_dir.py"),
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/file_editor.py"),
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/execute_bash.py"),
                Path("/minimax-dialogue/users/ruobai/cogito_local/r2e-gym/src/r2egym/agenthub/tools/finish.py")
            ]
            # Filter existing command files
            existing_command_files = [f for f in command_files if f.exists()]
            
            actor = R2EEnvActor.remote(ds, existing_command_files)
            self.save_env(trajectory_id, actor)

        # 2) Decide whether we are rendering the first page or taking a step.
        fut = (
            actor.start_env.remote()
            if action is None or action == ""
            else actor.step_env.remote(action)
        )

        # 3) Wait for the Ray RPC to finish (blocks the calling thread only).
        result = ray.get(fut)
        if isinstance(result, tuple):           # step_env
            obs, done, valid = result
        else:                                   # start_env
            obs, done, valid = result, False, True

        # Debug output
        # output = (
        #     "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        #     f"trajectory_id: {trajectory_id}\n"
        #     f"action: {action}\n"
        #     f"extra_field: {extra_field}\n"
        #     f"observation: {obs}\n"
        #     f"done: {done}, valid: {valid}\n"
        #     "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        # )
        # print(output)

        # 4) Refresh LRU order *after* the step.
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)
        self.actor_creation_order.append(trajectory_id)

        # 5) Clean-up if the episode finished.

        if not valid:
            obs = f"The action {action} is invalid, please retry, obs is {obs}"

        return obs, done, valid

    def get_observations(self, trajectory_ids, actions, extra_fields):
        """
        Batched version of `conduct_action` with thread-pool parallelism.
        (A process-pool is **not** required; Ray already runs the envs
        out-of-process.)

        Parameters
        ----------
        trajectory_ids : list[str]
        actions        : list[str | None]
        extra_fields   : list[dict]

        Returns
        -------
        observations : list[str]
        dones        : list[bool]
        valid_flags  : list[bool]
        """
        # print(f"get_observations: {trajectory_ids}, {actions}")
        n = len(trajectory_ids)
        observations = [""]   * n
        dones        = [False] * n
        valid_flags  = [True]  * n

        # ----------------------------------------------------------------- #
        # Parallel fan-out using a thread pool                              #
        # ----------------------------------------------------------------- #

        actions_with_last_step = []
        for i in range(len(trajectory_ids)):
            if extra_fields[i].get('is_last_step', False): # 强行把last step改成calculate reward
                actions_with_last_step.append("<compute_reward>sandbox_r2e</compute_reward>") # If there is last step, directly compute the reward
            else:
                actions_with_last_step.append(actions[i])

        def _worker(idx: int):
            tid   = trajectory_ids[idx]
            act   = actions_with_last_step[idx] # Changed from actions if last step
            extra = extra_fields[idx].get("extra_fields", extra_fields[idx])
            try:
                return (*self.conduct_action(tid, act, extra), None)
            except Exception as e:
                raise e
                return ("", False, False, e)   # bubble error to main thread

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(_worker, i) for i in range(n)]
            for i, fut in enumerate(futures):
                obs, done, valid, err = fut.result()
                observations[i] = obs
                dones[i]        = done
                valid_flags[i]  = valid
                if err:
                    print(f"[ERROR] trajectory_id={trajectory_ids[i]}: {err}")

        # print(f"Checkpoint 1, len(trajectory_ids): {len(trajectory_ids)}")
        # 并发执行delete_env
        def _delete_env_if_needed(i):
            # print(f"deleting env, trajectory_id: {trajectory_ids[i]}, is_last_step: {extra_fields[i].get('is_last_step', False)}, done: {dones[i]}")
            if extra_fields[i].get('is_last_step', False) or dones[i]:
                self.delete_env(trajectory_ids[i])

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            pool.map(_delete_env_if_needed, range(len(trajectory_ids)))

        # print(f"get_observations finished, observations: {observations}, dones: {dones}, valid_flags: {valid_flags}")
        # print(f"env_actors: {self.env_actors}")

        return observations, dones, valid_flags

    def _cleanup_actors_if_needed(self):
        """Remove oldest actors if count exceeds limit."""
        while len(self.env_actors) > 512:
            # raise RuntimeError("Too many actors, please reduce the number of concurrent requests.")
            oldest = self.actor_creation_order.pop(0)
            print(f"[INFO] Deleting actor {oldest} due to too many actors.")
            # if oldest in self.env_actors:
            #     ray.kill(self.env_actors[oldest], no_restart=True)
            #     del self.env_actors[oldest]
            # if oldest in self.actor_creation_order:
            #     self.actor_creation_order.remove(oldest)
            self.delete_env(oldest)