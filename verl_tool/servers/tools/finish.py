from .base import BaseTool, register_tool
import regex as re
import asyncio

@register_tool
class FinishTool(BaseTool):
    tool_type = "finish"
    timeout = 10
    
    def __init__(self, num_workers=1, other_tools:list = []):
        super().__init__(num_workers)
        self.other_tools = other_tools
    
    def get_usage_inst(self):
        return ""
    
    def parse_action(self, action:str):
        """
        Parse the raw action string to check for answer tags or finish conditions.
        Implements the finish condition logic that was originally in serve.py lines 107-109.
        """
        # Default behavior - trajectory ends without explicit answer
        return "", False
    
    def conduct_action(self, trajectory_id, action, extra_data):
        """同步版本 - 为了兼容性保留，但建议使用异步版本"""
        action, is_valid = self.parse_action(action)
        
        observation = ""
        done = True
        
        # 同步清理 - 只处理没有异步版本的工具
        for tool in self.other_tools:
            if tool.has_env(trajectory_id):
                # 只清理没有异步版本的工具，避免事件循环冲突
                if not hasattr(tool, 'adelete_env'):
                    tool.delete_env(trajectory_id)
                # 有异步版本的工具留给异步方法处理
        
        return observation, done, is_valid
    
    async def aconduct_action(self, trajectory_id, action, extra_field):
        """异步版本 - 推荐使用"""
        action, is_valid = self.parse_action(action)
        
        observation = ""
        done = True
        
        # 异步清理所有环境
        cleanup_tasks = []
        for tool in self.other_tools:
            if tool.has_env(trajectory_id):
                if hasattr(tool, 'adelete_env'):
                    # 优先使用异步版本
                    cleanup_tasks.append(tool.adelete_env(trajectory_id))
                else:
                    # 对于只有同步版本的工具，在线程池中运行
                    cleanup_tasks.append(
                        asyncio.get_event_loop().run_in_executor(
                            None, tool.delete_env, trajectory_id
                        )
                    )
        
        # 等待所有清理任务完成
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        return observation, done, is_valid
    
    async def aget_observations(self, trajectory_ids, actions, extra_fields):
        """批量异步处理"""
        tasks = []
        for i in range(len(trajectory_ids)):
            task = self.aconduct_action(trajectory_ids[i], actions[i], extra_fields[i])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        observations, dones, valids = zip(*results)
        return list(observations), list(dones), list(valids)