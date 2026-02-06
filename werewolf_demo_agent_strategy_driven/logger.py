# logger.py
"""
游戏日志系统 - 记录所有Agent的思考过程
"""
import os
from datetime import datetime
from typing import Optional


class GameLogger:
    """游戏日志记录器"""
    
    def __init__(self, log_dir: str = "./game_logs", game_id: str = None):
        """
        初始化日志记录器
        
        :param log_dir: 日志目录
        :param game_id: 游戏ID，如果为None则使用时间戳
        """
        self.log_dir = log_dir
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 生成日志文件名
        if game_id is None:
            game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_file = os.path.join(log_dir, f"game_{game_id}.txt")
        
        # 初始化日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"狼人杀游戏日志\n")
            f.write(f"游戏ID: {game_id}\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        print(f"✓ 日志文件已创建: {self.log_file}")
    
    def log_section(self, title: str):
        """记录章节标题"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_phase(self, phase: str, day: int):
        """记录游戏阶段"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"第 {day} 天 - {phase}\n")
            f.write("-" * 80 + "\n\n")

    def log_agent_thinking(self, player_id: str, role: str, context: str,
                           character: str = "",  # 改为可选参数
                           query: Optional[str] = None, memory_result: Optional[str] = None,
                           llm_prompt: Optional[str] = None, llm_response: Optional[str] = None,
                           final_decision: Optional[str] = None):
        """
        记录Agent的完整思考过程

        :param player_id: 玩家ID
        :param role: 角色
        :param context: 思考场景
        :param character: 性格特征（可选）
        :param query: 查询问题
        :param memory_result: 记忆检索结果
        :param llm_prompt: 发送给LLM的完整prompt
        :param llm_response: LLM的原始响应
        :param final_decision: 最终决策
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        with open(self.log_file, 'a', encoding='utf-8') as f:
            # 添加性格信息
            if character:
                f.write(f"\n[{timestamp}] {player_id} ({role}) - {character} - {context}\n")
            else:
                f.write(f"\n[{timestamp}] {player_id} ({role}) - {context}\n")

            f.write("-" * 60 + "\n")

            if query:
                f.write(f"\n【查询问题】\n{query}\n")

            if memory_result:
                f.write(f"\n【记忆检索结果】\n{memory_result}\n")

            if llm_prompt:
                f.write(f"\n【LLM Prompt】\n{llm_prompt}\n")

            if llm_response:
                f.write(f"\n【LLM 原始响应】\n{llm_response}\n")

            if final_decision:
                f.write(f"\n【最终决策】\n{final_decision}\n")

            f.write("\n" + "." * 60 + "\n")

    def log_event(self, event: str):
        """记录游戏事件"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {event}\n")
    
    def log_game_state(self, day: int, alive_players: list, dead_players: list, roles: dict = None):
        """记录游戏状态"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n【游戏状态 - 第{day}天】\n")
            f.write(f"存活玩家: {', '.join(alive_players)}\n")
            f.write(f"死亡玩家: {', '.join(dead_players) if dead_players else '无'}\n")
            
            if roles:
                f.write(f"\n【角色信息】\n")
                for player_id in sorted(roles.keys()):
                    status = "存活" if player_id in alive_players else "死亡"
                    f.write(f"  {player_id}: {roles[player_id]} ({status})\n")
            
            f.write("\n")
    
    def log_memory_retrieval(self, player_id: str, queries: list, results: list):
        """记录记忆检索过程"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] {player_id} - 记忆检索过程\n")
            f.write("-" * 60 + "\n")
            
            for i, (query, result) in enumerate(zip(queries, results), 1):
                f.write(f"\n查询 {i}: {query}\n")
                f.write(f"结果:\n{result}\n")
                f.write("-" * 40 + "\n")
    
    def log_cot_reasoning(self, player_id: str, query: str, intent: dict, 
                         seed_count: int, expanded_count: int, reasoning_chain: str):
        """记录CoT推理过程"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] {player_id} - CoT推理过程\n")
            f.write("-" * 60 + "\n")
            f.write(f"查询: {query}\n")
            f.write(f"意图分析: {intent}\n")
            f.write(f"种子记忆数: {seed_count}\n")
            f.write(f"扩展后记忆数: {expanded_count}\n")
            f.write(f"\n推理链:\n{reasoning_chain}\n")
    
    def log_game_end(self, winner: str, final_state: dict):
        """记录游戏结束"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("游戏结束\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"胜利方: {winner}\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'alive_players' in final_state:
                f.write(f"存活玩家: {', '.join(final_state['alive_players'])}\n")
            if 'dead_players' in final_state:
                f.write(f"死亡玩家: {', '.join(final_state['dead_players'])}\n")
            
            if 'roles' in final_state:
                f.write(f"\n最终角色揭示:\n")
                for player_id, role in sorted(final_state['roles'].items()):
                    status = "存活" if player_id in final_state.get('alive_players', []) else "死亡"
                    f.write(f"  {player_id}: {role} ({status})\n")
    
    def close(self):
        """关闭日志（添加结束标记）"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("日志记录结束\n")
            f.write("=" * 80 + "\n")


# 全局日志实例（单例模式）
_global_logger: Optional[GameLogger] = None

def init_logger(log_dir: str = "./game_logs", game_id: str = None) -> GameLogger:
    """初始化全局日志记录器"""
    global _global_logger
    _global_logger = GameLogger(log_dir, game_id)
    return _global_logger

def get_logger() -> Optional[GameLogger]:
    """获取全局日志记录器"""
    return _global_logger

# 删除有问题的log_agent_thinking快捷函数，改为更简单的方式
def log_agent_thinking(player_id: str, role: str, context: str, character: str = "", **kwargs):
    """快捷方法：记录Agent思考"""
    if _global_logger:
        # 直接调用类方法，明确传递所有参数
        _global_logger.log_agent_thinking(
            player_id=player_id,
            role=role,
            context=context,
            character=character,
            query=kwargs.get('query'),
            memory_result=kwargs.get('memory_result'),
            llm_prompt=kwargs.get('llm_prompt'),
            llm_response=kwargs.get('llm_response'),
            final_decision=kwargs.get('final_decision')
        )

def log_event(event: str):
    """快捷方法：记录事件"""
    if _global_logger:
        _global_logger.log_event(event)


def log_memory_retrieval(*args, **kwargs):
    """快捷方法：记录记忆检索"""
    if _global_logger:
        _global_logger.log_memory_retrieval(*args, **kwargs)


def log_cot_reasoning(*args, **kwargs):
    """快捷方法：记录CoT推理"""
    if _global_logger:
        _global_logger.log_cot_reasoning(*args, **kwargs)
