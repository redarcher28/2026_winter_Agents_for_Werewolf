# game_manager.py
"""
游戏管理器 - 控制游戏流程
"""
import asyncio
import random
from typing import Dict, List, Optional
from datetime import datetime
from openai import AsyncOpenAI

from game_config import *
from agent import create_agent, WerewolfAgent
import shutil
import os


class GameManager:
    """游戏管理器"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        
        # 随机分配角色
        self.roles = self._assign_roles()
        
        # 游戏状态
        self.day = 0
        self.alive_players = list(self.roles.keys())
        self.dead_players = []
        
        # 创建所有 Agent（每个使用独立的 API Key）
        self.agents: Dict[str, WerewolfAgent] = {}
        for player_id, role in self.roles.items():
            api_key = api_keys.get(player_id, list(api_keys.values())[0])
            self.agents[player_id] = create_agent(player_id, role, api_key)
        
        # 为狼人设置队友信息
        werewolves = [pid for pid, role in self.roles.items() if role == 'werewolf']
        for wolf_id in werewolves:
            teammates = [w for w in werewolves if w != wolf_id]
            if hasattr(self.agents[wolf_id], 'set_teammates'):
                self.agents[wolf_id].set_teammates(teammates)
        
        print(f"✓ 已创建 {len(self.agents)} 个 Agent（每个使用独立 API Key）")
        print(f"\n角色分配（随机）：")
        for player_id, agent in self.agents.items():
            print(f"  - {player_id}: {agent.role}")
    
    def _assign_roles(self) -> Dict[str, str]:
        """随机分配角色"""
        # 生成玩家 ID 列表
        player_ids = [f"player_{i+1}" for i in range(TOTAL_PLAYERS)]
        
        # 生成角色列表
        roles_list = []
        for role, count in ROLE_CONFIG.items():
            roles_list.extend([role] * count)
        
        # 验证角色数量
        if len(roles_list) != TOTAL_PLAYERS:
            raise ValueError(f"角色总数 ({len(roles_list)}) 与玩家数 ({TOTAL_PLAYERS}) 不匹配")
        
        # 随机打乱角色
        random.shuffle(roles_list)
        
        # 分配角色
        role_assignment = dict(zip(player_ids, roles_list))
        
        return role_assignment
    
    def get_game_state(self) -> Dict:
        """获取当前游戏状态"""
        return {
            "day": self.day,
            "alive_players": self.alive_players.copy(),
            "dead_players": self.dead_players.copy()
        }
    
    def add_public_event(self, event_type: str, data: Dict, description: str):
        """添加公共事件到所有存活玩家的记忆"""
        event = {
            "event_id": f"evt_{datetime.now().timestamp()}",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": {**data, "day": self.day}
        }
        
        for player_id in self.alive_players:
            self.agents[player_id].add_public_memory(event, description)
    
    def add_private_event(self, player_ids: List[str], event_type: str, 
                         data: Dict, description: str):
        """添加私有事件到指定玩家的记忆"""
        event = {
            "event_id": f"evt_{datetime.now().timestamp()}",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": {**data, "day": self.day}
        }
        
        for player_id in player_ids:
            if player_id in self.agents:
                self.agents[player_id].add_public_memory(event, description)
    
    async def night_phase(self):
        """夜晚阶段"""
        print(f"\n{'='*70}")
        print(f"第 {self.day} 夜".center(70))
        print(f"{'='*70}\n")
        
        # 添加夜晚开始的公共事件
        self.add_public_event(
            "phase_change",
            {"phase": "night"},
            f"【系统公告】第{self.day}天夜晚开始。当前存活：{', '.join(self.alive_players)}"
        )
        
        # 1. 狼人行动
        killed_player = await self.werewolf_phase()
        
        # 2. 预言家查验
        await self.seer_phase()
        
        # 3. 女巫行动
        saved_player, poisoned_player = await self.witch_phase(killed_player)
        
        # 确定最终死亡名单
        final_deaths = []
        if killed_player and killed_player != saved_player:
            final_deaths.append(killed_player)
        if poisoned_player:
            final_deaths.append(poisoned_player)
        
        # 更新死亡名单
        for player in final_deaths:
            if player in self.alive_players:
                self.alive_players.remove(player)
                self.dead_players.append(player)
                self.agents[player].is_alive = False
        
        return final_deaths
    
    async def werewolf_phase(self) -> Optional[str]:
        """狼人阶段"""
        print("\n[狼人阶段]")
        
        werewolves = [pid for pid, role in self.roles.items() 
                     if role == 'werewolf' and pid in self.alive_players]
        
        if not werewolves:
            return None
        
        # 狼人讨论（3轮）
        discussion_log = []
        for round_num in range(1, WEREWOLF_DISCUSSION_ROUNDS + 1):
            print(f"\n  第 {round_num} 轮讨论：")
            
            previous_discussion = "\n".join(discussion_log) if discussion_log else "这是第一轮讨论"
            
            for wolf_id in werewolves:
                result = await self.agents[wolf_id].night_discussion(
                    round_num, WEREWOLF_DISCUSSION_ROUNDS,
                    [w for w in werewolves if w != wolf_id],
                    previous_discussion,
                    self.get_game_state()
                )
                
                speech = result['speech']
                print(f"    {wolf_id}: {speech}")
                discussion_log.append(f"[{wolf_id}]: {speech}")
                
                # 添加到狼人共享记忆
                self.add_private_event(
                    werewolves,
                    "werewolf_discussion",
                    {"speaker": wolf_id, "content": speech, "round": round_num},
                    f"狼人讨论第{round_num}轮 - {wolf_id}说：{speech}"
                )
        
        # 狼人投票
        print(f"\n  狼人投票：")
        votes = {}
        discussion_summary = "\n".join(discussion_log)
        
        for wolf_id in werewolves:
            target = await self.agents[wolf_id].night_vote(
                discussion_summary,
                self.get_game_state(),
                self.roles  # 传入角色信息
            )
            votes[wolf_id] = target
            print(f"    {wolf_id} 投票刀 {target}")
        
        # 统计票数
        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1
        
        killed = max(vote_counts.items(), key=lambda x: x[1])[0]
        print(f"\n  ✓ 狼人决定刀 {killed}")
        
        # 添加到狼人记忆
        self.add_private_event(
            werewolves,
            "werewolf_kill",
            {"target": killed, "votes": votes},
            f"狼人投票决定刀 {killed}"
        )
        
        return killed
    
    async def seer_phase(self):
        """预言家阶段"""
        print("\n[预言家阶段]")
        
        seer_id = None
        for pid, role in self.roles.items():
            if role == 'seer' and pid in self.alive_players:
                seer_id = pid
                break
        
        if not seer_id:
            print("  预言家已死亡")
            return
        
        result = await self.agents[seer_id].night_action(
            self.get_game_state(),
            self.roles  # 传入角色信息
        )
        
        if result:
            target = result['target']
            check_result = result['result']
            print(f"  {seer_id} 查验 {target}，结果：{check_result}")
            
            # 添加到预言家私有记忆
            self.add_private_event(
                [seer_id],
                "seer_check",
                {"target": target, "result": check_result},
                f"我查验了 {target}，结果是：{check_result}"
            )
    
    async def witch_phase(self, killed_player: Optional[str]) -> tuple:
        """女巫阶段"""
        print("\n[女巫阶段]")
        
        witch_id = None
        for pid, role in self.roles.items():
            if role == 'witch' and pid in self.alive_players:
                witch_id = pid
                break
        
        if not witch_id:
            print("  女巫已死亡")
            return None, None
        
        game_state = self.get_game_state()
        game_state['killed_tonight'] = killed_player
        
        result = await self.agents[witch_id].night_action(game_state)
        
        saved_player = None
        poisoned_player = None
        
        if result:
            action = result['action']
            target = result.get('target')
            
            if action == 'save':
                saved_player = target
                print(f"  {witch_id} 使用解药救了 {target}")
                self.add_private_event(
                    [witch_id],
                    "witch_save",
                    {"target": target},
                    f"我使用解药救了 {target}"
                )
            
            elif action == 'poison':
                poisoned_player = target
                print(f"  {witch_id} 使用毒药毒了 {target}")
                self.add_private_event(
                    [witch_id],
                    "witch_poison",
                    {"target": target},
                    f"我使用毒药毒了 {target}"
                )
        else:
            print(f"  {witch_id} 不使用药水")
        
        return saved_player, poisoned_player
    
    async def day_phase(self):
        """白天阶段"""
        print(f"\n{'='*70}")
        print(f"第 {self.day} 天 - 讨论阶段".center(70))
        print(f"{'='*70}\n")
        
        # 公布死亡信息
        if self.dead_players:
            last_dead = self.dead_players[-1] if len(self.dead_players) == 1 else self.dead_players[-2:]
            death_msg = f"昨晚 {', '.join(last_dead) if isinstance(last_dead, list) else last_dead} 死亡"
            print(f"[系统公告] {death_msg}")
        
        # 添加白天开始的公共事件
        self.add_public_event(
            "phase_change",
            {"phase": "day_discussion"},
            f"【系统公告】第{self.day}天白天开始。昨晚死亡：{', '.join(self.dead_players[-2:]) if len(self.dead_players) >= 2 else self.dead_players[-1] if self.dead_players else '无'}。当前存活：{', '.join(self.alive_players)}"
        )
        
        # 按顺序发言
        print(f"\n[发言阶段]")
        for i, player_id in enumerate(sorted(self.alive_players), 1):
            # 在发言前添加发言顺序信息到所有人的记忆
            speech_order_msg = f"【发言顺序】现在轮到第{i}位玩家 {player_id} 发言（共{len(self.alive_players)}位存活）。"
            if i == 1:
                speech_order_msg += " 这是第一位发言，之前还没有人发言。"
            else:
                speech_order_msg += f" 前面已有{i-1}位玩家发言完毕。"
            
            self.add_public_event(
                "speech_turn",
                {"current_speaker": player_id, "order": i, "total_alive": len(self.alive_players)},
                speech_order_msg
            )
            
            print(f"\n  第 {i} 位发言 - {player_id} ({self.roles[player_id]})：")
            
            speech = await self.agents[player_id].day_speech(self.get_game_state())
            print(f"    {speech}")
            
            # 发言后立即添加到所有人的记忆
            self.add_public_event(
                "player_speech",
                {"player_id": player_id, "content": speech, "order": i},
                f"{player_id} 发言说：{speech}"
            )
            
            await asyncio.sleep(0.5)
    
    async def vote_phase(self):
        """投票阶段"""
        print(f"\n{'='*70}")
        print(f"第 {self.day} 天 - 投票阶段".center(70))
        print(f"{'='*70}\n")
        
        # 添加投票开始的公共事件
        self.add_public_event(
            "phase_change",
            {"phase": "voting"},
            f"【系统公告】第{self.day}天投票阶段开始。当前存活：{', '.join(self.alive_players)}"
        )
        
        # 所有人投票
        votes = {}
        print("[投票中...]")
        for player_id in self.alive_players:
            target = await self.agents[player_id].vote(self.get_game_state())
            votes[player_id] = target
            print(f"  {player_id} 投票给 {target}")
            await asyncio.sleep(0.3)
        
        # 统计票数
        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1
        
        # 确定被放逐的玩家
        exiled = max(vote_counts.items(), key=lambda x: x[1])[0]
        
        print(f"\n[投票结果]")
        for target, count in vote_counts.items():
            print(f"  {target}: {count}票")
        print(f"\n  ✓ {exiled} 被放逐")
        
        # 更新状态
        if exiled in self.alive_players:
            self.alive_players.remove(exiled)
            self.dead_players.append(exiled)
            self.agents[exiled].is_alive = False
        
        # 添加到所有人的记忆
        self.add_public_event(
            "vote_result",
            {"votes": votes, "result": exiled, "vote_counts": vote_counts},
            f"投票结束，{exiled} 被放逐。当前存活：{', '.join(self.alive_players)}"
        )
        
        return exiled
    
    def check_game_end(self) -> Optional[str]:
        """检查游戏是否结束"""
        alive_werewolves = [p for p in self.alive_players if self.roles[p] == 'werewolf']
        alive_good = [p for p in self.alive_players if self.roles[p] != 'werewolf']
        
        if not alive_werewolves:
            return "good"  # 好人胜利
        elif len(alive_werewolves) >= len(alive_good):
            return "werewolf"  # 狼人胜利
        
        return None
    
    async def run_game(self, max_days: int = 5):
        """运行游戏"""
        print(f"\n{'='*70}")
        print("狼人杀多 Agent 游戏开始".center(70))
        print(f"{'='*70}\n")
        
        while self.day < max_days:
            self.day += 1
            
            # 夜晚阶段
            await self.night_phase()
            
            # 检查游戏是否结束
            winner = self.check_game_end()
            if winner:
                break
            
            # 白天阶段
            await self.day_phase()
            
            # 投票阶段
            await self.vote_phase()
            
            # 检查游戏是否结束
            winner = self.check_game_end()
            if winner:
                break
        
        # 游戏结束
        print(f"\n{'='*70}")
        print("游戏结束".center(70))
        print(f"{'='*70}\n")
        
        if winner == "good":
            print("✓ 好人阵营胜利！")
        elif winner == "werewolf":
            print("✓ 狼人阵营胜利！")
        else:
            print("游戏达到最大天数")
        
        print(f"\n存活玩家：{', '.join(self.alive_players)}")
        print(f"死亡玩家：{', '.join(self.dead_players)}")
        
        print(f"\n角色揭示：")
        for player_id, role in self.roles.items():
            status = "存活" if player_id in self.alive_players else "死亡"
            print(f"  {player_id}: {role} ({status})")
