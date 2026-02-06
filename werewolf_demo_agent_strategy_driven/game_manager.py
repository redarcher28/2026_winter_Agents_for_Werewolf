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
from logger import get_logger, log_event, log_agent_thinking
from human_interface import HumanInterface, MessageType

class GameManager:
    """游戏管理器"""
    
    def __init__(self, api_keys: Dict[str, str], HUMAN_PLAYER: bool = False, human_player_id: Optional[str] = None, human_player_role: Optional[str] = None):
        self.api_keys = api_keys
        self.human_player_id = human_player_id
        self.HUMAN_PLAYER_exist = HUMAN_PLAYER
        self.human_player_role = human_player_role
        
        # 随机分配角色
        self.roles = self._assign_roles()

        # 随机分配性格
        self.characters = self._assign_characters()
        
        # 游戏状态
        self.day = 0
        self.alive_players = list(self.roles.keys())
        self.dead_players = []
        self.night_deaths = []  # 记录当晚死亡的玩家
        self.evening_deaths = []  # 记录每晚死亡的玩家
        self.vote_deaths = []  # 记录投票死亡的玩家
        
        # 创建所有 Agent（每个使用独立的 API Key）
        self.agents: Dict[str, WerewolfAgent] = {}
        for player_id, role in self.roles.items():
            api_key = api_keys.get(player_id, list(api_keys.values())[0])
            character = self.characters[player_id]
            self.agents[player_id] = create_agent(player_id, role, api_key, character)

            # 判断是否为人类玩家
            is_human = (player_id == human_player_id) if human_player_id else False

            if is_human:
                print(f"✓ {player_id} 被设置为人类玩家")
                # 人类玩家使用伪API Key
                api_key = "human_player_key"

            self.agents[player_id] = create_agent(
                player_id, role, api_key, character, is_human=is_human
            )
        
        # 为狼人设置队友信息
        werewolves = [pid for pid, role in self.roles.items() if role == 'werewolf']
        for wolf_id in werewolves:
            teammates = [w for w in werewolves if w != wolf_id]
            if hasattr(self.agents[wolf_id], 'set_teammates'):
                self.agents[wolf_id].set_teammates(teammates)
        
        print(f"✓ 已创建 {len(self.agents)} 个 Agent（每个使用独立 API Key）")
        print(f"\n角色分配（随机）：")
        for player_id, agent in self.agents.items():
            # print(f"  - {player_id}: {agent.role}")
            character_key = agent.character
            character_name = CHARACTER_CONFIG[character_key]["name"]
            player_type = "【人类】" if player_id == human_player_id else "【AI】"
            if not self.HUMAN_PLAYER_exist:
                print(f"  - {player_id}: {agent.role}, {character_name}, {player_type}")

    
    # def _assign_roles(self) -> Dict[str, str]:
    #     """随机分配角色"""
    #     # 生成玩家 ID 列表
    #     player_ids = [f"player_{i+1}" for i in range(TOTAL_PLAYERS)]
    #
    #     # 生成角色列表
    #     roles_list = []
    #     for role, count in ROLE_CONFIG.items():
    #         roles_list.extend([role] * count)
    #
    #     # 验证角色数量
    #     if len(roles_list) != TOTAL_PLAYERS:
    #         raise ValueError(f"角色总数 ({len(roles_list)}) 与玩家数 ({TOTAL_PLAYERS}) 不匹配")
    #
    #     # 随机打乱角色
    #     random.shuffle(roles_list)
    #
    #     # 分配角色
    #     role_assignment = dict(zip(player_ids, roles_list))
    #
    #     return role_assignment
    def _assign_roles(self) -> Dict[str, str]:
        """随机分配角色，支持指定人类玩家角色"""
        # 生成玩家 ID 列表
        player_ids = [f"player_{i + 1}" for i in range(TOTAL_PLAYERS)]

        # 如果存在人类玩家
        if self.HUMAN_PLAYER_exist and self.human_player_id:
            print(f"\n[调试] 存在人类玩家: {self.human_player_id}")
            print(f"[调试] 指定的人类玩家角色: {self.human_player_role}")

            # 确保人类玩家ID在玩家列表中
            if self.human_player_id not in player_ids:
                raise ValueError(f"人类玩家ID {self.human_player_id} 不在玩家列表中")

            # 生成角色池
            role_pool = []
            for role, count in ROLE_CONFIG.items():
                role_pool.extend([role] * count)

            print(f"[调试] 原始角色池: {role_pool}")

            # 确定人类玩家的角色
            human_role = None

            # 如果有指定角色
            if self.human_player_role and self.human_player_role in ROLE_CONFIG:
                human_role = self.human_player_role
                print(f"[调试] 使用指定角色: {human_role}")
            else:
                # 随机分配一个角色
                human_role = random.choice(role_pool)
                print(f"[调试] 随机分配角色: {human_role}")

            # 检查选择的角色是否在角色池中
            if human_role not in role_pool:
                raise ValueError(f"选择的角色 {human_role} 不在角色池中")

            # 从角色池中移除这个角色（因为已经分配给人类玩家了）
            role_pool.remove(human_role)
            print(f"[调试] 移除人类玩家角色后角色池: {role_pool}")

            # 生成AI玩家的ID列表（排除人类玩家）
            ai_player_ids = [pid for pid in player_ids if pid != self.human_player_id]
            print(f"[调试] AI玩家ID列表: {ai_player_ids}")

            # 验证角色数量是否匹配
            if len(role_pool) != len(ai_player_ids):
                raise ValueError(f"剩余角色数 ({len(role_pool)}) 与AI玩家数 ({len(ai_player_ids)}) 不匹配")

            # 随机打乱剩余角色
            random.shuffle(role_pool)
            print(f"[调试] 打乱后的剩余角色池: {role_pool}")

            # 分配角色：先分配人类玩家，再分配AI玩家
            role_assignment = {}
            role_assignment[self.human_player_id] = human_role

            for ai_player_id, role in zip(ai_player_ids, role_pool):
                role_assignment[ai_player_id] = role

            print(f"  ✓ 人类玩家 {self.human_player_id} 被分配为: {human_role}")

            return role_assignment

        # 如果没有人类玩家，使用原有的随机分配逻辑
        else:
            print("\n[调试] 没有人类玩家，使用纯随机分配")
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

    def _assign_characters(self) -> Dict[str, str]:
        """从16个性格中随机抽取8个不同的性格分配给8个玩家，确保有至少2个正经性格"""
        # 生成玩家 ID 列表
        player_ids = [f"player_{i + 1}" for i in range(TOTAL_PLAYERS)]

        # 定义正经性格和搞怪性格
        serious_characters = ["character_01", "character_02", "character_03", "character_04"]
        funny_characters = ["character_05", "character_06", "character_07", "character_08",
                            "character_09", "character_10", "character_11", "character_12",
                            "character_13", "character_14", "character_15", "character_16",
                            "character_17", "character_18", "character_19", "character_20"]

        # 确定要抽取的正经性格数量（2-4个）
        serious_count = random.randint(2, 4)
        funny_count = TOTAL_PLAYERS - serious_count

        # 抽取正经性格
        selected_serious = random.sample(serious_characters, serious_count)

        # 抽取搞怪性格
        selected_funny = random.sample(funny_characters, funny_count)

        # 合并并随机打乱
        selected_characters = selected_serious + selected_funny
        random.shuffle(selected_characters)

        # 分配性格
        character_assignment = dict(zip(player_ids, selected_characters))

        return character_assignment
    
    def get_game_state(self) -> Dict:
        """获取当前游戏状态"""
        return {
            "day": self.day,
            "alive_players": self.alive_players.copy(),
            "dead_players": self.dead_players.copy(),
            "evening_deaths": self.evening_deaths.copy(),
            "vote_deaths": self.vote_deaths.copy()
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
        
        # 记录到日志
        if ENABLE_LOGGING:
            logger = get_logger()
            if logger:
                logger.log_phase("夜晚阶段", self.day)
        
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
                self.evening_deaths.append(player)
                self.agents[player].is_alive = False
        
        # 记录本夜死亡的玩家（用于白天公布）
        self.night_deaths = final_deaths
        
        return final_deaths
    
    async def werewolf_phase(self) -> Optional[str]:
        """狼人阶段"""
        print("\n[狼人阶段]")
        
        werewolves = [pid for pid, role in self.roles.items() 
                     if role == 'werewolf' and pid in self.alive_players]
        
        if not werewolves:
            return None

        # 检查是否有狼人是人类玩家
        human_werewolves = [w for w in werewolves if w == self.human_player_id]
        has_human_werewolf = len(human_werewolves) > 0
        
        # 狼人讨论（3轮）
        discussion_log = []
        for round_num in range(1, WEREWOLF_DISCUSSION_ROUNDS + 1):
            if has_human_werewolf or not self.HUMAN_PLAYER_exist:
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

                # 只有有人类狼人或者没有人类时才打印AI狼人讨论
                if has_human_werewolf or not self.HUMAN_PLAYER_exist:
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

            if has_human_werewolf or not self.HUMAN_PLAYER_exist:
                print(f"    {wolf_id} 投票刀 {target}")
        
        # 统计票数
        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1
        
        killed = max(vote_counts.items(), key=lambda x: x[1])[0]
        if has_human_werewolf or not self.HUMAN_PLAYER_exist:
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

        seers = [pid for pid, role in self.roles.items()
                      if role == 'seer' and pid in self.alive_players]

        human_seers = [w for w in seers if w == self.human_player_id]
        has_human_seers = len(human_seers) > 0
        
        seer_id = None
        for pid, role in self.roles.items():
            if role == 'seer' and pid in self.alive_players:
                seer_id = pid
                break
        
        if not seer_id:
            if not self.HUMAN_PLAYER_exist or has_human_seers or self.human_player_role=='seer':
                print("  预言家已死亡")
            return
        
        result = await self.agents[seer_id].night_action(
            self.get_game_state(),
            self.roles  # 传入角色信息
        )
        
        if result and result.get('action') == 'check':
            target = result['target']
            check_result = result['result']

            if has_human_seers or not self.HUMAN_PLAYER_exist:
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

        witchs = [pid for pid, role in self.roles.items()
                 if role == 'witch' and pid in self.alive_players]

        human_witchs = [w for w in witchs if w == self.human_player_id]
        has_human_witchs = len(human_witchs) > 0

        for pid, role in self.roles.items():
            if role == 'witch' and pid in self.alive_players:
                witch_id = pid
                break
        
        if not witch_id:
            if not self.HUMAN_PLAYER_exist or has_human_witchs or self.human_player_role=='witch':
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
                if not self.HUMAN_PLAYER_exist or has_human_witchs:
                    print(f"  {witch_id} 使用解药救了 {target}")
                if witch_id in self.agents and hasattr(self.agents[witch_id], 'has_antidote'):
                    self.agents[witch_id].has_antidote = False
                    if not self.HUMAN_PLAYER_exist or has_human_witchs:
                        print(f"  {witch_id} 的解药已消耗")
                self.add_private_event(
                    [witch_id],
                    "witch_save",
                    {"target": target},
                    f"我使用解药救了 {target}"
                )
            
            elif action == 'poison':
                poisoned_player = target
                if not self.HUMAN_PLAYER_exist or has_human_witchs:
                    print(f"  {witch_id} 使用毒药毒了 {target}")
                if witch_id in self.agents and hasattr(self.agents[witch_id], 'has_poison'):
                    self.agents[witch_id].has_poison = False
                    if not self.HUMAN_PLAYER_exist or has_human_witchs:
                        print(f"  {witch_id} 的毒药已消耗")
                self.add_private_event(
                    [witch_id],
                    "witch_poison",
                    {"target": target},
                    f"我使用毒药毒了 {target}"
                )
        else:
            if not self.HUMAN_PLAYER_exist or has_human_witchs:
                print(f"  {witch_id} 不使用药水")
        
        return saved_player, poisoned_player
    
    async def day_phase(self):
        """白天阶段"""
        print(f"\n{'='*70}")
        print(f"第 {self.day} 天 - 讨论阶段".center(70))
        print(f"{'='*70}\n")
        
        # 记录到日志
        if ENABLE_LOGGING:
            logger = get_logger()
            if logger:
                logger.log_phase("白天讨论阶段", self.day)
                logger.log_game_state(self.day, self.alive_players, self.dead_players)
        
        # 公布昨晚死亡信息（只公布夜晚死亡的玩家）
        if self.night_deaths:
            death_msg = f"昨晚 {', '.join(self.night_deaths)} 死亡"
            print(f"[系统公告] {death_msg}")
            
            # 添加白天开始的公共事件
            self.add_public_event(
                "phase_change",
                {"phase": "day_discussion"},
                f"【系统公告】第{self.day}天白天开始。昨晚死亡：{', '.join(self.night_deaths)}。当前存活：{', '.join(self.alive_players)}"
            )
        else:
            # 第一天或者昨晚没人死
            if self.day == 1:
                print(f"[系统公告] 第一天白天，昨晚平安夜")
                self.add_public_event(
                    "phase_change",
                    {"phase": "day_discussion"},
                    f"【系统公告】第{self.day}天白天开始。昨晚平安夜。当前存活：{', '.join(self.alive_players)}"
                )
            else:
                print(f"[系统公告] 昨晚平安夜")
                self.add_public_event(
                    "phase_change",
                    {"phase": "day_discussion"},
                    f"【系统公告】第{self.day}天白天开始。昨晚平安夜。当前存活：{', '.join(self.alive_players)}"
                )
        
        # 按顺序发言
        print(f"\n[发言阶段]")
        for i, player_id in enumerate(sorted(self.alive_players), 1):
            # 【保留但优化】在发言前添加发言顺序信息
            # 通过简洁描述避免语义干扰
            already_spoke = sorted(self.alive_players)[:i-1] if i > 1 else []
            speech_order_msg = f"发言顺序：第{i}位是{player_id}。"
            if already_spoke:
                speech_order_msg += f"已发言：{', '.join(already_spoke)}。"
            
            self.add_public_event(
                "speech_turn",
                {
                    "current_speaker": player_id, 
                    "order": i, 
                    "total_alive": len(self.alive_players),
                    "already_spoke": already_spoke
                },
                speech_order_msg
            )

            if not self.HUMAN_PLAYER_exist:
                print(f"\n  第 {i} 位发言 - {player_id} ({self.roles[player_id]})：")
            else:
                print(f"\n  第 {i} 位发言 - {player_id}：")
            
            speech = await self.agents[player_id].day_speech(self.get_game_state())
            print(f"    {speech}")
            
            # 记录到日志
            if ENABLE_LOGGING:
                log_event(f"{player_id} 发言: {speech}")
            
            # 【关键】发言后立即添加到所有人的记忆
            # 这样后续发言的玩家就能看到之前的发言内容
            self.add_public_event(
                "player_speech",
                {"player_id": player_id, "content": speech, "order": i},
                f"{player_id} 发言说：{speech}"
            )
            
            # 给其他玩家一点时间"消化"这条发言（模拟真实游戏）
            await asyncio.sleep(0.5)

    # 在 game_manager.py 的 vote_phase 方法中，确保投票结果被充分记录

    async def vote_phase(self):
        """投票阶段"""
        print(f"\n{'=' * 70}")
        print(f"第 {self.day} 天 - 投票阶段".center(70))
        print(f"{'=' * 70}\n")

        # 记录到日志
        if ENABLE_LOGGING:
            logger = get_logger()
            if logger:
                logger.log_phase("投票阶段", self.day)

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
            self.vote_deaths.append(exiled)
            self.agents[exiled].is_alive = False

        # 【增强】更详细地记录投票结果
        vote_details = []
        for voter, target in votes.items():
            vote_details.append(f"{voter}→{target}")

        # 添加到所有人的记忆 - 使用更详细的描述
        self.add_public_event(
            "vote_result",
            {
                "votes": votes,
                "result": exiled,
                "vote_counts": vote_counts,
                "vote_details": vote_details,
                "vote_day": self.day  # 明确记录投票天数
            },
            f"【投票结果】第{self.day}天投票结束。{exiled}被放逐出局。" +
            f"票型统计：{', '.join([f'{target}({count}票)' for target, count in vote_counts.items()])}。" +
            f"详细投票：{', '.join(vote_details)}"
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
    
    async def run_game(self, max_days: int = 5, have_human: bool = False):
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
        
            await self.summary_phase(have_human)
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

    async def summary_phase(self, have_human: bool = False):
        """日终总结阶段"""
        print(f"\n{'-'*30} 第 {self.day} 天 日终复盘 {'-'*30}")
        log_event(f"进入第 {self.day} 天日终复盘阶段")
        
        # 所有存活玩家进行总结（并行执行以节省时间，或者为了打印清晰串行执行）
        # 这里建议串行打印，方便观察 Agent 思路
        for player_id in self.alive_players:
            agent = self.agents[player_id]
            print(f"[{player_id}] 正在复盘局势...")
            
            summary = await agent.summarize_day(self.day, {
                "alive_players": self.alive_players,
                "dead_players": self.dead_players
            })
            
            # 打印独白 (带颜色区分会更好，这里用简单的格式)
            if not have_human:
                print(f"  > 💭 {player_id} ({self.roles[player_id]}): \"{summary}\"\n")
            
            # 记录到日志
            log_agent_thinking(player_id, self.roles[player_id], "daily_summary", 
                             f"复盘内容: {summary}")
            

        
