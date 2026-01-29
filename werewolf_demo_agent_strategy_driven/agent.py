# agent.py
"""
Agent 基类和具体角色实现
"""
import json
import re
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from game_config import *
from prompts import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import AgentMemory
from config import AgentConfig


class WerewolfAgent:
    """狼人杀 Agent 基类"""
    
    def __init__(self, player_id: str, role: str, api_key: str):
        self.player_id = player_id
        self.role = role
        
        # 为每个 Agent 创建独立的 LLM 客户端
        self.llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url=SILICONFLOW_BASE_URL
        )
        
        # 初始化记忆系统
        config = AgentConfig(
            agent_id=player_id,
            game_id=GAME_ID,
            db_path=f"{MEMORY_DB_BASE_PATH}/{player_id}",
            max_memory_entries=200
        )
        self.memory = AgentMemory(config)
        
        # 角色特定数据
        self.is_alive = True
        self.checked_players = []  # 预言家查验过的玩家
        self.has_antidote = True   # 女巫解药
        self.has_poison = True     # 女巫毒药
        
        # 植入游戏规则
        self._init_game_rules()
    
    def _init_game_rules(self):
        """植入游戏规则到记忆"""
        self.memory.add_event(
            {
                "event_id": f"rules_{self.player_id}",
                "event_type": "game_rules",
                "timestamp": "",
                "data": {"rules": GAME_RULES}
            },
            text_description=GAME_RULES
        )
    
    def add_public_memory(self, event: Dict, description: str):
        """添加公共记忆"""
        self.memory.add_event(event, text_description=description)
    
    def get_system_prompt(self) -> str:
        """获取角色系统提示词"""
        prompts = {
            "werewolf": WEREWOLF_SYSTEM_PROMPT,
            "seer": SEER_SYSTEM_PROMPT,
            "witch": WITCH_SYSTEM_PROMPT,
            "villager": VILLAGER_SYSTEM_PROMPT
        }
        return prompts.get(self.role, VILLAGER_SYSTEM_PROMPT)
    
    async def call_llm(self, user_prompt: str, temperature: float = LLM_TEMPERATURE, 
                      custom_system_prompt: str = None) -> str:
        """调用 LLM"""
        try:
            system_prompt = custom_system_prompt if custom_system_prompt else self.get_system_prompt()
            response = await self.llm_client.chat.completions.create(
                model=SILICONFLOW_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[错误] {self.player_id} LLM 调用失败: {e}")
            return "{}"
    
    def parse_json_response(self, response: str) -> Dict:
        """解析 JSON 响应"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except:
            return {}
    
    async def night_action(self, game_state: Dict) -> Optional[Dict]:
        """夜晚行动（子类实现）"""
        return None
    
    async def generate_speech_queries(self, game_state: Dict) -> List[str]:
        """生成发言前需要查询的问题"""
        system_prompt = f"""你是狼人杀游戏中的 {self.player_id}，身份是 {self.role}。
现在轮到你发言，你需要先决定要查询哪些记忆信息来帮助你发言。

请生成3-5个查询问题，这些问题应该：
1. 帮助你了解当前局势（谁死了、谁跳身份等）
2. 找出可疑的玩家
3. 验证其他玩家的发言
4. 根据你的角色制定策略

输出格式（JSON数组）：
["问题1", "问题2", "问题3", ...]

只输出JSON数组，不要有其他内容。"""
        
        user_prompt = f"""当前游戏状态：
- 第 {game_state['day']} 天，白天讨论阶段
- 存活玩家: {', '.join(game_state['alive_players'])}
- 死亡玩家: {', '.join(game_state['dead_players'])}
- 我是 {self.player_id}，身份是 {self.role}

请生成你需要查询的问题："""
        
        response = await self.call_llm(user_prompt, temperature=0.7, custom_system_prompt=system_prompt)
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
            else:
                queries = json.loads(response)
            
            if isinstance(queries, list) and len(queries) > 0:
                return queries
        except Exception as e:
            print(f"[警告] {self.player_id} 生成查询问题失败: {e}")
        
        # 默认查询
        return [
            "昨晚谁死了？",
            "谁跳了预言家？",
            "有人被怀疑吗？"
        ]
    
    async def day_speech(self, game_state: Dict) -> str:
        """白天发言"""
        # 步骤1: 生成查询问题
        print(f"  [{self.player_id}] 正在生成查询问题...")
        queries = await self.generate_speech_queries(game_state)
        print(f"  [{self.player_id}] 生成了 {len(queries)} 个查询问题")
        
        # 步骤2: 查询记忆
        all_memories = []
        for query in queries:
            memory_context = self.memory.get_relevant_context(
                query=query,
                top_k=3,
                max_chars=500
            )
            all_memories.append(f"关于'{query}':\n{memory_context}")
        
        combined_memory = "\n\n".join(all_memories)
        
        # 步骤3: 生成角色特定信息
        role_specific_info = self.get_role_specific_info_for_day(game_state)
        
        # 步骤4: 生成发言
        prompt = DAY_SPEECH_PROMPT.format(
            player_id=self.player_id,
            day=game_state['day'],
            alive_players=', '.join(game_state['alive_players']),
            alive_count=len(game_state['alive_players']),
            dead_players=', '.join(game_state['dead_players']),
            role=self.role,
            role_specific_info=role_specific_info,
            memory_context=combined_memory
        )
        
        response = await self.call_llm(prompt)
        result = self.parse_json_response(response)
        
        return result.get('speech', '我还在思考...')
    
    def get_role_specific_info_for_day(self, game_state: Dict) -> str:
        """获取角色特定信息（用于白天发言）"""
        return ""  # 基类返回空字符串，子类可以重写
    
    async def generate_vote_queries(self, game_state: Dict) -> List[str]:
        """生成投票前需要查询的问题"""
        alive_others = [p for p in game_state['alive_players'] if p != self.player_id]
        
        system_prompt = f"""你是狼人杀游戏中的 {self.player_id}，身份是 {self.role}。
现在是投票阶段，你需要决定投票给谁。

在做出投票决策前，你需要先查询记忆来分析每个玩家。
请生成4-6个查询问题，这些问题应该帮助你：
1. 了解每个玩家的发言内容
2. 分析每个玩家的行为是否可疑
3. 验证预言家的查验结果
4. 找出逻辑矛盾
5. 根据你的角色制定投票策略

输出格式（JSON数组）：
["问题1", "问题2", "问题3", ...]

只输出JSON数组，不要有其他内容。"""
        
        user_prompt = f"""当前游戏状态：
- 第 {game_state['day']} 天，投票阶段
- 存活玩家: {', '.join(game_state['alive_players'])}
- 死亡玩家: {', '.join(game_state['dead_players'])}
- 我是 {self.player_id}，身份是 {self.role}
- 需要分析的玩家: {', '.join(alive_others)}

请生成你需要查询的问题来辅助投票决策："""
        
        response = await self.call_llm(user_prompt, temperature=0.7, custom_system_prompt=system_prompt)
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
            else:
                queries = json.loads(response)
            
            if isinstance(queries, list) and len(queries) > 0:
                return queries
        except Exception as e:
            print(f"[警告] {self.player_id} 生成投票查询问题失败: {e}")
        
        # 默认查询
        return [
            "每个玩家都说了什么？",
            "谁的发言最可疑？",
            "谁最可能是狼人？"
        ]
    
    async def vote(self, game_state: Dict) -> str:
        """投票"""
        # 步骤1: 生成查询问题
        print(f"  [{self.player_id}] 正在生成投票查询问题...")
        queries = await self.generate_vote_queries(game_state)
        print(f"  [{self.player_id}] 生成了 {len(queries)} 个查询问题")
        
        # 步骤2: 查询记忆
        all_memories = []
        for query in queries:
            memory_context = self.memory.get_relevant_context(
                query=query,
                top_k=5,
                max_chars=600
            )
            all_memories.append(f"关于'{query}':\n{memory_context}")
        
        combined_memory = "\n\n".join(all_memories)
        
        # 步骤3: 生成角色特定信息
        role_specific_info = self.get_role_specific_info_for_vote(game_state)
        
        # 步骤4: 生成投票决策
        prompt = VOTE_PROMPT.format(
            player_id=self.player_id,
            day=game_state['day'],
            alive_players=', '.join([p for p in game_state['alive_players'] if p != self.player_id]),
            role=self.role,
            role_specific_info=role_specific_info,
            memory_context=combined_memory
        )
        
        response = await self.call_llm(prompt)
        result = self.parse_json_response(response)
        
        target = result.get('target')
        
        # 验证投票目标（只验证基本规则：不能投自己，必须是存活玩家）
        if target and target in game_state['alive_players'] and target != self.player_id:
            return target
        
        # 默认投第一个不是自己的玩家
        others = [p for p in game_state['alive_players'] if p != self.player_id]
        return others[0] if others else self.player_id
    
    def get_role_specific_info_for_vote(self, game_state: Dict) -> str:
        """获取角色特定信息（用于投票）"""
        return ""  # 基类返回空字符串，子类可以重写


class WerewolfAgentRole(WerewolfAgent):
    """狼人 Agent"""
    
    def __init__(self, player_id: str, role: str, api_key: str):
        super().__init__(player_id, role, api_key)
        self.teammates = []  # 存储队友列表
    
    def set_teammates(self, teammates: List[str]):
        """设置狼人队友"""
        self.teammates = teammates
    
    def get_role_specific_info_for_day(self, game_state: Dict) -> str:
        """狼人白天发言时的特定信息"""
        alive_teammates = [t for t in self.teammates if t in game_state['alive_players']]
        dead_teammates = [t for t in self.teammates if t in game_state['dead_players']]
        
        info = f"""【狼人身份提醒】：
- 你是狼人，你的目标是帮助狼人阵营获胜
- 你的狼人队友：{', '.join(self.teammates)}
- 存活的队友：{', '.join(alive_teammates) if alive_teammates else '无（你是最后一个狼人）'}
- 已死亡的队友：{', '.join(dead_teammates) if dead_teammates else '无'}

【发言策略建议】：
- 最重要的是：不要暴露你是狼人
- 要伪装成好人，发言要像一个好人的思考方式
- 通常情况下，不要主动攻击或怀疑队友
- 但在某些情况下，为了隐藏身份，你可以：
  * 适当质疑队友（如果队友已经被怀疑）
  * 与队友保持距离（避免被发现关联）
  * 在必要时牺牲队友来保护自己
- 可以适当怀疑其他非狼人玩家
- 可以伪装成神职或村民
- 注意不要与队友的发言产生明显矛盾"""
        
        return info
    
    def get_role_specific_info_for_vote(self, game_state: Dict) -> str:
        """狼人投票时的特定信息"""
        alive_teammates = [t for t in self.teammates if t in game_state['alive_players']]
        non_werewolf_players = [p for p in game_state['alive_players'] 
                               if p != self.player_id and p not in self.teammates]
        
        info = f"""【狼人身份提醒】：
- 你是狼人，你的队友：{', '.join(self.teammates)}
- 存活的队友：{', '.join(alive_teammates) if alive_teammates else '无'}

【投票策略建议】：
- 通常情况下，应该投票给非狼人玩家：{', '.join(non_werewolf_players)}
- 但如果为了隐藏身份、避免暴露，你也可以选择投票给队友
- 投票给队友的情况：
  * 当队友已经被大多数人怀疑，你跟票可以避免暴露
  * 当你需要表现得像好人一样
  * 当牺牲队友可以保护你自己时
- 投票理由要合理，要像一个好人的思考方式
- 最重要的是：不要暴露你是狼人"""
        
        return info
    
    async def night_discussion(self, round_num: int, total_rounds: int, 
                              teammates: List[str], previous_discussion: str,
                              game_state: Dict) -> Dict:
        """狼人夜晚讨论"""
        # 根据天数生成上下文提示
        day = game_state.get('day', 1)
        if day == 1:
            day_context = """【第一晚特别提示】：
- 这是游戏的第一个夜晚
- 还没有进行过白天发言阶段
- 你们只能根据座位号和角色配置进行推测
- 不要提及任何玩家的"发言"或"行为"（因为还没发生）
- 可以讨论：哪个座位号可能是神职、随机选择击杀目标等"""
        else:
            day_context = f"""【第{day}晚提示】：
- 已经进行过{day-1}天的白天讨论
- 可以根据之前的发言和投票记录分析
- 结合记忆中的信息做出判断"""
        
        prompt = WEREWOLF_NIGHT_DISCUSSION_PROMPT.format(
            player_id=self.player_id,
            day=day,
            round=round_num,
            total_rounds=total_rounds,
            alive_players=', '.join(game_state['alive_players']),
            teammates=', '.join(teammates),
            day_context=day_context,
            previous_discussion=previous_discussion
        )
        
        response = await self.call_llm(prompt)
        result = self.parse_json_response(response)
        
        return {
            "speech": result.get('speech', '我同意队友的意见'),
            "suggested_target": result.get('suggested_target')
        }
    
    async def night_vote(self, discussion_summary: str, game_state: Dict, all_roles: Dict) -> str:
        """狼人投票刀人"""
        # 过滤掉狼人队友
        non_werewolf_players = [p for p in game_state['alive_players'] 
                               if all_roles.get(p) != 'werewolf']
        
        prompt = WEREWOLF_VOTE_PROMPT.format(
            player_id=self.player_id,
            discussion_summary=discussion_summary,
            alive_players=', '.join(non_werewolf_players)
        )
        
        response = await self.call_llm(prompt)
        result = self.parse_json_response(response)
        
        target = result.get('target')
        # 确保不刀队友
        if target and target in game_state['alive_players'] and all_roles.get(target) != 'werewolf':
            return target
        
        # 默认刀第一个非狼人
        for player in game_state['alive_players']:
            if all_roles.get(player) != 'werewolf':
                return player
        
        return game_state['alive_players'][0]


class SeerAgentRole(WerewolfAgent):
    """预言家 Agent"""
    
    async def night_action(self, game_state: Dict, all_roles: Dict = None) -> Optional[Dict]:
        """预言家查验"""
        # 查询记忆
        memory_context = self.memory.get_relevant_context(
            query="哪些玩家可疑，需要查验",
            top_k=5,
            max_chars=800
        )
        
        # 根据天数生成上下文提示
        day = game_state.get('day', 1)
        if day == 1:
            day_context = """【第一晚特别提示】：
- 这是游戏的第一个夜晚
- 还没有进行过白天发言
- 你只能根据座位号随机选择查验目标
- 建议查验中间座位号的玩家"""
        else:
            day_context = f"""【第{day}晚提示】：
- 已经进行过{day-1}天的白天讨论
- 可以根据发言和投票记录选择可疑玩家查验"""
        
        prompt = SEER_CHECK_PROMPT.format(
            player_id=self.player_id,
            day=day,
            alive_players=', '.join([p for p in game_state['alive_players'] 
                                    if p != self.player_id and p not in self.checked_players]),
            checked_players=', '.join(self.checked_players) if self.checked_players else '无',
            day_context=day_context,
            memory_context=memory_context
        )
        
        response = await self.call_llm(prompt)
        result = self.parse_json_response(response)
        
        target = result.get('target')
        if target and target in game_state['alive_players'] and target != self.player_id:
            self.checked_players.append(target)
            # 返回查验结果（需要 all_roles）
            is_werewolf = all_roles.get(target) == 'werewolf' if all_roles else False
            return {
                "action": "check",
                "target": target,
                "result": "狼人" if is_werewolf else "好人"
            }
        
        return None


class WitchAgentRole(WerewolfAgent):
    """女巫 Agent"""
    
    async def night_action(self, game_state: Dict) -> Optional[Dict]:
        """女巫使用药水"""
        killed_player = game_state.get('killed_tonight')
        
        # 查询记忆
        memory_context = self.memory.get_relevant_context(
            query="谁是狼人，谁应该被毒",
            top_k=5,
            max_chars=800
        )
        
        # 根据天数生成上下文提示
        day = game_state.get('day', 1)
        if day == 1:
            day_context = """【第一晚特别提示】：
- 这是游戏的第一个夜晚
- 还没有进行过白天发言
- 通常第一晚不建议使用解药（自救除外）
- 毒药更不建议第一晚使用（信息不足）"""
        else:
            day_context = f"""【第{day}晚提示】：
- 已经进行过{day-1}天的白天讨论
- 可以根据发言和投票记录判断是否使用药水
- 解药：如果被杀的是重要好人可以考虑救
- 毒药：如果确定某人是狼人可以考虑毒"""
        
        prompt = WITCH_ACTION_PROMPT.format(
            player_id=self.player_id,
            day=day,
            killed_player=killed_player if killed_player else '无',
            has_antidote='是' if self.has_antidote else '否',
            has_poison='是' if self.has_poison else '否',
            alive_players=', '.join(game_state['alive_players']),
            day_context=day_context,
            memory_context=memory_context
        )
        
        response = await self.call_llm(prompt)
        result = self.parse_json_response(response)
        
        action = result.get('action', 'none')
        
        if action == 'save' and self.has_antidote and killed_player:
            self.has_antidote = False
            return {"action": "save", "target": killed_player}
        
        elif action == 'poison' and self.has_poison:
            target = result.get('target')
            if target and target in game_state['alive_players']:
                self.has_poison = False
                return {"action": "poison", "target": target}
        
        return None


def create_agent(player_id: str, role: str, api_key: str) -> WerewolfAgent:
    """创建 Agent 工厂函数"""
    if role == "werewolf":
        return WerewolfAgentRole(player_id, role, api_key)
    elif role == "seer":
        return SeerAgentRole(player_id, role, api_key)
    elif role == "witch":
        return WitchAgentRole(player_id, role, api_key)
    else:
        return WerewolfAgent(player_id, role, api_key)
