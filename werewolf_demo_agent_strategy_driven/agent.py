# agent.py
"""
Agent 基类和具体角色实现
"""
import json
import re
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
from sympy import true

from game_config import *
from prompts import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import AgentMemory
from config import AgentConfig
from logger import log_agent_thinking, log_memory_retrieval
from human_interface import HumanInterface, MessageType


class WerewolfAgent:
    """狼人杀 Agent 基类"""
    
    def __init__(self, player_id: str, role: str, api_key: str, character: str) -> None:
        self.player_id = player_id
        self.role = role
        self.character = character
        
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
            max_memory_entries=200,
            verbose=VERBOSE_MODE  # 从配置文件读取日志模式
        )
        self.memory = AgentMemory(config)
        
        # 角色特定数据
        self.is_alive = True
        self.checked_players = []  # 预言家查验过的玩家
        self.has_antidote = True   # 女巫解药
        self.has_poison = True     # 女巫毒药
        
        # 植入游戏规则即性格信息
        self._init_game_rules()
        self._init_character_info()
    
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

    def _init_character_info(self):
        """植入性格信息到记忆"""
        character_info = CHARACTER_CONFIG.get(self.character, CHARACTER_CONFIG["character_01"])
        character_desc = f"【性格特征】\n姓名：{character_info['name']}\n描述：{character_info['description']}\n发言风格：{character_info['speech_style']}"

        self.memory.add_event(
            {
                "event_id": f"character_{self.player_id}",
                "event_type": "character_info",
                "timestamp": "",
                "data": character_info
            },
            text_description=character_desc
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

        base_prompt = prompts.get(self.role, VILLAGER_SYSTEM_PROMPT)

        # 添加性格描述
        character_info = CHARACTER_CONFIG.get(self.character, CHARACTER_CONFIG["character_01"])
        character_prompt = f"\n\n【你的性格特点】\n你是{character_info['name']}：{character_info['description']}\n发言时请体现这种风格：{character_info['speech_style']}"

        return base_prompt + character_prompt

    async def call_llm(self, user_prompt: str, temperature: float = LLM_TEMPERATURE,
                       custom_system_prompt: str = None) -> str:
        """调用 LLM"""
        try:
            system_prompt = custom_system_prompt if custom_system_prompt else self.get_system_prompt()

            # 记录完整的prompt到日志
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

            #根据性格调整temperature（某些性格可能更随机）
            adjusted_temp = self._adjust_temperature_by_character(temperature)

            response = await self.llm_client.chat.completions.create(
                model=SILICONFLOW_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=adjusted_temp,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT
            )

            response_text = response.choices[0].message.content

            # 记录LLM调用到日志
            if ENABLE_LOGGING:
                log_agent_thinking(
                    player_id=self.player_id,
                    role=self.role,
                    character=self.character,
                    context="LLM调用",
                    llm_prompt=full_prompt,
                    llm_response=response_text
                )

            return response_text
        except Exception as e:
            print(f"[错误] {self.player_id} LLM 调用失败: {e}")
            return "{}"

    def _adjust_temperature_by_character(self, base_temp: float) -> float:
        """根据性格调整temperature"""
        character_info = CHARACTER_CONFIG.get(self.character, CHARACTER_CONFIG["character_01"])
        character_name = character_info["name"]

        # 根据性格调整随机性
        if "激情" in character_name or "冲动" in character_name:
            return min(base_temp + 0.1, 1.0)  # 更随机
        elif "理性" in character_name or "分析" in character_name:
            return max(base_temp - 0.1, 0.1)  # 更确定
        elif "幽默" in character_name:
            return min(base_temp + 0.05, 1.0)  # 稍随机
        else:
            return base_temp



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

    def get_character_context(self) -> str:
        """获取性格上下文信息"""
        character_info = CHARACTER_CONFIG.get(self.character, CHARACTER_CONFIG["character_01"])
        return f"\n\n【你的性格特点】\n你是{character_info['name']}：{character_info['description']}\n发言时请体现这种风格：{character_info['speech_style']}"
    
    async def day_speech(self, game_state: Dict) -> str:
        """白天发言"""
        # 步骤1: 生成查询问题
        if VERBOSE_MODE:
            print(f"  [{self.player_id}] 正在生成查询问题...")
        queries = await self.generate_speech_queries(game_state)
        if VERBOSE_MODE:
            print(f"  [{self.player_id}] 生成了 {len(queries)} 个查询问题")
            for i, q in enumerate(queries, 1):
                print(f"    {i}. {q}")
        
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
        
        # 记录记忆检索到日志
        if ENABLE_LOGGING:
            log_memory_retrieval(self.player_id, queries, all_memories)
        
        # 步骤3: 生成角色特定信息
        role_specific_info = self.get_role_specific_info_for_day(game_state)

        # 步骤4: 添加性格信息
        character_info = self.get_character_context()
        
        # 步骤5: 生成发言
        prompt = DAY_SPEECH_PROMPT.format(
            player_id=self.player_id,
            day=game_state['day'],
            alive_players=', '.join(game_state['alive_players']),
            alive_count=len(game_state['alive_players']),
            dead_players=', '.join(game_state['dead_players']),
            evening_deaths=', '.join(game_state['evening_deaths']),
            vote_deaths=', '.join(game_state['vote_deaths']),
            role=self.role,
            role_specific_info=role_specific_info,
            character_info=character_info,
            memory_context=combined_memory
        )
        
        response = await self.call_llm(prompt)
        result = self.parse_json_response(response)
        
        speech = result.get('speech', '我还在思考...')
        
        # 记录最终决策到日志
        if ENABLE_LOGGING:
            log_agent_thinking(
                player_id=self.player_id,
                role=self.role,
                character=self.character,
                context="白天发言",
                final_decision=f"发言内容: {speech}"
            )
        
        return speech
    
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
        if VERBOSE_MODE:
            print(f"  [{self.player_id}] 正在生成投票查询问题...")
        queries = await self.generate_vote_queries(game_state)
        if VERBOSE_MODE:
            print(f"  [{self.player_id}] 生成了 {len(queries)} 个查询问题")
            for i, q in enumerate(queries, 1):
                print(f"    {i}. {q}")
        
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
        
        # 记录记忆检索到日志
        if ENABLE_LOGGING:
            log_memory_retrieval(self.player_id, queries, all_memories)
        
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
        reason = result.get('reason', '无')
        
        # 记录投票决策到日志
        if ENABLE_LOGGING:
            log_agent_thinking(
                player_id=self.player_id,
                role=self.role,
                character=self.character,
                context="投票决策",
                final_decision=f"投票目标: {target}\n理由: {reason}"
            )
        
        # 验证投票目标（只验证基本规则：不能投自己，必须是存活玩家）
        if target and target in game_state['alive_players'] and target != self.player_id:
            return target
        
        # 默认投第一个不是自己的玩家
        others = [p for p in game_state['alive_players'] if p != self.player_id]
        return others[0] if others else self.player_id
    
    def get_role_specific_info_for_vote(self, game_state: Dict) -> str:
        """获取角色特定信息（用于投票）"""
        return ""  # 基类返回空字符串，子类可以重写




    async def summarize_day(self, day: int, game_state: Dict[str, Any]) -> str:
        """
        [增强版] 进行日终总结
        """
        # 1. 多重查询策略，确保覆盖所有关键信息
        queries = [
            f"第{day}天 投票结果 放逐 谁被投票出局",
            f"第{day}天 发言 谁说了什么",
            f"第{day}天 白天讨论 玩家观点",
            f"第{day}天 死亡 谁死了",
            f"第{day}天 狼人刀人 女巫用药",
            f"个人日终总结-{self.player_id}",
            f"是预言家",
            f"是女巫",
            f"是狼人",
            f"当前存活"
        ]

        all_memories = []
        for query in queries:
            # 检索相关记忆，指定按当天过滤，并增加时间权重
            recent_memories = self.memory.retrieve_recent_memories(
                query=query,
                n_results=10,  # 每个查询检索10条
                day_filter=day,
                recency_weight=0.4,  # 增加时间权重
                exclude_system_events = True,
            )
            all_memories.extend(recent_memories)

        # 去重并限制数量
        unique_memories = []
        seen_texts = set()
        for memory in all_memories:
            # 简单的文本去重
            if memory not in seen_texts and len(memory) > 20:  # 过滤太短的文本
                unique_memories.append(memory)
                seen_texts.add(memory)

        # 按相关性排序（假设开头相同的更相关）
        def relevance_score(text):
            # 检查是否包含关键词
            keywords = ["投票", "放逐", "投给", "结果", "查验", "死亡", "刀", "救", "毒"]
            score = sum(1 for kw in keywords if kw in text)
            # 时间因素：更详细的描述可能更相关
            score += len(text) * 0.001
            return score

        unique_memories.sort(key=relevance_score, reverse=True)
        memory_context = "\n".join([f"- {m}" for m in unique_memories[:20]])  # 最多20条

        # 2. 构建 Prompt - 提供更具体的指导
        prompt = DAILY_SUMMARY_PROMPT.format(
            day=day,
            player_id=self.player_id,
            role=self.role,
            memory_context=memory_context if memory_context else "今天没有特别值得注意的记忆。"
        )

        # 3. 调用 LLM
        try:
            response = await self.call_llm(prompt)
            result = self.parse_json_response(response)

            content = result.get('summary_content', '无总结')
            suspects = result.get('key_suspects', [])

            # 4. 将总结存入记忆，赋予高重要性
            summary_text = f"【个人日终总结-{self.player_id}】第{day}天复盘：{content}。当前怀疑对象：{', '.join(suspects)}。"

            self.memory.add_memory(
                content={
                    "summary": content,
                    "suspects": suspects,
                    "day": day,
                    "vote_result": self._extract_vote_result(unique_memories)  # 提取投票结果
                },
                text=summary_text,
                event_type="daily_summary",
                day=day,
                phase="daily_summary",
                importance=0.7  # 进一步提高重要性
            )

            # 5. 特别记录投票结果（如果有）
            vote_results = self._extract_vote_result_text(unique_memories)
            if vote_results:
                self.memory.add_memory(
                    content={"vote_summary": vote_results, "day": day},
                    text=f"【投票结果总结】第{day}天：{vote_results}",
                    event_type="vote_summary",
                    day=day,
                    phase="daily_summary",
                    importance=0.95  # 投票结果非常重要
                )

            return content

        except Exception as e:
            print(f"Error in summarize_day for {self.player_id}: {e}")
            return f"思考被打断了... 但我记得今天的主要事件是：{self._extract_quick_summary(unique_memories)}"

    def _extract_vote_result(self, memories: List[str]) -> Dict:
        """从记忆中提取投票结果"""
        result = {}
        for text in memories:
            if "投票" in text and "投给" in text:
                # 简单的文本解析
                import re
                match = re.search(r'(\w+)\s*投票?给\s*(\w+)', text)
                if match:
                    voter, target = match.groups()
                    if voter.startswith('player_'):
                        result[voter] = target
        return result

    def _extract_vote_result_text(self, memories: List[str]) -> str:
        """从记忆中提取投票结果文本"""
        vote_texts = []
        for text in memories:
            if "投票结果" in text or "放逐" in text:
                vote_texts.append(text)
        return " | ".join(vote_texts[:3])  # 最多3条

    def _extract_quick_summary(self, memories: List[str]) -> str:
        """快速提取摘要"""
        key_events = []
        for text in memories[:5]:  # 只看前5条
            if any(kw in text for kw in ["死亡", "放逐", "查验", "救", "毒"]):
                key_events.append(text[:50] + "...")
        return "；".join(key_events) if key_events else "没有特别事件"

    async def night_discussion(self, round_num: int, total_rounds: int,
                               teammates: List[str], previous_discussion: str,
                               game_state: Dict) -> Dict:
        """狼人夜晚讨论"""
        # 根据天数生成上下文提示
        day = game_state.get('day', 1)

        # 【新增】在夜晚讨论前检索神职相关的记忆
        seer_witch_memories = []
        if day > 1:  # 第一天夜晚没有白天的记忆
            # 关键词检索，寻找跳神职的玩家
            keywords = ["预言家", "女巫", "查验", "验", "解药", "救药", "毒药", "跳预言家", "跳女巫", "神职"]

            for keyword in keywords:
                recent_memories = self.memory.retrieve_recent_memories(
                    query=keyword,
                    n_results=5,  # 每个关键词检索5条
                    day_filter=day - 1,  # 检索前一天白天的记忆
                    recency_weight=0.4,  # 增加时间权重
                    exclude_system_events=True,
                )
                seer_witch_memories.extend(recent_memories)

        # 去重
        unique_seer_witch_memories = []
        seen_texts = set()
        for memory in seer_witch_memories:
            if memory not in seen_texts and len(memory) > 20:
                unique_seer_witch_memories.append(memory)
                seen_texts.add(memory)

        # 按相关性排序
        def seer_witch_score(text):
            keywords = ["预言家", "女巫", "查验", "验", "解药", "毒药"]
            score = sum(2 for kw in keywords if kw in text)  # 神职相关关键词权重更高
            score += len(text) * 0.001
            return score

        unique_seer_witch_memories.sort(key=seer_witch_score, reverse=True)
        seer_witch_context = "\n".join([f"- {m}" for m in unique_seer_witch_memories[:10]])

        if seer_witch_context:
            print(f"  [{self.player_id}] 检索到神职相关信息:")
            for memory in unique_seer_witch_memories[:5]:
                print(f"    {memory[:80]}...")

        if day == 1:
            day_context = """【第一晚特别提示】：
    - 这是游戏的第一个夜晚
    - 还没有进行过白天发言阶段
    - 你们只能根据座位号和角色配置进行推测
    - 不要提及任何玩家的"发言"或"行为"（因为还没发生）
    - 可以讨论：哪个座位号可能是神职、随机选择击杀目标等"""
        else:
            day_context = f"""【第{day}晚提示】：
    - 已经进行过{day - 1}天的白天讨论
    - 可以根据之前的发言和投票记录分析
    - 结合记忆中的信息做出判断"""

        # 如果有神职相关信息，添加到day_context中
        if seer_witch_context:
            day_context += f"\n\n【白天神职相关信息】：\n{seer_witch_context}\n\n【重要提示】：如果发现有玩家明确声称自己是预言家或女巫，并且不是你的队友，优先考虑刀掉他们！"

        prompt = WEREWOLF_NIGHT_DISCUSSION_PROMPT.format(
            player_id=self.player_id,
            day=day,
            round=round_num,
            total_rounds=total_rounds,
            alive_players=', '.join(game_state['alive_players']),
            teammates=', '.join(teammates),
            day_context=day_context,
            character_info=self.get_character_context(),
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

class WerewolfAgentRole(WerewolfAgent):
    """狼人 Agent"""
    
    def __init__(self, player_id: str, role: str, api_key: str, character: str = "character_01"):
        super().__init__(player_id, role, api_key, character)
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

        # 【新增】在夜晚讨论前检索神职相关的记忆
        seer_witch_memories = []
        if day > 1:  # 第一天夜晚没有白天的记忆
            # 关键词检索，寻找跳神职的玩家
            keywords = ["预言家", "女巫", "查验", "验", "解药", "救药", "毒药", "跳预言家", "跳女巫", "神职",
                            "查杀", "金水", "银水"]

            for keyword in keywords:
                recent_memories = self.memory.retrieve_recent_memories(
                    query=keyword,
                    n_results=5,  # 每个关键词检索5条
                    day_filter=day - 1,  # 检索前一天白天的记忆
                    recency_weight=0.4,  # 增加时间权重
                    exclude_system_events=True,
                )
                seer_witch_memories.extend(recent_memories)

        # 去重
        unique_seer_witch_memories = []
        seen_texts = set()
        for memory in seer_witch_memories:
            if memory not in seen_texts and len(memory) > 20:
                unique_seer_witch_memories.append(memory)
                seen_texts.add(memory)

        # 按相关性排序
        def seer_witch_score(text):
            keywords = ["预言家", "女巫", "查验", "验", "解药", "毒药"]
            score = sum(2 for kw in keywords if kw in text)  # 神职相关关键词权重更高
            score += len(text) * 0.001
            return score

        unique_seer_witch_memories.sort(key=seer_witch_score, reverse=True)
        seer_witch_context = "\n".join([f"- {m}" for m in unique_seer_witch_memories[:10]])

        if seer_witch_context and VERBOSE_MODE:
            print(f"  [{self.player_id}] 检索到神职相关信息:")
            for memory in unique_seer_witch_memories[:5]:
                print(f"    {memory[:80]}...")

        if day == 1:
            day_context = """【第一晚特别提示】：
    - 这是游戏的第一个夜晚
    - 还没有进行过白天发言阶段
    - 你们只能根据座位号和角色配置进行推测
    - 不要提及任何玩家的"发言"或"行为"（因为还没发生）
    - 可以讨论：哪个座位号可能是神职、随机选择击杀目标等"""
        else:
            day_context = f"""【第{day}晚提示】：
    - 已经进行过{day - 1}天的白天讨论
    - 可以根据之前的发言和投票记录分析
    - 结合记忆中的信息做出判断"""

        # 如果有神职相关信息，添加到day_context中
        if seer_witch_context:
            day_context += f"\n\n【白天神职相关信息】：\n{seer_witch_context}\n\n【重要提示】：如果发现有玩家明确声称自己是预言家或女巫，并且不是你的队友，优先考虑刀掉他们！预言家的优先级高于女巫。"

        prompt = WEREWOLF_NIGHT_DISCUSSION_PROMPT.format(
            player_id=self.player_id,
            day=day,
            round=round_num,
            total_rounds=total_rounds,
            alive_players=', '.join(game_state['alive_players']),
            teammates=', '.join(teammates),
            day_context=day_context,
            character_info=self.get_character_context(),
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

    def __init__(self, player_id: str, role: str, api_key: str, character: str = "character_01"):  # 添加character参数
        super().__init__(player_id, role, api_key, character)  # 传递给父类
    
    def get_role_specific_info_for_day(self, game_state: Dict) -> str:
        """预言家白天发言时的特定信息"""
        if not self.checked_players:
            return """【预言家身份提醒】：
- 你是预言家，但还没有查验过任何玩家
- 第一天白天不要暴露身份，除非有明确的狼人目标
- 可以先观察局势，等有更多信息再跳身份"""
        
        # 构建查验记录
        check_records = []
        for player in self.checked_players:
            # 从记忆中查找查验结果
            for entry in self.memory.entries:
                if (entry.event_type == "seer_check" and 
                    entry.content.get("target") == player):
                    result = entry.content.get("result", "未知")
                    check_records.append(f"  - {player}: {result}")
                    break
        
        info = f"""【预言家身份提醒】：
- 你是预言家
- 你已查验过的玩家：
{chr(10).join(check_records) if check_records else '  - 无'}

【发言策略建议】：
- 如果查到狼人，可以考虑跳身份公布查验结果
- 如果还没查到狼人，可以继续隐藏身份观察
- 公布查验结果时要明确说明：
  * "我是预言家"
  * "第X夜查验了player_Y，结果是狼人/好人"
  * 不要编造没有查验过的结果
- 注意：只公布你实际查验过的结果，不要虚构"""
        
        return info
    
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

    def __init__(self, player_id: str, role: str, api_key: str, character: str = "character_01"):  # 添加character参数
        super().__init__(player_id, role, api_key, character)  # 传递给父类
    
    def get_role_specific_info_for_day(self, game_state: Dict) -> str:
        """女巫白天发言时的特定信息"""
        # 检查药水使用记录
        saved_players = []
        poisoned_players = []
        
        for entry in self.memory.entries:
            if entry.event_type == "witch_save":
                target = entry.content.get("target")
                if target:
                    saved_players.append(target)
            elif entry.event_type == "witch_poison":
                target = entry.content.get("target")
                if target:
                    poisoned_players.append(target)
        
        info = f"""【女巫身份提醒】：
- 你是女巫
- 解药剩余：{'是' if self.has_antidote else '否（已使用）'}
- 毒药剩余：{'是' if self.has_poison else '否（已使用）'}"""
        
        if saved_players:
            info += f"\n- 你救过的玩家：{', '.join(saved_players)}"
        if poisoned_players:
            info += f"\n- 你毒过的玩家：{', '.join(poisoned_players)}"
        
        info += """

【发言策略建议】：
- 如果救过人，可以考虑跳身份公布银水（被救的玩家）
- 如果毒过人，可以说明理由
- 不要轻易暴露身份，除非有明确的战略价值
- 公布信息时要准确：
  * "我是女巫"
  * "第X夜救了player_Y"或"第X夜毒了player_Z"
  * 不要编造没有使用过的药水"""
        
        return info
    
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

def create_agent(player_id: str, role: str, api_key: str, character: str = "character_01") -> WerewolfAgent:
    """创建 Agent 工厂函数"""
    if role == "werewolf":
        return WerewolfAgentRole(player_id, role, api_key, character)
    elif role == "seer":
        return SeerAgentRole(player_id, role, api_key, character)
    elif role == "witch":
        return WitchAgentRole(player_id, role, api_key, character)
    else:
        return WerewolfAgent(player_id, role, api_key, character)


# 添加人类玩家Agent类
class HumanAgent(WerewolfAgent):
    """人类玩家 Agent"""

    def __init__(self, player_id: str, role: str, api_key: str, character: str = "character_01"):
        super().__init__(player_id, role, api_key, character)
        self.interface = HumanInterface(player_id, role, character)

        # 人类玩家不需要LLM客户端（但保留以兼容）
        self.llm_client = None

        # 初始化角色特定属性
        if role == "seer":
            self.checked_players = []
        elif role == "witch":
            self.has_antidote = True
            self.has_poison = True
        elif role == "werewolf":
            self.teammates = []

        # 为人类玩家添加初始信息
        self.interface.add_message(
            MessageType.SYSTEM,
            f"欢迎来到狼人杀游戏！你是 {player_id}，角色是 {role}。",
            phase="initialization"
        )

        self.interface.add_message(
            MessageType.SYSTEM,
            f"你的性格是: {CHARACTER_CONFIG[character]['name']}",
            phase="initialization"
        )

    def add_public_memory(self, event: Dict, description: str):
        """添加公共记忆，同时显示给人类玩家"""
        super().add_public_memory(event, description)

        # 根据事件类型决定显示方式
        event_type = event.get("event_type", "")

        # 过滤发言顺序信息（减少信息过载）
        if event_type == "speech_turn":
            # 只显示当前发言者，不显示完整的发言顺序
            data = event.get("data", {})
            current_speaker = data.get("current_speaker")
            if current_speaker and current_speaker != self.player_id:
                self.interface.add_message(
                    MessageType.PUBLIC,
                    f"现在轮到 {current_speaker} 发言",
                    phase=event.get("phase", "day"),
                    sender="系统"
                )
        elif event_type == "player_speech":
            # 玩家发言
            data = event.get("data", {})
            speaker = data.get("player_id")
            content = data.get("content", "")

            if speaker != self.player_id:  # 不显示自己的发言（会在发言时看到）
                self.interface.add_message(
                    MessageType.PUBLIC,
                    f"{speaker} 发言: {content}",
                    phase=event.get("phase", "day"),
                    sender=speaker
                )
        elif event_type == "vote_result":
            # 投票结果
            data = event.get("data", {})
            result = data.get("result", "")
            self.interface.add_message(
                MessageType.PUBLIC,
                f"投票结果: {result} 被放逐",
                phase=event.get("phase", "voting"),
                sender="系统"
            )
        elif event_type == "phase_change":
            # 阶段变更
            data = event.get("data", {})
            phase = data.get("phase", "")
            self.interface.add_message(
                MessageType.SYSTEM,
                description,
                phase=phase,
                sender="系统"
            )
        elif event_type == "player_death":
            # 玩家死亡
            self.interface.add_message(
                MessageType.PUBLIC,
                description,
                phase=event.get("phase", "night"),
                sender="系统"
            )

    def add_private_memory(self, event: Dict, description: str):
        """添加私有记忆（只显示给人类玩家）"""
        self.memory.add_event(event, description)

        # 显示给人类玩家
        self.interface.add_message(
            MessageType.PRIVATE,
            description,
            phase=event.get("phase", "night"),
            sender="系统"
        )

    async def day_speech(self, game_state: Dict) -> str:
        """人类玩家白天发言"""
        # 更新界面显示
        self.interface.update_game_state({
            **game_state,
            "phase": "day_discussion",
            "is_alive": self.is_alive
        })

        # 提示发言
        speech = self.interface.prompt_speech(game_state)

        # 添加发言到自己的记忆
        self.add_public_memory(
            {
                "event_id": f"speech_{self.player_id}_{game_state['day']}",
                "event_type": "player_speech",
                "timestamp": "",  # 实际使用时应该添加时间戳
                "data": {
                    "player_id": self.player_id,
                    "content": speech,
                    "order": 0  # 顺序会在game_manager中设置
                }
            },
            f"我发言说：{speech}"
        )

        return speech

    async def vote(self, game_state: Dict) -> str:
        """人类玩家投票"""
        # 更新界面显示
        self.interface.update_game_state({
            **game_state,
            "phase": "voting",
            "is_alive": self.is_alive
        })

        # 提示投票
        target = self.interface.prompt_vote(game_state)

        # 验证投票目标
        alive_players = game_state.get('alive_players', [])
        if target not in alive_players or target == self.player_id:
            # 无效投票，默认投第一个不是自己的玩家
            others = [p for p in alive_players if p != self.player_id]
            target = others[0] if others else self.player_id

        return target

    async def night_action(self, game_state: Dict, all_roles: Dict = None) -> Optional[Dict]:
        """人类玩家夜晚行动"""
        if not self.is_alive:
            return None

        # 更新界面显示
        self.interface.update_game_state({
            **game_state,
            "phase": "night",
            "is_alive": self.is_alive
        })

        # 根据角色提示行动
        if self.role == "werewolf":
            # 添加队友信息
            game_state["teammates"] = getattr(self, 'teammates', [])
            return self.interface.prompt_night_action(game_state, "werewolf")

        elif self.role == "seer":
            # 添加已查验玩家信息
            game_state["checked_players"] = getattr(self, 'checked_players', [])
            # 添加 all_roles 信息用于确定查验结果
            game_state["all_roles"] = all_roles if all_roles else {}
            return self.interface.prompt_night_action(game_state, "seer")

        elif self.role == "witch":
            # 添加药水状态
            game_state["has_antidote"] = getattr(self, 'has_antidote', True)
            game_state["has_poison"] = getattr(self, 'has_poison', True)
            return self.interface.prompt_night_action(game_state, "witch")

        return None

    async def night_discussion(self, round_num: int, total_rounds: int,
                               teammates: List[str], previous_discussion: str,
                               game_state: Dict) -> Dict:
        """人类玩家狼人讨论"""
        # 更新界面显示
        self.interface.update_game_state({
            **game_state,
            "phase": "werewolf_night",
            "is_alive": self.is_alive,
            "teammates": teammates
        })

        # 显示之前的讨论
        if previous_discussion:
            self.interface.add_message(
                MessageType.PRIVATE,
                f"之前的讨论:\n{previous_discussion}",
                phase="werewolf_night"
            )

        # 提示讨论发言
        prompt = f"""
【狼人讨论】第 {game_state.get('day', 1)} 夜，第 {round_num}/{total_rounds} 轮
你的狼人队友: {teammates}
存活玩家: {game_state.get('alive_players', [])}

请发表你的观点：
"""

        speech = self.interface.prompt_action(prompt)

        return {
            "speech": speech,
            "suggested_target": None
        }

    async def night_vote(self, discussion_summary: str, game_state: Dict, all_roles: Dict) -> str:
        """人类玩家狼人投票"""
        # 显示讨论总结
        self.interface.add_message(
            MessageType.PRIVATE,
            f"讨论总结:\n{discussion_summary}",
            phase="werewolf_night"
        )

        # 过滤掉狼人队友
        non_werewolf_players = [p for p in game_state['alive_players']
                                if all_roles.get(p) != 'werewolf']

        prompt = f"""
【狼人投票】请选择今晚要击杀的目标：

可击杀目标：{non_werewolf_players}

请选择：
"""

        target = self.interface.prompt_action(prompt, non_werewolf_players)

        # 确保不刀队友
        if target and target in game_state['alive_players'] and all_roles.get(target) != 'werewolf':
            return target

        # 默认选择
        return non_werewolf_players[0] if non_werewolf_players else game_state['alive_players'][0]

    async def summarize_day(self, day: int, game_state: Dict[str, Any]) -> str:
        """人类玩家日终总结"""
        # 简单显示总结提示
        self.interface.add_message(
            MessageType.SYSTEM,
            f"第 {day} 天结束，即将进入第 {day + 1} 天",
            phase="daily_summary"
        )
        return f"第 {day} 天总结：游戏继续"

    async def call_llm(self, *args, **kwargs):
        """人类玩家不需要调用LLM"""
        return "{}"


# 修改 create_agent 函数
def create_agent(player_id: str, role: str, api_key: str, character: str = "character_01",
                 is_human: bool = False) -> WerewolfAgent:
    """创建 Agent 工厂函数"""
    if is_human:
        return HumanAgent(player_id, role, api_key, character)
    elif role == "werewolf":
        return WerewolfAgentRole(player_id, role, api_key, character)
    elif role == "seer":
        return SeerAgentRole(player_id, role, api_key, character)
    elif role == "witch":
        return WitchAgentRole(player_id, role, api_key, character)
    else:
        return WerewolfAgent(player_id, role, api_key, character)