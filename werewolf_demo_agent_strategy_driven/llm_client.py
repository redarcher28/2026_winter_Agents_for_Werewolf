# llm_client.py
import json
import logging
import re
import openai
from typing import Dict, List, Any, Optional
from config import LLMConfig, StrategyDecision


class LLMClient:
    """LLM客户端，按决策种类和行动类型调整调用端口"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"LLMClient-{self.config.provider}")

        if self.config.provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.config.api_key)

    # 狼人专用端口
    async def decide_wolf_vote(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """狼人投票决策"""
        if prompt is None:
            prompt = f"""
            你是狼人，需隐藏身份。上下文：{json.dumps(context)}
            决定投票目标，避免暴露自己。输出JSON：
            {{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.7}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["target_id", "reason", "confidence"])
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", "")}
        )

    async def decide_wolf_speech(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """狼人发言决策"""
        if prompt is None:
            prompt = f"""
            你是狼人，需伪装成好人。上下文：{json.dumps(context)}
            生成隐藏身份的发言。输出JSON：
            {{"speech": "发言内容", "confidence": 0.7}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["speech", "confidence"])
        return StrategyDecision(
            decision_type="speech",
            data={"content": raw.get("speech", ""), "speech_round": context.get("speech_round", 1),
                  "turn_order": context.get("turn_order", 0)},
            confidence=raw.get("confidence", 0.5),
            debug={}
        )

    async def decide_wolf_kill(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """狼人夜晚刀人决策"""
        if prompt is None:
            prompt = f"""
            你是狼人，选择今晚刀人目标。上下文：{json.dumps(context)}
            输出JSON：
            {{"target_id": "玩家ID", "reason": "刀人理由", "confidence": 0.7}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["target_id", "reason", "confidence"])
        return StrategyDecision(
            decision_type="night_action",
            data={"action_type": "kill", "target_id": raw.get("target_id")},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", "")}
        )

    # 神职专用端口
    async def decide_seer_vote(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """预言家投票决策"""
        if prompt is None:
            prompt = f"""
            你是预言家，基于查验结果推理。上下文：{json.dumps(context)}
            决定投票目标。输出JSON：
            {{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.8}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["target_id", "reason", "confidence"])
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", "")}
        )

    async def decide_seer_speech(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """预言家发言决策"""
        if prompt is None:
            prompt = f"""
            你是预言家，谨慎发言避免暴露。上下文：{json.dumps(context)}
            生成发言。输出JSON：
            {{"speech": "发言内容", "confidence": 0.8}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["speech", "confidence"])
        return StrategyDecision(
            decision_type="speech",
            data={"content": raw.get("speech", ""), "speech_round": context.get("speech_round", 1),
                  "turn_order": context.get("turn_order", 0)},
            confidence=raw.get("confidence", 0.5),
            debug={}
        )

    async def decide_seer_check(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """预言家夜晚查验决策"""
        if prompt is None:
            prompt = f"""
            你是预言家，选择查验目标。上下文：{json.dumps(context)}
            输出JSON：
            {{"target_id": "玩家ID", "reason": "查验理由", "confidence": 0.8}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["target_id", "reason", "confidence"])
        return StrategyDecision(
            decision_type="night_action",
            data={"action_type": "check", "target_id": raw.get("target_id")},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", "")}
        )

    async def decide_witch_vote(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """女巫投票决策"""
        if prompt is None:
            prompt = f"""
            你是女巫，基于药水使用推理。上下文：{json.dumps(context)}
            决定投票目标。输出JSON：
            {{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.6}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["target_id", "reason", "confidence"])
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", "")}
        )

    async def decide_witch_speech(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """女巫发言决策"""
        if prompt is None:
            prompt = f"""
            你是女巫，低调发言。上下文：{json.dumps(context)}
            生成发言。输出JSON：
            {{"speech": "发言内容", "confidence": 0.6}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["speech", "confidence"])
        return StrategyDecision(
            decision_type="speech",
            data={"content": raw.get("speech", ""), "speech_round": context.get("speech_round", 1),
                  "turn_order": context.get("turn_order", 0)},
            confidence=raw.get("confidence", 0.5),
            debug={}
        )

    async def decide_witch_action(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """女巫夜晚行动决策（救人或毒人）"""
        if prompt is None:
            prompt = f"""
            你是女巫，决定使用药水。上下文：{json.dumps(context)}
            输出JSON：
            {{"action_type": "save/poison/no_potion", "target_id": "玩家ID或null", "reason": "行动理由", "confidence": 0.6}}
            """
        raw = await self._call_llm_with_retry(prompt,
                                              required_fields=["action_type", "target_id", "reason", "confidence"])
        action_type = raw.get("action_type", "no_potion")
        return StrategyDecision(
            decision_type="night_action",
            data={"action_type": action_type, "target_id": raw.get("target_id")},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", "")}
        )

    # 村民专用端口
    async def decide_villager_vote(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """村民投票决策"""
        if prompt is None:
            prompt = f"""
            你是村民，推理生存。上下文：{json.dumps(context)}
            选择投票目标。输出JSON：
            {{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.5}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["target_id", "reason", "confidence"])
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", "")}
        )

    async def decide_villager_speech(self, context: Dict, prompt: str = None) -> StrategyDecision:
        """村民发言决策"""
        if prompt is None:
            prompt = f"""
            你是村民，表达怀疑。上下文：{json.dumps(context)}
            生成发言。输出JSON：
            {{"speech": "发言内容", "confidence": 0.5}}
            """
        raw = await self._call_llm_with_retry(prompt, required_fields=["speech", "confidence"])
        return StrategyDecision(
            decision_type="speech",
            data={"content": raw.get("speech", ""), "speech_round": context.get("speech_round", 1),
                  "turn_order": context.get("turn_order", 0)},
            confidence=raw.get("confidence", 0.5),
            debug={}
        )

    # 通用底层调用（私有方法）
    async def _call_llm_with_retry(self, prompt: str, required_fields: List[str] = None) -> Dict:
        """底层LLM调用，带重试和校验"""
        for attempt in range(2):  # 重试一次
            try:
                response_str = await self._call_llm(prompt)
                raw = json.loads(response_str)
                # 简单校验：确保有必要字段
                if not isinstance(raw, dict):
                    raise ValueError("Not a dict")
                if required_fields:
                    for field in required_fields:
                        if field not in raw or raw[field] is None:
                            raise ValueError(f"Missing required field: {field}")
                return raw
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.warning(f"LLM output invalid on attempt {attempt + 1}: {e}")
                if attempt == 1:  # 最后一次失败，抛出异常
                    raise ValueError("LLM output invalid after retries") from e

    async def _call_llm(self, prompt: str) -> str:
        """底层LLM调用，返回JSON字符串"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            # 返回默认JSON响应
            return '{"error": "LLM call failed", "speech": "我还没有想好", "confidence": 0.1}'