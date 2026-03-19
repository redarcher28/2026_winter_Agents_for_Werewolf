# anthropic_llm_client.py
"""
Anthropic Claude LLM 客户端

实现与 LLMClient (OpenAI) 相同的接口，使用 Anthropic Claude API。
支持 Claude 的 extended thinking（扩展思考）模式，即「算法艺术」(Algorithm Art)技术。

什么是「算法艺术」(Algorithm Art)?
    Anthropic 将 Claude 进行复杂推理时展示的结构化思考过程称为"算法艺术"——
    模型不仅给出结论，还展示推理链（Chain-of-Thought），就像一位艺术家展示
    创作过程。在狼人杀场景中，这意味着 Claude 会逐步分析：
      1. 当前已知信息（法官广播 + 私有信息）
      2. 每位玩家的行为模式
      3. 概率推断（贝叶斯更新）
      4. 最优行动（博弈论最优解）
    最终输出结构化 JSON 决策，同时在 debug 字段中保留推理过程。

使用方式：
    config = LLMConfig(
        provider="anthropic",
        api_key="sk-ant-...",
        model="claude-opus-4-5",
        temperature=1.0,        # extended thinking 要求 temperature=1
        max_tokens=16000,       # extended thinking 需要较大 token budget
    )
    client = AnthropicLLMClient(config)
    decision = await client.decide_wolf_kill(context)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import anthropic

from config import LLMConfig, StrategyDecision

# Claude 模型默认值
_DEFAULT_MODEL = "claude-opus-4-5"
# extended thinking 的最小 budget（tokens）
_THINKING_BUDGET_TOKENS = 8000


class AnthropicLLMClient:
    """
    Anthropic Claude 决策客户端。

    接口与 LLMClient (OpenAI 版) 完全一致，可直接替换使用。
    当 LLMConfig.model 为 claude-3-5-sonnet / claude-opus-4-5 等支持
    extended thinking 的模型时，自动启用思考模式（Algorithm Art 模式）。
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"AnthropicLLMClient-{config.model}")
        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self._use_thinking = _supports_thinking(config.model)
        if self._use_thinking:
            self.logger.info(
                "Extended thinking (Algorithm Art) enabled for model %s", config.model
            )

    # ------------------------------------------------------------------
    # 狼人专用端口
    # ------------------------------------------------------------------

    async def decide_wolf_vote(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """狼人投票决策"""
        if prompt is None:
            prompt = (
                f"你是狼人，需隐藏身份。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "决定投票目标，避免暴露自己。输出JSON：\n"
                '{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.7}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["target_id", "reason", "confidence"]
        )
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", ""), "thinking": raw.get("_thinking", "")},
        )

    async def decide_wolf_speech(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """狼人发言决策"""
        if prompt is None:
            prompt = (
                f"你是狼人，需伪装成好人。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "生成隐藏身份的发言。输出JSON：\n"
                '{"speech": "发言内容", "confidence": 0.7}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["speech", "confidence"]
        )
        return StrategyDecision(
            decision_type="speech",
            data={
                "content": raw.get("speech", ""),
                "speech_round": context.get("speech_round", 1),
                "turn_order": context.get("turn_order", 0),
            },
            confidence=raw.get("confidence", 0.5),
            debug={"thinking": raw.get("_thinking", "")},
        )

    async def decide_wolf_kill(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """狼人夜晚刀人决策"""
        if prompt is None:
            prompt = (
                f"你是狼人，选择今晚刀人目标。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "输出JSON：\n"
                '{"target_id": "玩家ID", "reason": "刀人理由", "confidence": 0.7}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["target_id", "reason", "confidence"]
        )
        return StrategyDecision(
            decision_type="night_action",
            data={"action_type": "kill", "target_id": raw.get("target_id")},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", ""), "thinking": raw.get("_thinking", "")},
        )

    # ------------------------------------------------------------------
    # 神职专用端口
    # ------------------------------------------------------------------

    async def decide_seer_vote(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """预言家投票决策"""
        if prompt is None:
            prompt = (
                f"你是预言家，基于查验结果推理。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "决定投票目标。输出JSON：\n"
                '{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.8}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["target_id", "reason", "confidence"]
        )
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", ""), "thinking": raw.get("_thinking", "")},
        )

    async def decide_seer_speech(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """预言家发言决策"""
        if prompt is None:
            prompt = (
                f"你是预言家，谨慎发言避免暴露。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "生成发言。输出JSON：\n"
                '{"speech": "发言内容", "confidence": 0.8}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["speech", "confidence"]
        )
        return StrategyDecision(
            decision_type="speech",
            data={
                "content": raw.get("speech", ""),
                "speech_round": context.get("speech_round", 1),
                "turn_order": context.get("turn_order", 0),
            },
            confidence=raw.get("confidence", 0.5),
            debug={"thinking": raw.get("_thinking", "")},
        )

    async def decide_seer_check(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """预言家夜晚查验决策"""
        if prompt is None:
            prompt = (
                f"你是预言家，选择查验目标。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "输出JSON：\n"
                '{"target_id": "玩家ID", "reason": "查验理由", "confidence": 0.8}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["target_id", "reason", "confidence"]
        )
        return StrategyDecision(
            decision_type="night_action",
            data={"action_type": "check", "target_id": raw.get("target_id")},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", ""), "thinking": raw.get("_thinking", "")},
        )

    async def decide_witch_vote(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """女巫投票决策"""
        if prompt is None:
            prompt = (
                f"你是女巫，基于药水使用推理。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "决定投票目标。输出JSON：\n"
                '{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.6}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["target_id", "reason", "confidence"]
        )
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", ""), "thinking": raw.get("_thinking", "")},
        )

    async def decide_witch_speech(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """女巫发言决策"""
        if prompt is None:
            prompt = (
                f"你是女巫，低调发言。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "生成发言。输出JSON：\n"
                '{"speech": "发言内容", "confidence": 0.6}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["speech", "confidence"]
        )
        return StrategyDecision(
            decision_type="speech",
            data={
                "content": raw.get("speech", ""),
                "speech_round": context.get("speech_round", 1),
                "turn_order": context.get("turn_order", 0),
            },
            confidence=raw.get("confidence", 0.5),
            debug={"thinking": raw.get("_thinking", "")},
        )

    async def decide_witch_action(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """女巫夜晚行动决策（救人或毒人）"""
        if prompt is None:
            prompt = (
                f"你是女巫，决定使用药水。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "输出JSON：\n"
                '{"action_type": "save/poison/no_potion", "target_id": "玩家ID或null", '
                '"reason": "行动理由", "confidence": 0.6}'
            )
        raw = await self._call_with_retry(
            prompt,
            required_fields=["action_type", "target_id", "reason", "confidence"],
        )
        return StrategyDecision(
            decision_type="night_action",
            data={"action_type": raw.get("action_type", "no_potion"), "target_id": raw.get("target_id")},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", ""), "thinking": raw.get("_thinking", "")},
        )

    # ------------------------------------------------------------------
    # 村民专用端口
    # ------------------------------------------------------------------

    async def decide_villager_vote(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """村民投票决策"""
        if prompt is None:
            prompt = (
                f"你是村民，推理生存。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "选择投票目标。输出JSON：\n"
                '{"target_id": "玩家ID", "reason": "投票理由", "confidence": 0.5}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["target_id", "reason", "confidence"]
        )
        return StrategyDecision(
            decision_type="vote",
            data={"target_id": raw.get("target_id"), "round": context.get("day", 1)},
            confidence=raw.get("confidence", 0.5),
            debug={"reason": raw.get("reason", ""), "thinking": raw.get("_thinking", "")},
        )

    async def decide_villager_speech(
        self, context: Dict, prompt: Optional[str] = None
    ) -> StrategyDecision:
        """村民发言决策"""
        if prompt is None:
            prompt = (
                f"你是村民，表达怀疑。上下文：{json.dumps(context, ensure_ascii=False)}\n"
                "生成发言。输出JSON：\n"
                '{"speech": "发言内容", "confidence": 0.5}'
            )
        raw = await self._call_with_retry(
            prompt, required_fields=["speech", "confidence"]
        )
        return StrategyDecision(
            decision_type="speech",
            data={
                "content": raw.get("speech", ""),
                "speech_round": context.get("speech_round", 1),
                "turn_order": context.get("turn_order", 0),
            },
            confidence=raw.get("confidence", 0.5),
            debug={"thinking": raw.get("_thinking", "")},
        )

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self, prompt: str, required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """带重试的 LLM 调用，校验必要字段"""
        for attempt in range(2):
            try:
                result = await self._call_claude(prompt)
                if not isinstance(result, dict):
                    raise ValueError("response is not a dict")
                if required_fields:
                    for field in required_fields:
                        if field not in result or result[field] is None:
                            raise ValueError(f"missing required field: {field}")
                return result
            except (json.JSONDecodeError, ValueError) as exc:
                self.logger.warning(
                    "Claude output invalid (attempt %d/2): %s", attempt + 1, exc
                )
                if attempt == 1:
                    raise ValueError("Claude output invalid after retries") from exc
        return {}  # unreachable, satisfies type checker

    async def _call_claude(self, prompt: str) -> Dict[str, Any]:
        """
        调用 Claude API，返回解析后的 dict。

        当模型支持 extended thinking（Algorithm Art 模式）时，
        额外传入 thinking 参数并把思考过程存入 _thinking 键。
        """
        try:
            kwargs: Dict[str, Any] = {
                "model": self.config.model or _DEFAULT_MODEL,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            if self._use_thinking:
                # extended thinking 要求 temperature=1 且需要 thinking budget
                kwargs["temperature"] = 1.0
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": min(
                        _THINKING_BUDGET_TOKENS, self.config.max_tokens // 2
                    ),
                }
            else:
                kwargs["temperature"] = self.config.temperature

            response = await self._client.messages.create(**kwargs)

            thinking_text = ""
            answer_text = ""

            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    answer_text += block.text

            raw = _parse_json(answer_text)
            if thinking_text:
                raw["_thinking"] = thinking_text
            return raw

        except anthropic.APIError as exc:
            self.logger.error("Anthropic API error: %s", exc)
            return {"error": str(exc), "speech": "我还没有想好", "confidence": 0.1}
        except Exception as exc:
            self.logger.error("Unexpected error calling Claude: %s", exc)
            return {"error": str(exc), "speech": "我还没有想好", "confidence": 0.1}


# ------------------------------------------------------------------
# 模块级辅助函数
# ------------------------------------------------------------------

def _supports_thinking(model: Optional[str]) -> bool:
    """判断模型是否支持 extended thinking（Algorithm Art 模式）"""
    if not model:
        return False
    thinking_models = (
        "claude-3-7-sonnet",
        "claude-opus-4",
        "claude-sonnet-4",
    )
    return any(m in model for m in thinking_models)


def _parse_json(text: str) -> Dict[str, Any]:
    """从文本中提取 JSON 对象（容忍 markdown 代码块）"""
    text = text.strip()
    # 去除 markdown 代码块标记
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取第一个 {...} 块
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])
    raise json.JSONDecodeError("No JSON object found", text, 0)
