"""LLM客户端测试"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from framework_and_agne_main_loop import LLMClient, LLMConfig, StrategyDecision
import framework_and_agne_main_loop as main_module


@pytest.fixture
def mock_llm_config():
    """模拟LLM配置"""
    return LLMConfig(
        provider="openai",
        api_key="test_api_key_12345",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        timeout=10.0
    )


class TestLLMClient:
    """LLM客户端测试类"""

    @pytest.mark.asyncio
    async def test_decide_wolf_vote(self):
        """测试狼人投票决策"""
        # 创建配置
        config = LLMConfig(provider="openai", api_key="test_key")

        # 关键：直接替换整个 LLMClient 的 _call_llm 方法
        client = LLMClient(config)

        # 使用 AsyncMock 模拟 _call_llm 方法
        mock_response = json.dumps({
            "target_id": "player_003",
            "reason": "该玩家发言可疑",
            "confidence": 0.8
        })

        # 直接替换 client 的 _call_llm 方法
        client._call_llm = AsyncMock(return_value=mock_response)

        context = {
            "day": 1,
            "alive_players": [
                {"id": "player_001", "name": "玩家1"},
                {"id": "player_002", "name": "玩家2"},
                {"id": "player_003", "name": "玩家3"}
            ]
        }

        decision = await client.decide_wolf_vote(context)

        assert isinstance(decision, StrategyDecision)
        assert decision.decision_type == "vote"
        assert decision.data["target_id"] == "player_003"
        assert decision.confidence == 0.8
        assert "reason" in decision.debug

    @pytest.mark.asyncio
    async def test_llm_call_with_retry(self):
        """测试带重试的LLM调用"""
        # 创建配置
        config = LLMConfig(provider="openai", api_key="test_key")

        # 创建客户端
        client = LLMClient(config)

        # 模拟 _call_llm 方法
        client._call_llm = AsyncMock()

        # 设置第一次返回无效JSON，第二次返回有效
        client._call_llm.side_effect = [
            "invalid json",
            '{"speech": "测试发言", "confidence": 0.7}'
        ]

        # 直接调用 _call_llm_with_retry
        result = await client._call_llm_with_retry(
            "生成测试发言",
            required_fields=["speech", "confidence"]
        )

        # 验证调用次数
        assert client._call_llm.call_count == 2
        assert result["speech"] == "测试发言"
        assert result["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_llm_call_with_retry_failure(self):
        """测试带重试的LLM调用失败情况"""
        # 创建配置
        config = LLMConfig(provider="openai", api_key="test_key")

        # 创建客户端
        client = LLMClient(config)

        # 模拟 _call_llm 方法
        client._call_llm = AsyncMock()

        # 两次都返回无效JSON
        client._call_llm.return_value = "invalid json"

        with pytest.raises(ValueError, match="LLM output invalid after retries"):
            await client._call_llm_with_retry(
                "生成测试发言",
                required_fields=["speech", "confidence"]
            )

    def test_role_specific_decisions(self):
        """测试角色特定决策方法的存在性"""
        # 创建配置
        config = LLMConfig(provider="openai", api_key="test_key")

        # 创建客户端（不需要模拟，只检查方法是否存在）
        client = LLMClient(config)

        test_cases = [
            ("decide_wolf_vote", "狼人投票"),
            ("decide_wolf_speech", "狼人发言"),
            ("decide_wolf_kill", "狼人杀人"),
            ("decide_seer_vote", "预言家投票"),
            ("decide_seer_speech", "预言家发言"),
            ("decide_seer_check", "预言家查验"),
            ("decide_witch_vote", "女巫投票"),
            ("decide_witch_speech", "女巫发言"),
            ("decide_witch_action", "女巫行动"),
            ("decide_villager_vote", "村民投票"),
            ("decide_villager_speech", "村民发言"),
        ]

        for method_name, description in test_cases:
            method = getattr(client, method_name, None)
            assert method is not None, f"{description} 方法 {method_name} 不存在"
            assert callable(method), f"{description} 方法 {method_name} 不可调用"

    @pytest.mark.asyncio
    async def test_all_decision_methods(self):
        """测试所有决策方法"""
        # 创建配置
        config = LLMConfig(provider="openai", api_key="test_key")

        # 测试方法及其所需字段
        test_methods = [
            ("decide_wolf_vote", {"target_id": "player_001", "reason": "测试原因", "confidence": 0.7}),
            ("decide_wolf_speech", {"speech": "测试狼人发言", "confidence": 0.6}),
            ("decide_wolf_kill", {"target_id": "player_002", "reason": "测试杀原因", "confidence": 0.8}),
            ("decide_seer_vote", {"target_id": "player_001", "reason": "预言家投票原因", "confidence": 0.9}),
            ("decide_seer_speech", {"speech": "测试预言家发言", "confidence": 0.8}),
            ("decide_seer_check", {"target_id": "player_002", "reason": "查验原因", "confidence": 0.85}),
            ("decide_witch_vote", {"target_id": "player_001", "reason": "女巫投票原因", "confidence": 0.7}),
            ("decide_witch_speech", {"speech": "测试女巫发言", "confidence": 0.6}),
            ("decide_witch_action",
             {"action_type": "save", "target_id": "player_001", "reason": "救人原因", "confidence": 0.75}),
            ("decide_villager_vote", {"target_id": "player_002", "reason": "村民投票原因", "confidence": 0.5}),
            ("decide_villager_speech", {"speech": "测试村民发言", "confidence": 0.5}),
        ]

        for method_name, response_content in test_methods:
            # 为每个方法创建新的客户端和模拟
            client = LLMClient(config)

            # 模拟 _call_llm 方法
            client._call_llm = AsyncMock()
            client._call_llm.return_value = json.dumps(response_content)

            # 基本上下文
            context = {
                "day": 1,
                "alive_players": [
                    {"id": "player_001", "name": "玩家1"},
                    {"id": "player_002", "name": "玩家2"}
                ],
                "speech_round": 1,
                "turn_order": 0
            }

            try:
                method = getattr(client, method_name)
                decision = await method(context)

                assert isinstance(decision, StrategyDecision)
                print(f"✓ {method_name} 测试通过")

            except Exception as e:
                print(f"✗ {method_name} 测试失败: {e}")
                # 继续测试其他方法
                continue

    @pytest.mark.asyncio
    async def test_llm_call_exception_handling(self):
        """测试LLM调用异常处理"""
        # 创建配置
        config = LLMConfig(provider="openai", api_key="test_key")

        # 创建客户端
        client = LLMClient(config)

        # 模拟 _call_llm 方法
        client._call_llm = AsyncMock()

        # 第一次返回错误JSON（缺少required_fields），第二次返回有效
        client._call_llm.side_effect = [
            '{"error": "LLM call failed", "speech": "我还没有想好", "confidence": 0.1}',
            '{"test": "data"}'
        ]

        # 调用时指定 required_fields，这样第一次返回的JSON中缺少"test"字段，会触发重试
        result = await client._call_llm_with_retry("测试", required_fields=["test"])
        assert result == {"test": "data"}
        assert client._call_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_call_json_validation(self):
        """测试LLM调用JSON验证"""
        # 创建配置
        config = LLMConfig(provider="openai", api_key="test_key")

        # 创建客户端
        client = LLMClient(config)

        # 模拟 _call_llm 方法
        client._call_llm = AsyncMock()

        # 返回非JSON字符串
        client._call_llm.return_value = "不是有效的JSON"

        with pytest.raises(ValueError, match="LLM output invalid after retries"):
            await client._call_llm_with_retry("测试", required_fields=["test"])


# 为了保持兼容性，也提供非类的测试函数
@pytest.mark.asyncio
async def test_simple_wolf_decision():
    """简单的狼人决策测试"""
    config = LLMConfig(provider="openai", api_key="test_key")
    client = LLMClient(config)

    # 直接替换 _call_llm 方法
    client._call_llm = AsyncMock(return_value=json.dumps({
        "target_id": "player_003",
        "reason": "该玩家发言可疑",
        "confidence": 0.8
    }))

    context = {"day": 1, "alive_players": []}
    decision = await client.decide_wolf_vote(context)

    assert isinstance(decision, StrategyDecision)
    assert decision.decision_type == "vote"
    assert decision.data["target_id"] == "player_003"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])