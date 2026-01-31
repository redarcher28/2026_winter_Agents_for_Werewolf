"""基础配置测试"""
import pytest
import tempfile
import os
from dataclasses import asdict
from framework_and_agne_main_loop import AgentConfig, LLMConfig, Role


def test_agent_config_creation():
    """测试AgentConfig创建"""
    config = AgentConfig(
        agent_id="test_agent_1",
        game_id="test_game_1",
        speech_style="moderate",
        risk_tolerance=0.7,
        max_memory_entries=50
    )

    assert config.agent_id == "test_agent_1"
    assert config.game_id == "test_game_1"
    assert config.speech_style == "moderate"
    assert 0 <= config.risk_tolerance <= 1
    assert config.max_memory_entries == 50


def test_llm_config_with_env():
    """测试LLMConfig环境变量"""
    os.environ["OPENAI_API_KEY"] = "test_key_123"

    config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        temperature=0.8
    )

    assert config.provider == "openai"
    assert config.model == "gpt-3.5-turbo"
    assert config.temperature == 0.8


def test_config_serialization():
    """测试配置序列化"""
    config = AgentConfig(
        agent_id="test_serialize",
        game_id="game_1"
    )

    config_dict = asdict(config)
    assert isinstance(config_dict, dict)
    assert config_dict["agent_id"] == "test_serialize"
    assert "game_id" in config_dict
    assert "speech_style" in config_dict


@pytest.mark.parametrize("role_str,expected_role", [
    ("werewolf", Role.WEREWOLF),
    ("seer", Role.SEER),
    ("witch", Role.WITCH),
    ("villager", Role.VILLAGER),
])
def test_role_enum(role_str, expected_role):
    """测试Role枚举"""
    role = Role(role_str)
    assert role == expected_role
    assert role.value == role_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])