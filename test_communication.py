"""通信客户端测试"""
import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from framework_and_agne_main_loop import CommunicationClient, AgentConfig, BaseWerewolfAgent


class MockAgent:
    """模拟Agent类用于测试"""

    def __init__(self):
        self.config = AgentConfig(
            agent_id="test_comm_agent",
            game_id="test_comm_game"
        )
        self.logger = Mock()
        self.logger.info = Mock()
        self.logger.error = Mock()
        self.logger.warning = Mock()
        self.logger.debug = Mock()
        self.game_state = {"day": 1}
        self.my_role = None
        self.processed_ids = set()


@pytest.fixture
def temp_game_dir():
    """临时游戏目录fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        game_dir = Path(tmpdir) / "game_data" / "game_test_comm_game"
        game_dir.mkdir(parents=True)

        # 创建必要的子目录
        (game_dir / "logs").mkdir()
        (game_dir / "config").mkdir()
        (game_dir / "private" / "roles").mkdir(parents=True)
        (game_dir / "agents" / "test_comm_agent").mkdir(parents=True)

        yield game_dir


@pytest.fixture
def comm_client(temp_game_dir):
    """通信客户端fixture"""
    agent = MockAgent()
    client = CommunicationClient(agent)
    client.game_dir = temp_game_dir  # 覆盖为测试目录

    # 更新相关路径
    client.game_events_log = temp_game_dir / "logs" / "game_events.log"
    client.public_speech_log = temp_game_dir / "logs" / "public_speech.log"
    client.game_state_log = temp_game_dir / "logs" / "game_state.log"
    client.action_file = temp_game_dir / "agent_actions.json"
    client.memory_file = temp_game_dir / "agents" / "test_comm_agent" / "memory.json"

    return client


@pytest.mark.asyncio
async def test_connect_disconnect(comm_client):
    """测试连接和断开连接"""
    result = await comm_client.connect()
    assert result is True
    assert comm_client.connected is True

    await comm_client.disconnect()
    assert comm_client.connected is False


@pytest.mark.asyncio
async def test_heartbeat(comm_client):
    """测试心跳检查"""
    await comm_client.connect()

    # 测试缺少文件的情况
    heartbeat = await comm_client.send_heartbeat()
    assert heartbeat["status"] == "partial"
    assert "missing_files" in heartbeat

    # 创建必要文件
    comm_client.game_events_log.touch()
    comm_client.game_state_log.touch()
    comm_client.action_file.touch()

    heartbeat = await comm_client.send_heartbeat()
    assert heartbeat["status"] == "healthy"


@pytest.mark.asyncio
async def test_event_polling(comm_client):
    """测试事件轮询"""
    await comm_client.connect()

    # 创建测试事件文件
    test_events = [
        {
            "event_id": "evt_001",
            "event_type": "player_speech",
            "timestamp": datetime.now().isoformat(),
            "player_id": "player1",
            "text": "测试发言1"
        },
        {
            "event_id": "evt_002",
            "event_type": "phase_change",
            "timestamp": datetime.now().isoformat(),
            "old_phase": "day",
            "new_phase": "night"
        }
    ]

    # 写入事件日志
    with open(comm_client.game_events_log, 'w') as f:
        for event in test_events:
            f.write(json.dumps(event) + '\n')

    # 轮询事件
    await comm_client.poll_events()

    # 检查事件队列
    event1 = comm_client.get_next_event()
    event2 = comm_client.get_next_event()
    event3 = comm_client.get_next_event()  # 应该没有更多事件

    assert event1 is not None
    assert event2 is not None
    assert event3 is None


@pytest.mark.asyncio
async def test_action_submission(comm_client):
    """测试行动提交"""
    await comm_client.connect()

    action_data = {
        "action": "submit_vote",
        "data": {"target_id": "player2"},
        "timestamp": datetime.now().isoformat()
    }

    result = await comm_client.submit_action(action_data)
    assert result["success"] is True
    assert "action_id" in result

    # 验证文件内容
    assert comm_client.action_file.exists()
    with open(comm_client.action_file, 'r') as f:
        saved_data = json.load(f)
        assert len(saved_data["actions"]) == 1
        assert saved_data["actions"][0]["agent_id"] == "test_comm_agent"


@pytest.mark.asyncio
async def test_query_functions(comm_client):
    """测试查询功能"""
    await comm_client.connect()

    # 创建配置文件
    config_data = {
        "game_name": "测试游戏",
        "player_count": 8,
        "roles": ["werewolf", "seer", "witch", "villager"]
    }
    config_file = comm_client.game_dir / "config" / "metadata.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    # 测试配置查询
    result = await comm_client.query("get_config", {})
    assert result["success"] is True
    assert result["data"]["game_metadata"]["game_name"] == "测试游戏"

    # 测试公共状态查询
    # 先创建状态文件
    state_data = {
        "phase": "DAY",
        "day_number": 1,
        "alive_players": ["player1", "player2"],
        "timestamp": datetime.now().isoformat()
    }
    with open(comm_client.game_state_log, 'w') as f:
        f.write(json.dumps(state_data) + '\n')

    result = await comm_client.query("query_public_state", {})
    assert result["success"] is True
    assert result["data"]["phase"] == "DAY"
    assert len(result["data"]["alive_players"]) == 2


@pytest.mark.asyncio
async def test_memory_persistence(comm_client):
    """测试记忆持久化"""
    await comm_client.connect()

    # 测试保存记忆
    memory_data = {
        "entries": [
            {
                "timestamp": "2024-01-01T00:00:00",
                "content": "测试记忆条目1",
                "type": "test",
                "importance": 0.8
            }
        ]
    }

    await comm_client.save_memory(memory_data)
    assert comm_client.memory_file.exists()

    # 测试加载记忆
    loaded = await comm_client.load_memory()
    assert len(loaded["entries"]) == 1
    assert loaded["entries"][0]["content"] == "测试记忆条目1"


def test_event_parsing(comm_client):
    """测试事件解析"""
    # 测试发言解析
    speech_data = {
        "player_id": "test_player",
        "player_name": "测试玩家",
        "text": "这是一条测试发言",
        "sentiment": 0.7,
        "keywords": ["测试", "发言"]
    }

    parsed = comm_client._parse_speech_event(speech_data)
    assert parsed["event_type"] == "player_speech"
    assert parsed["data"]["player_id"] == "test_player"
    assert parsed["data"]["content"] == "这是一条测试发言"

    # 测试投票解析
    vote_data = {
        "round_id": "vote_round_1",
        "day_number": 1,
        "candidates": ["p1", "p2"],
        "votes": {"p1": 3, "p2": 2},
        "result": "p1出局"
    }

    parsed = comm_client._parse_vote_event(vote_data)
    assert parsed["event_type"] == "vote_result"
    assert parsed["data"]["result"] == "p1出局"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])