# core_agent.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from config import AgentConfig, AgentState, Role, PlayerInfo, GamePhase
from memory import AgentMemory
from communication import CommunicationClient
from llm_client import LLMClient
from decision import DecisionEngine

class BaseWerewolfAgent(ABC):
    """Agent 基类 - 包含完整的状态管理和辅助方法"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = self._setup_logger()

        # 核心组件
        self.memory = AgentMemory(config)
        self.comm_client = CommunicationClient(self)
        self.llm_client = LLMClient(config.llm)
        self.lifecycle_manager = AgentLifecycleManager(self)
        self.decision_engine = DecisionEngine(self)

        # 状态
        self.state = AgentState.INITIALIZING
        self.game_state: Dict[str, Any] = {}
        self.known_players: Dict[str, PlayerInfo] = {}
        self.my_role: Optional[Role] = None
        self.my_id = config.agent_id
        self._running = False

        # 历史记录
        self.game_phase_history = []
        self.actions_taken = []
        self.speeches_made = []

    def _setup_logger(self):
        logger = logging.getLogger(f"Agent-{self.config.agent_id}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(self.config.log_level)
        return logger

    # --- 生命周期与主循环 ---
    async def start(self):
        await self.lifecycle_manager.transition_to(AgentState.INITIALIZING)
        await self.comm_client.connect()
        # 模拟认证和获取配置
        await self._fetch_initial_config()
        await self.lifecycle_manager.transition_to(AgentState.READY)
        self._running = True
        await self._main_loop()

    async def stop(self):
        self._running = False
        await self.comm_client.disconnect()
        await self.lifecycle_manager.transition_to(AgentState.STOPPED)

    async def _main_loop(self):
        while self._running:
            try:
                # 1. 轮询
                await self.comm_client.poll_events()

                # 2. 处理事件
                while True:
                    event = self.comm_client.get_next_event()
                    if not event: break
                    self._process_event(event)

                # 3. 阶段处理
                current_phase = self.game_state.get("phase")
                if current_phase:
                    await self._on_game_phase(current_phase)

                await asyncio.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                await asyncio.sleep(1)

    def _process_event(self, event):
        # 增强数据
        if "data" not in event: event["data"] = {}
        event["data"]["day"] = self.game_state.get("day", 1)

        # 存入记忆
        self.memory.add_event(event)

        # 分发处理
        etype = event.get("event_type")
        if etype == "phase_change":
            self.game_state["phase"] = event["data"]["new_phase"]
            self.game_state["day"] = event["data"]["day"]
            self.logger.info(f"阶段变更 -> {self.game_state['phase']}")

        elif etype == "player_speech":
            self.logger.info(f"收到发言: {event['data'].get('player_id')} - {event['data'].get('content')[:20]}...")

        elif etype == "player_death":
            pid = event["data"].get("player_id")
            if pid in self.known_players:
                self.known_players[pid].is_alive = False
            if pid == self.my_id:
                self.lifecycle_manager.transition_to(AgentState.DEAD)

    # --- 辅助方法 (恢复原有的信任计算逻辑) ---
    def _calculate_trust_score(self, player_id: str) -> float:
        if player_id not in self.known_players: return 0.5
        player_memories = self.memory.search_by_tag(f"player_{player_id}")
        if not player_memories: return 0.5

        trust = 0.5
        pos = sum(1 for m in player_memories if "诚实" in m.tags)
        neg = sum(1 for m in player_memories if "谎言" in m.tags or "可疑" in m.tags)

        if pos + neg > 0:
            trust += (pos - neg) * 0.1
        return max(0.0, min(1.0, trust))

    def get_memory_summary(self, limit=10):
        return self.memory.get_summary(limit)

    async def _fetch_initial_config(self):
        # 模拟从通信模块获取角色
        role_resp = await self.comm_client.query("query_role_info", {"info_type": "my_role"})
        if role_resp.get("success"):
            try:
                self.my_role = Role(role_resp["data"]["role"])
                self.logger.info(f"身份确认: {self.my_role}")
            except:
                self.my_role = Role.VILLAGER

        # 填充已知玩家
        state_resp = await self.comm_client.query("query_public_state")
        if state_resp.get("success"):
            for p in state_resp["data"].get("alive_players", []):
                self.known_players[p["id"]] = PlayerInfo(p["id"], p.get("name", p["id"]), False)

    # --- 抽象接口 ---
    @abstractmethod
    async def on_game_start(self):
        pass

    @abstractmethod
    async def _on_game_phase(self, phase):
        pass
    
class AgentLifecycleManager:

    def __init__(self, agent: 'BaseWerewolfAgent'):
        self.agent = agent
        self.state_history: List[Dict[str, Any]] = []
        self.transition_handlers: Dict[AgentState, Callable] = {}
        self._setup_transition_handlers()

    def _setup_transition_handlers(self):
        """设置状态转换处理器"""
        self.transition_handlers = {
            AgentState.INITIALIZING: self._on_initializing,
            AgentState.CONNECTING: self._on_connecting,
            AgentState.CONNECTED: self._on_connected,
            AgentState.AUTHENTICATED: self._on_authenticated,
            AgentState.READY: self._on_ready,
            AgentState.PLAYING: self._on_playing,
            AgentState.WAITING: self._on_waiting,
            AgentState.DEAD: self._on_dead,
            AgentState.ERROR: self._on_error,
            AgentState.DISCONNECTED: self._on_disconnected,
            AgentState.STOPPED: self._on_stopped,
        }

    async def transition_to(self, new_state: AgentState,
                            data: Dict[str, Any] = None,
                            reason: str = ""):
        """状态转换"""
        old_state = self.agent.state

        # 检查状态转换是否有效
        if not self._is_valid_transition(old_state, new_state):
            self.agent.logger.warning(
                f"Invalid state transition: {old_state} -> {new_state}"
            )
            return False

        # 记录状态转换
        transition_record = {
            "timestamp": datetime.now().isoformat(),
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "data": data or {}
        }
        self.state_history.append(transition_record)

        # 执行转换
        self.agent.state = new_state
        self.agent.logger.info(
            f"State transition: {old_state.value} -> {new_state.value} ({reason})"
        )

        # 调用状态处理器
        handler = self.transition_handlers.get(new_state)
        if handler:
            await handler(old_state, data or {})

        return True

    def _is_valid_transition(self, from_state: AgentState, to_state: AgentState) -> bool:
        """检查状态转换是否有效"""
        valid_transitions = {
            AgentState.INITIALIZING: [
                AgentState.CONNECTING, AgentState.ERROR, AgentState.STOPPED
            ],
            AgentState.CONNECTING: [
                AgentState.CONNECTED, AgentState.ERROR, AgentState.DISCONNECTED
            ],
            AgentState.CONNECTED: [
                AgentState.AUTHENTICATED, AgentState.DISCONNECTED, AgentState.ERROR
            ],
            AgentState.AUTHENTICATED: [
                AgentState.READY, AgentState.DISCONNECTED, AgentState.ERROR
            ],
            AgentState.READY: [
                AgentState.PLAYING, AgentState.WAITING, AgentState.DISCONNECTED
            ],
            AgentState.PLAYING: [
                AgentState.WAITING, AgentState.DEAD, AgentState.DISCONNECTED,
                AgentState.ERROR
            ],
            AgentState.WAITING: [
                AgentState.PLAYING, AgentState.DEAD, AgentState.DISCONNECTED
            ],
            AgentState.DEAD: [
                AgentState.DISCONNECTED, AgentState.STOPPED
            ],
            AgentState.DISCONNECTED: [
                AgentState.CONNECTING, AgentState.STOPPED
            ],
            AgentState.ERROR: [
                AgentState.CONNECTING, AgentState.STOPPED
            ],
            AgentState.STOPPED: []  # 终止状态
        }

        return to_state in valid_transitions.get(from_state, [])

    # 状态处理器方法
    async def _on_initializing(self, previous_state: AgentState, data: Dict):
        """初始化状态处理"""
        self.agent.logger.debug("Agent正在初始化...")

    async def _on_connecting(self, previous_state: AgentState, data: Dict):
        """连接中状态处理"""
        self.agent.logger.debug("正在连接到法官服务器...")

    async def _on_connected(self, previous_state: AgentState, data: Dict):
        """已连接状态处理"""
        self.agent.logger.info("已连接到法官服务器")

    async def _on_authenticated(self, previous_state: AgentState, data: Dict):
        """已认证状态处理"""
        self.agent.logger.info("身份认证成功")

    async def _on_ready(self, previous_state: AgentState, data: Dict):
        """准备就绪状态处理"""
        self.agent.logger.info("Agent准备就绪，等待游戏开始")
        await self.agent.on_game_start()

    async def _on_playing(self, previous_state: AgentState, data: Dict):
        """游戏中状态处理"""
        self.agent.logger.info("游戏进行中")

    async def _on_waiting(self, previous_state: AgentState, data: Dict):
        """等待状态处理"""
        self.agent.logger.debug("等待中...")

    async def _on_dead(self, previous_state: AgentState, data: Dict):
        """死亡状态处理"""
        self.agent.logger.info("玩家死亡，游戏结束")
        await self.agent._handle_player_death(data)

    async def _on_error(self, previous_state: AgentState, data: Dict):
        """错误状态处理"""
        error_msg = data.get("error", "Unknown error")
        self.agent.logger.error(f"Agent进入错误状态: {error_msg}")
        await self.agent._handle_loop_error(data)

    async def _on_disconnected(self, previous_state: AgentState, data: Dict):
            self.agent.logger.warning("Disconnected")

    async def _on_stopped(self, previous_state: AgentState, data: Dict):
            self.agent.logger.info("Stopped")

    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "current_state": self.agent.state.value,
            "history_length": len(self.state_history),
            "recent_transitions": self.state_history[-5:] if self.state_history else [],
            "uptime": self._calculate_uptime()
        }

    def _calculate_uptime(self) -> float:
        """计算运行时间"""
        if not self.state_history:
            return 0.0

        # 找到从INITIALIZING开始的时间
        for record in reversed(self.state_history):
            if record["from"] == AgentState.INITIALIZING.value:
                start_time = datetime.fromisoformat(record["timestamp"])
                return (datetime.now() - start_time).total_seconds()
        return 0.0