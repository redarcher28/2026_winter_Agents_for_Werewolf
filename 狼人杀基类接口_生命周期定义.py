"""
狼人杀 Agent 通用基类框架
设计原则：模块化、可扩展、角色无关
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime


# ==================== 数据结构定义 ====================

class GamePhase(Enum):
    """游戏阶段枚举"""
    WEREWOLF_NIGHT = "werewolf_night"  # 狼人夜晚
    SEER_NIGHT = "seer_night"  # 预言家夜晚
    WITCH_NIGHT = "witch_night"  # 女巫夜晚
    DAYTIME_DISCUSSION = "daytime_discussion"  # 白天讨论
    DAYTIME_VOTING = "daytime_voting"  # 白天投票
    GAME_END = "game_end"  # 游戏结束


class Role(Enum):
    """角色枚举"""
    WEREWOLF = "werewolf" # 狼人
    SEER = "seer" # 预言家
    WITCH = "witch" # 女巫
    VILLAGER = "villager" # 村民

class AgentState(Enum):
    """Agent状态枚举"""
    INITIALIZING = "initializing"     # 初始化
    CONNECTING = "connecting"         # 连接中
    CONNECTED = "connected"          # 已连接
    AUTHENTICATED = "authenticated"  # 已认证
    READY = "ready"                  # 准备就绪
    PLAYING = "playing"              # 游戏中
    WAITING = "waiting"              # 等待中（如等待回合）
    DEAD = "dead"                    # 死亡
    DISCONNECTED = "disconnected"    # 断开连接
    ERROR = "error"                  # 错误状态
    STOPPED = "stopped"              # 已停止

@dataclass
class PlayerInfo:
    """玩家信息"""
    id: str
    name: str
    is_ai: bool
    is_alive: bool = True
    role: Optional[Role] = None  # 仅自己知道自己的角色


@dataclass
class GameEvent:
    """游戏事件"""
    event_id: str
    event_type: str
    timestamp: str
    data: Dict[str, Any]


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    timestamp: str
    event_type: str
    content: Dict[str, Any]
    importance: float  # 0.0-1.0，重要性评分
    tags: List[str]  # 标签，如["谎言", "投票模式", "可疑行为"]


@dataclass
class AgentConfig:
    """Agent配置"""
    agent_id: str # 用户及Agent身份标识
    game_id: str # 房间标识（用以避免多批次同时开始的混乱
    server_url: str = "ws://judge-server" #法官服务器的 WebSocket 地址
    llm_config: Dict[str, Any] = None #大语言模型的配置参数
    speech_style: str = "moderate"  # 发言风格：aggressive/moderate/conservative
    risk_tolerance: float = 0.5  # 决策参数：0.0-1.0，其中0.0为极度保守，表现为人云亦云隐藏身份，而1.0则为高度激进，狼人表现为直接悍跳带节奏，预言家发金水，村民大胆推理
    trust_threshold: float = 0.6  # 信任阈值：0.0-1.0，其中0.0为曹贼一般用人必疑，1.0为轻易信任，当信任参数超过预设的信任阈值时决定采信
    decision_delay: float = 2.0  # 模拟思考时间（秒）
    max_memory_entries: int = 100  # 最大记忆条目数（定期清理旧记忆）
    log_level: str = "INFO" # 控制日志输出详细程度，"DEBUG": 最详细，用于开发和调试；"INFO": 一般信息，适合正常游戏；"WARNING": 警告信息；"ERROR": 错误信息；"CRITICAL": 严重错误

# ==================== 生命周期管理器 ====================
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

    def _is_valid_transition(self, from_state: AgentState,
                             to_state: AgentState) -> bool:
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
            if record["from"] == "initializing":
                start_time = datetime.fromisoformat(record["timestamp"])
                return (datetime.now() - start_time).total_seconds()
        return 0.0
# ==================== 基类定义 ====================

class BaseWerewolfAgent(ABC):
    """
    Agent通用基类
    职责：通信管理、记忆管理、事件处理、决策循环
    """

    def __init__(self, config: AgentConfig):
        """
        初始化Agent

        Args:
            config: Agent配置
        """
        self.config = config

        # 核心组件
        self.memory = AgentMemory(config.max_memory_entries)
        self.comm_client = CommunicationClient(self)
        self.llm_client = LLMClient(config.llm_config)
        self.decision_engine = DecisionEngine(self)

        # 生命周期管理器
        self.lifecycle_manager = AgentLifecycleManager(self)
        self.state = AgentState.INITIALIZING

        # 游戏状态
        self.game_state: Dict[str, Any] = {}
        self.known_players: Dict[str, PlayerInfo] = {}
        self.my_role: Optional[Role] = None
        self.my_id = config.agent_id

        # 统计信息
        self.actions_taken: List[Dict] = []
        self.speeches_made: List[Dict] = []
        self.performance_metrics = {
            "cycles_completed": 0,
            "events_processed": 0,
            "avg_cycle_time": 0.0,
            "decision_times": [],
            "errors_encountered": 0
        }

        # 异步任务
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # 新增：游戏特定状态
        self.game_phase_history: List[Dict] = []
        self.current_turn: Optional[str] = None
        self.last_action_time: Optional[datetime] = None
        self.consecutive_inactivity = 0

        # 日志
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """配置日志器"""
        logger = logging.getLogger(f"Agent-{self.config.agent_id}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.config.log_level))
        return logger


    # ============ 生命周期管理 ============

    async def start(self):
        """启动Agent"""
        try:
            self.logger.info(f"Starting agent {self.config.agent_id}")

            # 使用生命周期管理器进行状态转换
            if self.state != AgentState.INITIALIZING:
                await self.lifecycle_manager.transition_to(
                    AgentState.INITIALIZING,
                    {"config": self.config.__dict__}
                )

            # 1. 连接服务器
            await self._establish_connection()

            # 2. 认证
            await self._authenticate()

            # 3. 获取游戏配置
            await self._fetch_initial_config()

            # 4. 进入准备状态
            await self.lifecycle_manager.transition_to(
                AgentState.READY,
                {"message": "Agent ready for game"}
            )

            # 5. 启动主循环
            await self._start_main_loop()

            self.logger.info("Agent started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start agent: {e}")
            await self.lifecycle_manager.transition_to(
                AgentState.ERROR,
                {"error": str(e), "stage": "startup"}
            )
            await self.stop()
            raise

    async def _establish_connection(self):
        """建立连接"""
        await self.lifecycle_manager.transition_to(
            AgentState.CONNECTING,
            {"server_url": self.config.server_url}
        )

        await self.comm_client.connect()

        # 验证连接
        if not self.comm_client.connected:
            raise ConnectionError("Failed to establish connection")

    async def _authenticate(self):
        """身份认证"""
        # 这里可以添加额外的认证逻辑
        await self.lifecycle_manager.transition_to(
            AgentState.AUTHENTICATED,
            {"method": "token"}
        )

    async def _start_main_loop(self):
        """启动主循环"""
        self._running = True
        await self.lifecycle_manager.transition_to(
            AgentState.PLAYING,
            {"action": "start_main_loop"}
        )

        # 创建主循环任务
        self._tasks.append(asyncio.create_task(self._main_loop()))

    async def stop(self):
        """停止Agent"""
        self.logger.info("Stopping agent...")

        # 1. 停止主循环
        self._running = False

        # 2. 取消所有任务
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # 3. 等待任务完成
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # 4. 断开连接
        await self.comm_client.disconnect()

        # 5. 进入停止状态
        await self.lifecycle_manager.transition_to(
            AgentState.STOPPED,
            {"reason": "normal_shutdown"}
        )

        self.logger.info("Agent stopped")

    # 新增：暂停和恢复功能
    async def pause(self):
        """暂停Agent"""
        if self.state == AgentState.PLAYING:
            await self.lifecycle_manager.transition_to(
                AgentState.WAITING,
                {"action": "pause"}
            )
            self._running = False
            self.logger.info("Agent paused")

    async def resume(self):
        """恢复Agent"""
        if self.state == AgentState.WAITING:
            await self.lifecycle_manager.transition_to(
                AgentState.PLAYING,
                {"action": "resume"}
            )
            self._running = True
            # 重新启动主循环
            self._tasks.append(asyncio.create_task(self._main_loop()))
            self.logger.info("Agent resumed")

    async def _main_loop(self):
        """Agent主循环"""
        self.logger.debug("Main loop started")

        while self._running:
            cycle_start = datetime.now()

            try:
                # 1. 处理待处理事件
                await self._process_pending_events()

                # 2. 检查游戏阶段
                current_phase = self.game_state.get("phase")
                if current_phase:
                    await self._handle_current_phase(current_phase)

                # 3. 更新策略
                await self._update_strategy()

                # 4. 发送心跳
                await self.comm_client.send_heartbeat()

                # 5. 记录性能指标
                self.performance_metrics["cycles_completed"] += 1

                # 6. 基础等待，防止CPU占用过高
                # 计算循环时间，动态调整等待
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                wait_time = max(0.01, 0.1 - cycle_time)  # 目标100ms循环
                await asyncio.sleep(wait_time)

            except asyncio.CancelledError:
                self.logger.debug("Main loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.performance_metrics["errors_encountered"] += 1
                # 使用新定义的错误处理方法
                await self._handle_loop_error(e)
                await asyncio.sleep(1)  # 错误后等待

    # ============ 抽象方法（子类必须实现） ============

    @abstractmethod
    async def on_game_start(self):
        """游戏开始时调用"""
        pass

    @abstractmethod
    async def on_night_action(self, phase: GamePhase):
        """夜晚行动阶段"""
        pass

    @abstractmethod
    async def on_daytime_discussion(self):
        """白天讨论阶段"""
        pass

    @abstractmethod
    async def on_voting_phase(self):
        """投票阶段"""
        pass

    @abstractmethod
    async def analyze_speech(self, player_id: str, content: str):
        """分析玩家发言"""
        pass

    @abstractmethod
    async def formulate_strategy(self) -> Dict[str, Any]:
        """制定策略"""
        pass

    # ============ 具体方法 ============

    async def _fetch_initial_config(self):
        """获取初始配置"""
        try:
            # 获取Agent配置
            config = await self.comm_client.query("get_config", {
                "config_type": "agent_settings"
            })
            self.config.speech_style = config.get("speech_style", self.config.speech_style)
            self.config.risk_tolerance = config.get("risk_tolerance", self.config.risk_tolerance)

            # 获取角色信息
            role_info = await self.comm_client.query("query_role_info", {
                "info_type": "my_role"
            })
            self.my_role = Role(role_info.get("role"))

            self.logger.info(f"Agent role: {self.my_role}")

        except Exception as e:
            self.logger.error(f"Failed to fetch initial config: {e}")

    async def _process_pending_events(self):
        """处理待处理事件"""
        while True:
            event = self.comm_client.get_next_event()
            if not event:
                break

            # 存储到记忆
            self.memory.add_event(event)

            # 根据事件类型处理
            handler = self._get_event_handler(event["event_type"])
            if handler:
                await handler(event)

    def _get_event_handler(self, event_type: str) -> Optional[Callable]:
        """获取事件处理器"""
        handlers = {
            "phase_change": self._handle_phase_change,
            "player_speech": self._handle_player_speech,
            "vote_result": self._handle_vote_result,
            "night_reveal": self._handle_night_reveal,
            "player_death": self._handle_player_death,
        }
        return handlers.get(event_type)

    async def _handle_phase_change(self, event: Dict):
        """处理阶段变更事件"""
        data = event["data"]
        old_phase = data["old_phase"]
        new_phase = data["new_phase"]

        self.game_state["phase"] = new_phase
        self.logger.info(f"Phase changed: {old_phase} -> {new_phase}")

        # 更新记忆
        self.memory.add_phase_change(old_phase, new_phase)

    async def _handle_player_speech(self, event: Dict):
        """处理玩家发言事件"""
        data = event["data"]
        player_id = data["player_id"]
        content = data["content"]

        self.logger.info(f"Player {player_id} said: {content[:50]}...")

        # 分析发言
        await self.analyze_speech(player_id, content)

        # 更新玩家模型
        self._update_player_model(player_id, {
            "last_speech": content,
            "speech_style": self._analyze_speech_style(content),
            "speech_timestamp": event["timestamp"]
        })

    async def _handle_vote_result(self, event: Dict):
        """处理投票结果事件"""
        data = event["data"]
        vote_result = data["result"]

        self.logger.info(f"Vote result: {vote_result}")

        # 更新游戏状态
        self.game_state["last_vote"] = vote_result

        # 分析投票模式
        await self._analyze_voting_patterns(data["votes"])

    async def _handle_night_reveal(self, event: Dict):
        """处理夜晚行动揭示事件"""
        data = event["data"]

        self.logger.info(f"Night reveal: {data.get('announcement', 'No announcement')}")

        # 更新游戏状态
        self.game_state["last_night"] = data

        # 分析死亡/拯救模式
        await self._analyze_night_actions(data)

    async def _handle_player_death(self, event: Dict):
        """处理玩家死亡事件"""
        data = event["data"]
        player_id = data["player_id"]

        self.logger.info(f"Player {player_id} died")

        # 更新玩家状态
        if player_id in self.known_players:
            self.known_players[player_id].is_alive = False

        # 如果是自己死亡，通过生命周期管理器更新状态
        if player_id == self.my_id:
            await self.lifecycle_manager.transition_to(
                AgentState.DEAD,
                {"player_id": player_id, "reason": "killed"}
            )

    # ============ 游戏阶段处理 ============
    async def _handle_current_phase(self, phase: str):
        """处理当前游戏阶段"""
        # 记录阶段历史
        phase_record = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "day": self.game_state.get("day", 0)
        }
        self.game_phase_history.append(phase_record)

        # 保持历史记录大小
        if len(self.game_phase_history) > 20:
            self.game_phase_history = self.game_phase_history[-20:]

        # 调用相应的处理器
        await self._on_game_phase(phase)

    async def _on_game_phase(self, phase: str):
        """处理游戏阶段"""
        if not self._can_act():
            return

        phase_handlers = {
            "werewolf_night": self._on_werewolf_night,
            "seer_night": self._on_seer_night,
            "witch_night": self._on_witch_night,
            "daytime_discussion": self._on_daytime_discussion,
            "daytime_voting": self._on_daytime_voting,
        }

        handler = phase_handlers.get(phase)
        if handler:
            await handler()

    async def _handle_loop_error(self, error: Exception):
        """处理循环错误 """
        error_type = type(error).__name__

        if isinstance(error, ConnectionError):
            self.logger.error(f"Connection error in main loop: {error}")
            # 简单的重连尝试
            try:
                await self.comm_client.disconnect()
                await asyncio.sleep(1)
                await self.comm_client.connect()
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")

        elif isinstance(error, TimeoutError):
            self.logger.warning(f"Timeout in main loop: {error}")

        else:
            self.logger.error(f"Unexpected error in main loop: {error}")

    async def _check_agent_health(self) -> bool:
        """检查Agent健康状态"""
        # 基础健康检查
        if not self.comm_client.connected:
            self.logger.warning("Agent not connected")
            return False

        if self.state in [AgentState.ERROR, AgentState.DEAD, AgentState.STOPPED]:
            return False

        return True

    async def _handle_health_issue(self):
        """处理健康问题"""
        self.logger.warning("Health issue detected, attempting basic recovery...")

        if not self.comm_client.connected:
            await self._reconnect()

    async def _reconnect(self):
        """重新连接"""
        self.logger.info("Attempting to reconnect...")
        try:
            if hasattr(self.comm_client, 'disconnect'):
                await self.comm_client.disconnect()
            await asyncio.sleep(1)
            await self.comm_client.connect()
            self.logger.info("Reconnection successful")
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")


    async def _on_werewolf_night(self):
        """狼人夜晚"""
        if self.my_role == Role.WEREWOLF and self._is_my_turn():
            await self.on_night_action(GamePhase.WEREWOLF_NIGHT)

    async def _on_seer_night(self):
        """预言家夜晚"""
        if self.my_role == Role.SEER and self._is_my_turn():
            await self.on_night_action(GamePhase.SEER_NIGHT)

    async def _on_witch_night(self):
        """女巫夜晚"""
        if self.my_role == Role.WITCH and self._is_my_turn():
            await self.on_night_action(GamePhase.WITCH_NIGHT)

    async def _on_daytime_discussion(self):
        """白天讨论"""
        if self._can_speak():
            await self.on_daytime_discussion()

    async def _on_daytime_voting(self):
        """白天投票"""
        if self._can_vote():
            await self.on_voting_phase()

    # ============ 工具方法 ============

    def _can_act(self) -> bool:
        """检查是否可以行动"""
        return (self.state == AgentState.PLAYING and
                self._is_alive())

    def _can_speak(self) -> bool:
        """检查是否可以发言"""
        return self._can_act() and self.game_state.get("can_speak", False)

    def _can_vote(self) -> bool:
        """检查是否可以投票"""
        return self._can_act() and self.game_state.get("can_vote", False)

    def _is_my_turn(self) -> bool:
        """检查是否是我的回合"""
        # TODO: 实现回合检查逻辑
        return True

    def _is_alive(self) -> bool:
        """检查是否存活"""
        return self.state != AgentState.DEAD

    def _update_player_model(self, player_id: str, data: Dict):
        """更新玩家模型"""
        # TODO: 实现玩家行为模型更新
        pass

    def _analyze_speech_style(self, content: str) -> str:
        """分析发言风格"""
        # TODO: 实现发言风格分析
        return "neutral"

    async def _analyze_voting_patterns(self, votes: Dict):
        """分析投票模式"""
        # TODO: 实现投票模式分析
        pass

    async def _analyze_night_actions(self, night_data: Dict):
        """分析夜晚行动"""
        # TODO: 实现夜晚行动分析
        pass

    async def _update_strategy(self):
        """更新策略"""
        if not self._can_act():
            return

        # 获取最新策略
        strategy = await self.formulate_strategy()

        # 更新决策引擎
        self.decision_engine.update_strategy(strategy)

        # 如果需要，调整配置
        if "risk_tolerance" in strategy:
            self.config.risk_tolerance = strategy["risk_tolerance"]

    # ============ 公共接口 ============

    async def submit_action(self, action_type: str, data: Dict) -> bool:
        """提交行动"""
        try:
            await asyncio.sleep(self.config.decision_delay)  # 模拟思考时间

            result = await self.comm_client.submit_action({
                "action": action_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })

            # 记录行动
            self.actions_taken.append({
                "type": action_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "success": result is not None
            })

            return result is not None

        except Exception as e:
            self.logger.error(f"Failed to submit action: {e}")
            return False

    async def submit_speech(self, content: str, metadata: Dict = None) -> bool:
        """提交发言"""
        speech_data = {
            "speaker_id": self.my_id,
            "content": content,
            "speech_round": self.game_state.get("speech_round", 0),
            "turn_order": self.game_state.get("turn_order", 0),
            "metadata": metadata or {}
        }

        success = await self.submit_action("submit_speech", speech_data)

        if success:
            self.speeches_made.append(speech_data)

        return success

    async def query_game_state(self, query_type: str = "public") -> Dict:
        """查询游戏状态"""
        try:
            if query_type == "public":
                response = await self.comm_client.query("query_public_state", {})
            elif query_type == "private":
                response = await self.comm_client.query("query_role_info", {
                    "info_type": "my_private_info"
                })
            else:
                response = await self.comm_client.query("custom_query", {
                    "type": query_type
                })

            # 更新内部状态
            if "data" in response:
                self._update_game_state(response["data"])

            return response.get("data", {})

        except Exception as e:
            self.logger.error(f"Failed to query game state: {e}")
            return {}

    def _update_game_state(self, data: Dict):
        """更新游戏状态，法官系统 → Agent内部状态"""
        # 更新公共状态
        if "alive_players" in data:
            self.game_state["alive_players"] = data["alive_players"]

        if "phase" in data:
            self.game_state["phase"] = data["phase"]

        if "day_number" in data:
            self.game_state["day"] = data["day_number"]

        # 更新玩家信息
        if "alive_players" in data:
            for player_data in data["alive_players"]:
                player_id = player_data["id"]
                if player_id not in self.known_players:
                    self.known_players[player_id] = PlayerInfo(
                        id=player_id,
                        name=player_data["name"],
                        is_ai=player_data.get("is_ai", False)
                    )

    def get_memory_summary(self, limit: int = 10) -> List[Dict]:
        """获取记忆摘要，Agent记忆系统 → 决策上下文"""
        return self.memory.get_summary(limit)

    def get_player_analysis(self, player_id: str) -> Dict:
        """获取玩家分析，多个信息源 → 综合分析结果"""
        # TODO: 实现玩家行为分析
        player = self.known_players.get(player_id)
        if not player:
            return {}

        return {
            "id": player_id,
            "name": player.name,
            "trust_score": self._calculate_trust_score(player_id),
            "behavior_patterns": self._get_behavior_patterns(player_id),
            "speech_consistency": self._analyze_speech_consistency(player_id)
        }

    def _calculate_trust_score(self, player_id: str) -> float:
        """计算信任参数"""
        # TODO: 实现信任计算
        return 0.5

    def _get_behavior_patterns(self, player_id: str) -> List[str]:
        """获取行为模式"""
        # TODO: 实现行为模式识别
        return []

    def _analyze_speech_consistency(self, player_id: str) -> float:
        """分析发言一致性"""
        # TODO: 实现发言一致性分析
        return 0.5


# ==================== 支持类定义 ====================

class AgentMemory:
    """记忆管理类"""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.event_index: Dict[str, List[MemoryEntry]] = {}

    def add_event(self, event: Dict):
        """添加事件到记忆"""
        entry = MemoryEntry(
            id=event.get("event_id", f"evt_{len(self.entries)}"),
            timestamp=event.get("timestamp", datetime.now().isoformat()),
            event_type=event.get("event_type", "unknown"),
            content=event.get("data", {}),
            importance=self._calculate_importance(event),
            tags=self._generate_tags(event)
        )

        # 添加条目
        self.entries.append(entry)

        # 更新索引
        if entry.event_type not in self.event_index:
            self.event_index[entry.event_type] = []
        self.event_index[entry.event_type].append(entry)

        # 保持条目数量不超过限制
        if len(self.entries) > self.max_entries:
            self._remove_least_important()

    def add_phase_change(self, old_phase: str, new_phase: str):
        """添加阶段变更记忆"""
        event = {
            "event_id": f"phase_{len(self.entries)}",
            "event_type": "phase_change",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "old_phase": old_phase,
                "new_phase": new_phase
            }
        }
        self.add_event(event)

    def get_summary(self, limit: int = 10) -> List[Dict]:
        """获取记忆摘要，按重要性排序"""
        sorted_entries = sorted(self.entries,
                                key=lambda x: x.importance,
                                reverse=True)
        return [asdict(entry) for entry in sorted_entries[:limit]]

    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """按标签搜索记忆"""
        results = []
        for entry in self.entries:
            if tag in entry.tags:
                results.append(entry)
        return results

    def get_recent_events(self, event_type: str = None, limit: int = 5) -> List[MemoryEntry]:
        """获取最近事件"""
        if event_type and event_type in self.event_index:
            entries = self.event_index[event_type]
        else:
            entries = self.entries

        return entries[-limit:] if entries else []

    def _calculate_importance(self, event: Dict) -> float:
        """计算事件重要性"""
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        importance_scores = {
            "phase_change": 0.8,
            "vote_result": 0.9,
            "night_reveal": 0.9,
            "player_death": 0.95,
            "player_speech": 0.3,
        }

        base_score = importance_scores.get(event_type, 0.1)

        # 根据内容调整分数
        if "result" in data and data.get("result", {}).get("exiled_player"):
            base_score += 0.1

        return min(base_score, 1.0)

    def _generate_tags(self, event: Dict) -> List[str]:
        """生成事件标签"""
        tags = [event.get("event_type", "unknown")]
        data = event.get("data", {})

        if "player_id" in data:
            tags.append(f"player_{data['player_id']}")

        if event.get("event_type") == "player_speech":
            content = data.get("content", "").lower()
            if any(word in content for word in ["狼人", "狼", "werewolf"]):
                tags.append("mentions_werewolf")
            if any(word in content for word in ["预言家", "seer"]):
                tags.append("mentions_seer")

        return tags

    def _remove_least_important(self):
        """移除最不重要的条目"""
        if not self.entries:
            return

        # 按重要性排序
        self.entries.sort(key=lambda x: x.importance)

        # 移除最不重要的条目
        removed = self.entries.pop(0)

        # 更新索引
        if removed.event_type in self.event_index:
            self.event_index[removed.event_type].remove(removed)
            if not self.event_index[removed.event_type]:
                del self.event_index[removed.event_type]


class CommunicationClient:
    """通信客户端，处理与法官系统的WebSocket通信"""

    def __init__(self, agent: 'BaseWerewolfAgent'):
        self.agent = agent
        self.ws = None
        self.pending_events = asyncio.Queue()
        self.connected = False

    async def connect(self):
        """连接到法官服务器"""
        try:
            # TODO: 实现WebSocket连接
            url = f"{self.agent.config.server_url}/game/{self.agent.config.game_id}/agent/{self.agent.config.agent_id}"
            headers = {
                "Authorization": f"Bearer {self._get_token()}",
                "X-Agent-Role": self.agent.my_role.value if self.agent.my_role else "unknown",
                "X-Game-Version": "1.0"
            }

            # 这里使用假连接模拟
            self.connected = True
            self.agent.logger.info(f"Connected to server")

        except Exception as e:
            self.agent.logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self):
        """断开连接"""
        self.connected = False
        # TODO: 关闭WebSocket连接

    async def send_heartbeat(self):
        """发送心跳"""
        if not self.connected:
            return

        heartbeat = {
            "type": "ping",
            "timestamp": datetime.now().isoformat()
        }
        # TODO: 发送心跳

    async def submit_action(self, action_data: Dict) -> Dict:
        """提交行动"""
        if not self.connected:
            return None

        try:
            # TODO: 实现行动提交
            return {"success": True, "action_id": f"act_{len(self.agent.actions_taken)}"}
        except Exception as e:
            self.agent.logger.error(f"Failed to submit action: {e}")
            return None

    async def query(self, action: str, params: Dict) -> Dict:
        """查询信息"""
        if not self.connected:
            return {}

        try:
            # TODO: 实现查询
            request_id = f"req_{len(self.agent.actions_taken)}"

            # 模拟响应
            if action == "query_public_state":
                return {
                    "response_to": request_id,
                    "data": {
                        "game_id": self.agent.config.game_id,
                        "phase": "daytime_discussion",
                        "day_number": 1,
                        "alive_players": [],
                        "dead_players": [],
                        "time_remaining": 120
                    }
                }
            else:
                return {
                    "response_to": request_id,
                    "data": {}
                }
        except Exception as e:
            self.agent.logger.error(f"Query failed: {e}")
            return {}

    def get_next_event(self) -> Optional[Dict]:
        """获取下一个待处理事件"""
        try:
            return self.pending_events.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _get_token(self) -> str:
        """获取认证令牌"""
        # TODO: 实现令牌获取逻辑
        return "mock_token"


class LLMClient:
    """LLM客户端，处理与语言模型的交互"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    async def generate_response(self, prompt: str, context: Dict = None) -> str:
        """生成文本响应"""
        # TODO: 实现LLM调用
        # 这里返回模拟响应
        return f"LLM response to: {prompt[:50]}..."

    async def analyze_situation(self, situation: Dict) -> Dict:
        """分析游戏局势"""
        # TODO: 实现局势分析
        return {
            "threat_level": 0.5,
            "recommended_action": "observe",
            "confidence": 0.7
        }

    async def make_decision(self, options: List[Dict], context: Dict) -> Dict:
        """做出决策"""
        # TODO: 实现决策生成
        if options:
            return options[0]
        return {}


class DecisionEngine:
    """决策引擎，整合记忆、策略和LLM分析"""

    def __init__(self, agent: 'BaseWerewolfAgent'):
        self.agent = agent
        self.current_strategy: Dict = {}
        self.decision_history: List[Dict] = []

    def update_strategy(self, strategy: Dict):
        """更新策略"""
        self.current_strategy.update(strategy)

    async def decide_vote_target(self) -> Optional[str]:
        """决定投票目标"""
        # 获取当前局势分析
        situation = await self._analyze_current_situation()

        # 获取可疑玩家
        suspicious_players = self._identify_suspicious_players(situation)

        # 如果没有可疑玩家，随机选择（或弃权）
        if not suspicious_players:
            return None

        # 根据策略选择目标
        if self.current_strategy.get("risk_tolerance", 0.5) > 0.7:
            # 高风险策略：选择最可疑的
            return suspicious_players[0]["id"]
        else:
            # 保守策略：选择第二可疑的（避免太明显）
            return suspicious_players[min(1, len(suspicious_players) - 1)]["id"]

    async def decide_night_action(self, role: Role, context: Dict) -> Dict:
        """决定夜晚行动"""
        decision = {
            "action_type": "none",
            "target_id": None,
            "confidence": 0.0
        }

        # 根据角色制定决策
        if role == Role.WEREWOLF:
            decision = await self._decide_werewolf_action(context)
        elif role == Role.SEER:
            decision = await self._decide_seer_action(context)
        elif role == Role.WITCH:
            decision = await self._decide_witch_action(context)

        # 记录决策
        self.decision_history.append({
            "timestamp": datetime.now().isoformat(),
            "role": role.value,
            "decision": decision
        })

        return decision

    async def _analyze_current_situation(self) -> Dict:
        """分析当前局势"""
        # TODO: 实现局势分析
        return {}

    def _identify_suspicious_players(self, situation: Dict) -> List[Dict]:
        """识别可疑玩家"""
        # TODO: 实现可疑玩家识别
        return []

    async def _decide_werewolf_action(self, context: Dict) -> Dict:
        """狼人行动决策"""
        # TODO: 实现狼人决策逻辑
        return {
            "action_type": "kill",
            "target_id": None,  # 需要具体选择
            "confidence": 0.7
        }

    async def _decide_seer_action(self, context: Dict) -> Dict:
        """预言家行动决策"""
        # TODO: 实现预言家决策逻辑
        return {
            "action_type": "check",
            "target_id": None,  # 需要具体选择
            "confidence": 0.8
        }

    async def _decide_witch_action(self, context: Dict) -> Dict:
        """女巫行动决策"""
        # TODO: 实现女巫决策逻辑
        return {
            "action_type": "save",  # 或 "poison" 或 "none"
            "target_id": None,
            "confidence": 0.6
        }


# ==================== 使用示例 ====================

class ExampleWerewolfAgent(BaseWerewolfAgent):
    """狼人Agent示例实现"""

    async def on_game_start(self):
        self.logger.info("Game started! I'm a werewolf.")

    async def on_night_action(self, phase: GamePhase):
        if phase == GamePhase.WEREWOLF_NIGHT:
            # 获取同伴信息
            partners = await self.query_game_state("werewolf_partners")

            # 制定杀人决策
            decision = await self.decision_engine.decide_night_action(
                Role.WEREWOLF,
                {"partners": partners, "memory": self.get_memory_summary()}
            )

            # 提交行动
            if decision["target_id"]:
                await self.submit_action("submit_night_action", {
                    "action_type": "kill",
                    "performer_id": self.my_id,
                    "target_id": decision["target_id"],
                    "round": self.game_state.get("day", 1)
                })

    async def on_daytime_discussion(self):
        # 生成发言
        analysis = await self.llm_client.analyze_situation({
            "memory": self.get_memory_summary(5),
            "players": list(self.known_players.values())
        })

        speech = await self.llm_client.generate_response(
            "作为狼人，我应该说什么来隐藏身份？"
        )

        # 提交发言
        await self.submit_speech(speech, {
            "strategy": "defensive",
            "emotion": "calm"
        })

    async def on_voting_phase(self):
        # 决定投票目标
        target = await self.decision_engine.decide_vote_target()

        if target:
            await self.submit_action("submit_vote", {
                "voter_id": self.my_id,
                "target_id": target,
                "round": self.game_state.get("day", 1)
            })

    async def analyze_speech(self, player_id: str, content: str):
        # 分析发言内容和意图
        analysis = await self.llm_client.analyze_situation({
            "speaker": player_id,
            "content": content,
            "context": self.game_state
        })

        # 更新玩家模型
        self._update_player_model(player_id, {
            "last_speech_analysis": analysis,
            "speech_count": self._get_player_speech_count(player_id) + 1
        })

    async def formulate_strategy(self) -> Dict[str, Any]:
        # 根据当前局势制定策略
        if self.game_state.get("day", 1) <= 2:
            return {
                "strategy": "conservative",
                "risk_tolerance": 0.3,
                "primary_goal": "survive"
            }
        else:
            return {
                "strategy": "aggressive",
                "risk_tolerance": 0.7,
                "primary_goal": "eliminate_threats"
            }


# ==================== 启动函数 ====================

async def main():
    """启动Agent示例"""
    config = AgentConfig(
        agent_id="werewolf_001",
        game_id="game_123",
        server_url="ws://judge-server",
        speech_style="moderate",
        risk_tolerance=0.5,
        max_memory_entries=100
    )

    agent = ExampleWerewolfAgent(config)

    try:
        await agent.start()

        # 运行一段时间
        await asyncio.sleep(300)  # 运行5分钟

        await agent.stop()

    except KeyboardInterrupt:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())