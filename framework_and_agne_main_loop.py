"""
狼人杀 Agent 通用基类框架
设计原则：模块化、可扩展、角色无关
"""

from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import chromadb
import uuid
import os
import asyncio
import aiofiles
import json
import logging
import openai
import numpy as np
import re
import hashlib


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
    day: int
    phase: str
    event_type: str
    content: Dict[str, Any]
    text: str
    importance: float = 0.5   # 0.0-1.0，重要性评分
    tags: List[str] = field(default_factory=list)  # 标签，如["谎言", "投票模式", "可疑行为"]
    embedding: Optional[List[float]] = field(default=None, repr=False)

@dataclass
class StrategyDecision:
    """策略决策标准结构"""
    decision_type: str  # "speech", "vote", "night_action", "no_op"
    data: Dict[str, Any]  # 具体数据，如{"content": "..."} 或 {"target_id": "..."}
    confidence: float  # 0.0-1.0
    debug: Dict[str, Any] = None  # 可选调试信息，如{"reason": "..."}

@dataclass
class AgentConfig:
    """Agent配置"""

    # --- 身份与房间 ---
    agent_id: str # 用户及Agent身份标识
    game_id: str # 房间标识（用以避免多批次同时开始的混乱

    # --- 行为 / 策略 ---
    speech_style: str = "moderate"  # 发言风格：aggressive/moderate/conservative
    risk_tolerance: float = 0.5  # 决策参数：0.0-1.0，其中0.0为极度保守，表现为人云亦云隐藏身份，而1.0则为高度激进，狼人表现为直接悍跳带节奏，预言家发金水，村民大胆推理
    trust_threshold: float = 0.6  # 信任阈值：0.0-1.0，其中0.0为曹贼一般用人必疑，1.0为轻易信任，当信任参数超过预设的信任阈值时决定采信
    decision_delay: float = 2.0  # 模拟思考时间（秒）

    # --- 记忆与日志 ---
    max_memory_entries: int = 100  # 最大记忆条目数（定期清理旧记忆）
    log_level: str = "INFO" # 控制日志输出详细程度，"DEBUG": 最详细，用于开发和调试；"INFO": 一般信息，适合正常游戏；"WARNING": 警告信息；"ERROR": 错误信息；"CRITICAL": 严重错误
    db_path: str = "./memory_db"

    # 扩展LLM配置
    llm: Optional["LLMConfig"] = None


@dataclass
class LLMConfig:
    """LLM 模型与接口配置"""
    provider: str = "openai"          # openai / azure / local
    api_key: Optional[str] = None     # 从环境变量或配置注入
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 30.0

    # 预留扩展（如 Azure / 本地模型）
    extra: Dict[str, Any] = None




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
        self.memory = AgentMemory(config)
        self.comm_client = CommunicationClient(self)
        self.llm_client = LLMClient(config.llm)
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
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
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
                    {"config": asdict(self.config)}
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
        """建立连接（文件系统版本）"""
        await self.lifecycle_manager.transition_to(
            AgentState.CONNECTING,
            {"connection_type": "file_system"}  # 由websocket连接更改为文件系统，其中文件系统具体如何实现以及接口如何后续另做改动
        )

        await self.comm_client.connect()

        # 验证连接
        if not self.comm_client.connected:
            raise ConnectionError("Failed to establish file connection")

        # 连接成功 -> 切到 CONNECTED
        await self.lifecycle_manager.transition_to(
            AgentState.CONNECTED,
            {"connection_type": "file_system", "event_file": self.comm_client.event_file_path}
        )

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

        # 3. 等待任务完成，捕获取消异常
        results = await asyncio.gather(
            *self._tasks,
            return_exceptions=True
        )

        # 4. 记录取消的任务
        for i, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError):
                self.logger.debug(f"Task {i} was cancelled")
            elif isinstance(result, Exception):
                self.logger.error(f"Task {i} raised exception: {result}")

        # 5. 断开连接
        await self.comm_client.disconnect()

        # 6. 进入停止状态
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
                # 1. 轮询新事件
                await self.comm_client.poll_events()

                # 2. 处理待处理事件
                await self._process_pending_events()

                # 3. 检查游戏阶段
                current_phase = self.game_state.get("phase")
                if current_phase:
                    await self._handle_current_phase(current_phase)

                # 4. 更新策略
                await self._update_strategy()

                # 5. 发送心跳
                await self.comm_client.send_heartbeat()

                # 6. 定期保存记忆
                if self.performance_metrics["cycles_completed"] % 10 == 0:
                    await self._save_memory_to_file()

                # 7. 记录性能指标
                self.performance_metrics["cycles_completed"] += 1

                # 8. 动态等待
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                wait_time = max(0.01, 0.1 - cycle_time)
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


    # ============ 抽象方法（子类实现） ============

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

    async def _save_memory_to_file(self):
        """保存记忆到文件"""
        try:
            # 从内存模块获取记忆摘要
            memory_summary = self.get_memory_summary(limit=50)

            # 构建要保存的记忆数据
            memory_data = {
                "entries": [
                    {
                        "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                        "content": entry.get("text", ""),
                        "type": entry.get("event_type", "unknown"),
                        "importance": entry.get("importance", 0.5),
                        "tags": entry.get("tags", [])
                    }
                    for entry in memory_summary
                ],
                "last_updated": datetime.now().isoformat()
            }

            # 通过通信客户端保存
            await self.comm_client.save_memory(memory_data)

        except Exception as e:
            self.logger.error(f"Failed to save memory to file: {e}")

    async def _load_memory_from_file(self):
        """从文件加载记忆"""
        try:
            memory_data = await self.comm_client.load_memory()

            # 将加载的记忆添加到内存模块
            for entry in memory_data.get("entries", []):
                event = {
                    "event_id": f"loaded_{uuid.uuid4().hex[:8]}",
                    "event_type": entry.get("type", "loaded_memory"),
                    "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                    "data": {
                        "content": entry.get("content", ""),
                        "day": self.game_state.get("day", 1)
                    }
                }

                self.memory.add_event(
                    event,
                    text_description=entry.get("content", "加载的记忆条目")
                )

            self.logger.info(f"Loaded {len(memory_data.get('entries', []))} memory entries from file")

        except Exception as e:
            self.logger.error(f"Failed to load memory from file: {e}")

    async def _fetch_initial_config(self):
        """获取初始配置"""
        try:
            # 查询游戏配置
            config_resp = await self.comm_client.query("get_config", {})
            if config_resp.get("success"):
                config_data = config_resp.get("data", {})
                game_metadata = config_data.get("game_metadata", {})
                agent_settings = config_data.get("agent_settings", {})

                # 更新配置
                if agent_settings:
                    self.config.speech_style = agent_settings.get("speech_style", self.config.speech_style)
                    self.config.risk_tolerance = agent_settings.get("risk_tolerance", self.config.risk_tolerance)

            # 查询角色信息
            role_resp = await self.comm_client.query("query_role_info", {
                "info_type": "my_role"
            })

            if role_resp.get("success"):
                role_data = role_resp.get("data", {})
                role_val = role_data.get("role")

                if role_val:
                    try:
                        self.my_role = Role(role_val)
                        self.logger.info(f"Agent role assigned: {self.my_role}")
                    except ValueError:
                        self.logger.warning(f"Unknown role returned: {role_val}")
                        self.my_role = Role.VILLAGER
                else:
                    self.logger.debug("No role info returned yet")
                    self.my_role = Role.VILLAGER  # 默认角色
            else:
                self.logger.warning(f"Failed to query role info: {role_resp.get('error')}")
                self.my_role = Role.VILLAGER  # 默认角色

            # 加载历史记忆
            await self._load_memory_from_file()

        except Exception as e:
            self.logger.error(f"Failed to fetch initial config: {e}")
            self.my_role = Role.VILLAGER  # 出错时默认村民

    # 在 _process_pending_events 中增强事件数据
    async def _process_pending_events(self):
        while True:
            event = self.comm_client.get_next_event()
            if not event:
                break

            # 补充记忆模块需要的字段
            if "data" not in event:
                event["data"] = {}

            # 添加游戏日信息
            event["data"]["day"] = self.game_state.get("day", 1)

            # 添加阶段信息
            current_phase = self.game_state.get("phase")
            if current_phase:
                event["phase"] = current_phase

            # 存储到记忆
            text_description = event.get("data", {}).get("content", str(event))
            self.memory.add_event(event, text_description=text_description)


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

        # 存储到记忆（带分析标签）
        event_copy = event.copy()
        event_copy["data"] = data.copy()

        # 分析发言
        analysis = await self.analyze_speech(player_id, content)

        # 将分析结果也存入记忆
        analysis_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "speech_analysis",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "player_id": player_id,
                "original_speech": content[:100],
                "analysis": analysis,
                "day": self.game_state.get("day", 1)
            },
            "phase": self.game_state.get("phase", "unknown")
        }

        self.memory.add_event(
            analysis_event,
            text_description=f"对{player_id}号玩家发言的分析：{analysis}"
        )

        # 更新玩家模型
        self._update_player_model(player_id, {
            "last_speech": content,
            "speech_style": self._analyze_speech_style(content),
            "speech_timestamp": event["timestamp"],
            "analysis": analysis
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

    async def _handle_player_death(self, event_or_data: Dict):
        """处理玩家死亡事件"""
        """
        兼容两种输入形式：
         - event: {'event_type':'player_death', 'data': {...}}
         - data: {...}  （lifecycle manager 直接传入）
        注意：当 lifecycle manager 调用本方法时 agent.state 通常已是 DEAD，
              因此再次触发 transition 的分支会被跳过（防止循环）。
        """
        if not isinstance(event_or_data, dict):
            return
        if "data" in event_or_data and isinstance(event_or_data["data"], dict):
            data = event_or_data["data"]
        else:
            data = event_or_data

        player_id = data.get("player_id")
        self.logger.info(f"Player {player_id} died")

        if player_id in self.known_players:
            self.known_players[player_id].is_alive = False

        # 只有当自己尚未处于 DEAD 状态且发现自己死亡时，才触发 state transition
        if player_id == self.my_id and self.state != AgentState.DEAD:
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
                # on reconnect, try to update lifecycle
                if self.comm_client.connected:
                    await self.lifecycle_manager.transition_to(
                        AgentState.CONNECTED,
                        {"reason": "reconnected"}
                    )
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
        if not self._can_speak():
            return

        # 添加标志防止递归
        if getattr(self, "_processing_daytime_discussion", False):
            self.logger.debug("Already processing daytime discussion, skipping")
            return

        try:
            self._processing_daytime_discussion = True
            await self.on_daytime_discussion()
        finally:
            self._processing_daytime_discussion = False

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
            elif query_type == "recent_speeches":
                response = await self.comm_client.query("custom_query", {
                    "type": "recent_speeches",
                    "limit": 20
                })
            elif query_type == "game_history":
                response = await self.comm_client.query("custom_query", {
                    "type": "game_history",
                    "limit": 50
                })
            else:
                response = {"success": False, "data": {}}

            # 更新内部状态
            if response.get("success") and "data" in response:
                self._update_game_state(response["data"])

            return response.get("data", {})

        except Exception as e:
            self.logger.error(f"Failed to query game state: {e}")
            return {}

    async def _end_of_day_summary(self):
        """生成每日总结并存入记忆"""
        day = self.game_state.get("day", 1)

        # 获取当天所有事件
        day_events = self.memory.retrieve_day_events(day)

        # 使用LLM生成总结
        prompt = f"""
        作为{self.my_role.value if self.my_role else '玩家'}，请总结第{day}天的游戏情况：

        当天事件记录：
        {day_events}

        请分析：
        1. 今天的投票结果和原因
        2. 玩家的发言模式和可疑行为
        3. 你当前的身份推测（哪些人是好人/狼人）
        4. 明天的策略建议

        输出简洁的总结（300字以内）。
        """

        try:
            if hasattr(self.llm_client, '_call_llm'):
                summary = await self.llm_client._call_llm(prompt)
            else:
                summary = f"第{day}天总结：{day_events[:200]}..."

            # 存入记忆
            self.memory.save_summary(day, summary)
            self.logger.info(f"已生成第{day}天总结")
        except Exception as e:
            self.logger.error(f"生成每日总结失败: {e}")



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
                        name=player_data.get("name", player_id),
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
        if player_id not in self.known_players:
            return 0.5

        # 从记忆中检索与该玩家相关的事件
        player_memories = self.memory.search_by_tag(f"player_{player_id}")

        if not player_memories:
            return 0.5

        # 简单的信任计算逻辑
        trust = 0.5
        positive_indicators = 0
        negative_indicators = 0

        for memory in player_memories:
            # 检查是否有说谎的标签
            if "谎言" in memory.tags or "可疑行为" in memory.tags:
                negative_indicators += 1
            elif "诚实" in memory.tags or "合作" in memory.tags:
                positive_indicators += 1

        # 调整信任分
        total_indicators = positive_indicators + negative_indicators
        if total_indicators > 0:
            trust += (positive_indicators - negative_indicators) * 0.1

        return max(0.0, min(1.0, trust))

    def _get_behavior_patterns(self, player_id: str) -> List[str]:
        """获取行为模式"""
        patterns = []
        player_memories = self.memory.search_by_tag(f"player_{player_id}")

        # 分析投票模式
        vote_memories = [m for m in player_memories if m.event_type == "vote_result"]
        if len(vote_memories) >= 3:
            # 检查是否总是跟票
            patterns.append("跟随投票者")

        # 分析发言模式
        speech_memories = [m for m in player_memories if m.event_type == "player_speech"]
        if speech_memories:
            avg_importance = sum(m.importance for m in speech_memories) / len(speech_memories)
            if avg_importance < 0.3:
                patterns.append("低调发言")
            elif avg_importance > 0.7:
                patterns.append("积极带节奏")

        return patterns

    def _analyze_speech_consistency(self, player_id: str) -> float:
        """分析发言一致性"""
        # TODO: 实现发言一致性分析
        return 0.5


# ==================== 支持类定义 ====================

class AgentMemory:
    """
    双层记忆管理类：
    Layer 1 (RAM): self.entries -> 处理最近事件、高频逻辑查询 (速度极快)
    Layer 2 (Disk): ChromaDB -> 处理长期回忆、语义检索 (容量无限)
    """

    def __init__(self, config_or_max_entries=None, db_path="./memory_db"):
        """兼容多种初始化方式"""

        # 1. 处理配置差异
        if isinstance(config_or_max_entries, AgentConfig):
            self.config = config_or_max_entries
        elif isinstance(config_or_max_entries, int):
            self.config = AgentConfig(max_memory_entries=config_or_max_entries, db_path=db_path)
        else:
            self.config = AgentConfig(max_memory_entries=100, db_path=db_path)

        print(f"初始化双层记忆系统 (Max Entries: {self.config.max_memory_entries})...")

        # --- Layer 1: 内存层初始化 ---
        self.entries: List[MemoryEntry] = []
        self.event_index: Dict[str, List[MemoryEntry]] = {}

        # --- Layer 2: 向量层初始化 ---
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        #为每个 Agent 生成独立的数据库路径
        agent_id = getattr(self.config, 'agent_id', 'default_agent')
        if self.config.db_path == "./memory_db":
            final_db_path = f"./memory_db/{agent_id}"
        else:
            final_db_path = self.config.db_path

        print(f"数据库路径: {final_db_path}")

        self.client = chromadb.PersistentClient(path=final_db_path)
        self.collection = self.client.get_or_create_collection(name="events")

        print("记忆模块就绪。")

    def add_event(self, event: Dict, text_description: str = ""):
        """
        [核心] 添加事件：同时写入内存和向量数据库
        """
        if not text_description:
            etype = event.get("event_type")
            data = event.get("data", {})

            if etype == "player_speech":
                # 提取发言内容
                pid = data.get("player_id", "?")
                content = data.get("content", "")
                text_description = f"{pid}号玩家发言说：{content}"

            elif etype == "vote_result":
                # 提取投票结果
                votes = data.get("votes", {})
                res = data.get("result", "")
                text_description = f"投票结束。结果：{res}。详细票型：{votes}"

            elif etype == "player_death":
                pid = data.get("player_id", "?")
                text_description = f"{pid}号玩家死亡。"

            else:
                text_description = f"事件: {etype} | 数据: {data}"


        # 2. 计算衍生属性
        importance = self._calculate_importance(event)
        tags = self._generate_tags(event)

        # 3. 生成向量
        vector = self.encoder.encode(text_description).tolist()

        # 4. 准备元数据
        event_id = event.get("event_id", str(uuid.uuid4()))
        day = event.get("data", {}).get("day", 0)
        phase = event.get("phase", "unknown")
        timestamp = event.get("timestamp", datetime.now().isoformat())

        # ===========================
        # 存入 Layer 2: ChromaDB (持久化)
        # ===========================
        metadata = {
            "type": event.get("event_type", "unknown"),
            "day": day,
            "phase": phase,
            "timestamp": timestamp,
            "importance": importance
        }
        self.collection.add(
            ids=[event_id],
            embeddings=[vector],
            documents=[text_description],
            metadatas=[metadata]
        )

        # ===========================
        # 存入 Layer 1: 内存列表 (快速访问)
        # ===========================
        new_entry = MemoryEntry(
            id=event_id,
            timestamp=timestamp,
            day=day,
            phase=phase,
            event_type=event.get("event_type", "unknown"),
            content=event.get("data", {}),
            text=text_description,
            importance=importance,
            tags=tags,
            embedding=vector
        )

        self.entries.append(new_entry)

        # 更新索引
        if new_entry.event_type not in self.event_index:
            self.event_index[new_entry.event_type] = []
        self.event_index[new_entry.event_type].append(new_entry)

        # 内存限制清理
        if len(self.entries) > self.config.max_memory_entries:
            self._remove_least_important()

        print(f"[存入] Day{day} | Imp={importance:.1f} | {text_description[:40]}...")

    # ================= 逻辑检索 =================

    def add_phase_change(self, old_phase: str, new_phase: str):
        """记录阶段变更"""
        phase_map = {
            "werewolf_night": "狼人行动阶段",
            "seer_night": "预言家行动阶段",
            "witch_night": "女巫行动阶段",
            "daytime_discussion": "白天讨论阶段",
            "daytime_voting": "投票阶段",
            "game_end": "游戏结束"
        }

        old_cn = phase_map.get(old_phase, old_phase)
        new_cn = phase_map.get(new_phase, new_phase)

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "phase_change",
            "timestamp": datetime.now().isoformat(),
            "data": {"old_phase": old_phase, "new_phase": new_phase}
        }

        text = f"【系统公告】游戏阶段从 {old_cn} 变更为 {new_cn}。"
        self.add_event(event, text_description=text)

    def get_summary(self, limit: int = 10) -> List[Dict]:
        """获取内存中最重要的几条记忆"""
        sorted_entries = sorted(self.entries, key=lambda x: x.importance, reverse=True)
        return [asdict(e) for e in sorted_entries[:limit]]

    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """按标签精确搜索 (内存层)"""
        results = []
        for entry in self.entries:
            if tag in entry.tags:
                results.append(entry)
        return results

    def get_recent_events(self, event_type: str = None, limit: int = 5) -> List[MemoryEntry]:
        """获取最近发生的事件 (内存层)"""
        if event_type and event_type in self.event_index:
            entries = self.event_index[event_type]
        else:
            entries = self.entries
        return entries[-limit:] if entries else []



    def get_relevant_context(self, query: str, top_k: int = 5, day_filter: int = None, type_filter: str = None,
                             max_chars: int = 2000) -> str:
        """
        语义检索 + 逻辑过滤 + 窗口长度控制
        :param query: 用户的提问
        :param top_k: 尝试检索出的最大条数
        :param day_filter: 按天过滤 (可选)
        :param type_filter: 按类型过滤 (可选)
        :param max_chars: [新增] 返回文本的最大字符数，防止撑爆 LLM 上下文
        """
        print(f"[正在回忆] 思考: {query} (过滤条件: Day={day_filter}, Type={type_filter})")

        # 1. 构造过滤条件
        where_filter = {}
        if day_filter is not None:
            where_filter["day"] = day_filter
        if type_filter is not None:
            where_filter["type"] = type_filter
        final_where = where_filter if where_filter else None

        # 2. 检索
        query_vector = self.encoder.encode(query).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=final_where
            )
        except Exception as e:
            return f"【记忆检索】: 关于“{query}”没有找到匹配记录。"

        # 3. 提取结果
        if not results['documents'] or not results['documents'][0]:
            return f"【记忆检索】: 关于“{query}”没有找到匹配记录。"

        context_str = f"【关于“{query}”的相关记忆】:\n"
        current_len = len(context_str)
        found_docs = results['documents'][0]
        found_metas = results['metadatas'][0]

        for i, doc in enumerate(found_docs):
            day = found_metas[i].get('day', '?')
            kind = found_metas[i].get('type', 'unk')
            entry_text = f"- [Day {day} | {kind}] {doc}\n"

            if current_len + len(entry_text) > max_chars:
                context_str += "...(略)...\n"
                break

            context_str += entry_text
            current_len += len(entry_text)

        return context_str

    # ================= 内部工具方法 (私有) =================

    def _calculate_importance(self, event: Dict) -> float:
        """计算重要性 (保留组长逻辑)"""
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        # 基础分
        scores = {
            "phase_change": 0.2,  # 阶段变化没那么重要
            "vote_result": 0.9,  # 投票结果很重要
            "night_reveal": 1.0,  # 昨晚死人了，由于很重要
            "player_death": 1.0,
            "speech": 0.6,
        }
        base = scores.get(event_type, 0.5)

        res = data.get("result")

        # 只有当 result 存在，且它真的是个字典时，才去查 exiled_player
        if isinstance(res, dict) and res.get("exiled_player"):
            base += 0.1

        return min(base, 1.0)

    def _generate_tags(self, event: Dict) -> List[str]:
        """打标签"""
        # 1. 基础标签
        tags = [event.get("event_type", "unknown")]
        data = event.get("data", {})

        if "player_id" in data:
            tags.append(f"player_{data['player_id']}")

        # 2. 关键词提取
        if event.get("event_type") == "player_speech":

            content = data.get("content", "").lower()

            keywords = {
                "狼人": "mentions_werewolf", "wolf": "mentions_werewolf",
                "预言家": "mentions_seer", "seer": "mentions_seer",
                "查杀": "mentions_check",
                "女巫": "mentions_witch",
                "银水": "mentions_save",
                "金水": "mentions_good",
                "自爆": "mentions_suicide",
                "投票": "mentions_vote"
            }

            for word, tag in keywords.items():
                if word in content:
                    tags.append(tag)

        return list(set(tags))

    def retrieve_day_events(self, day: int) -> str:
        """获取某日全量记录，用于生成总结"""
        try:
            results = self.collection.get(where={"day": day})
            if not results['documents']:
                return f"第 {day} 天无记录。"
            return "\n".join([f"- {doc}" for doc in results['documents']])
        except Exception:
            return "获取记录失败。"

    def save_summary(self, day: int, summary: str):
        """存入总结"""
        self.add_event({
            "event_id": f"summary_day_{day}",
            "event_type": "daily_summary",
            "timestamp": datetime.now().isoformat(),
            "data": {"day": day, "content": summary},
            "phase": "night"
        }, text_description=f"【第{day}天总结】：{summary}")

    def _remove_least_important(self):
        """内存清理"""
        if not self.entries: return
        self.entries.sort(key=lambda x: x.importance)
        removed = self.entries.pop(0)
        if removed.event_type in self.event_index:
            try:
                self.event_index[removed.event_type].remove(removed)
            except ValueError:
                pass


class CommunicationClient:
    """通信客户端，通过文件系统与法官系统交互"""

    def __init__(self, agent: 'BaseWerewolfAgent'):
        self.agent = agent
        self.connected = False

        # 构建游戏数据目录路径
        self.game_dir = Path(f"./game_data/game_{agent.config.game_id}")

        # 公共日志文件路径
        self.game_events_log = self.game_dir / "logs" / "game_events.log"
        self.public_speech_log = self.game_dir / "logs" / "public_speech.log"
        self.vote_result_log = self.game_dir / "logs" / "vote_result.log"
        self.game_state_log = self.game_dir / "logs" / "game_state.log"

        # 私有数据路径
        self.private_dir = self.game_dir / "private" / "roles"
        self.agents_dir = self.game_dir / "agents" / agent.config.agent_id

        # Agent个人记忆文件
        self.memory_file = self.agents_dir / "memory.json"

        # 角色特定文件路径
        self.wolf_comm_log = self.private_dir / "wolf_communication.log"
        self.werewolf_file = self.private_dir / "werewolf.json"
        self.witch_file = self.private_dir / "witch.json"
        self.seer_file = self.private_dir / "seer.json"

        # 行动提交文件（Agent -> 法官）
        self.action_file = self.game_dir / "agent_actions.json"

        # 文件读取状态
        self.last_read_positions = {
            "game_events": 0,
            "public_speech": 0,
            "vote_result": 0,
            "game_state": 0,
            "wolf_comm": 0
        }

        # 事件队列
        self.pending_events = asyncio.Queue()

        # 确保必要的目录存在
        self._ensure_directories()

        # 用于跟踪已处理的事件ID
        self.processed_ids = set()

    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.game_dir / "logs",
            self.game_dir / "config",
            self.private_dir,
            self.agents_dir,
            self.agents_dir  # Agent个人目录
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    async def connect(self):
        """连接文件系统"""
        try:
            # 检查游戏目录是否存在
            if not self.game_dir.exists():
                self.agent.logger.warning(f"Game directory does not exist: {self.game_dir}")
                # 创建游戏目录（在开发环境中）
                self._ensure_directories()

            self.connected = True
            self.agent.logger.info(f"Connected to game file system: {self.game_dir}")
            return True

        except Exception as e:
            self.agent.logger.error(f"Failed to connect to file system: {e}")
            return False

    async def disconnect(self):
        """断开连接"""
        self.connected = False
        self.agent.logger.info("Disconnected from file system")

    async def send_heartbeat(self):
        """发送心跳（检查文件系统可用性）"""
        if not self.connected:
            return {"status": "disconnected"}

        try:
            # 检查关键文件是否存在
            files_to_check = [
                self.game_events_log,
                self.game_state_log,
                self.action_file
            ]

            missing_files = []
            for file_path in files_to_check:
                if not file_path.exists():
                    missing_files.append(file_path.name)

            if missing_files:
                return {
                    "status": "partial",
                    "missing_files": missing_files,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self.agent.logger.error(f"Heartbeat check failed: {e}")
            return {"status": "error", "error": str(e)}

    async def poll_events(self):
        """轮询所有日志文件，读取新事件"""
        if not self.connected:
            return

        try:
            # 轮询各个日志文件
            await self._poll_file(self.game_events_log, "game_events", self._parse_game_event)
            await self._poll_file(self.public_speech_log, "public_speech", self._parse_speech_event)
            await self._poll_file(self.vote_result_log, "vote_result", self._parse_vote_event)
            await self._poll_file(self.game_state_log, "game_state", self._parse_state_event)

            # 根据角色轮询私有文件
            if self.agent.my_role == Role.WEREWOLF:
                await self._poll_file(self.wolf_comm_log, "wolf_comm", self._parse_wolf_comm_event)

        except Exception as e:
            self.agent.logger.error(f"Error polling events: {e}")

    async def _poll_file(self, file_path: Path, file_key: str, parser_func):
        """轮询单个文件，读取新内容"""
        if not file_path.exists():
            return

        try:
            async with aiofiles.open(file_path, 'r') as f:
                # 定位到最后读取位置
                await f.seek(self.last_read_positions[file_key])

                # 读取新内容
                new_content = await f.read()

                if new_content:
                    # 更新读取位置
                    self.last_read_positions[file_key] = await f.tell()

                    # 解析每一行（JSONL格式）
                    lines = new_content.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                event_data = json.loads(line)
                                if event_data.get("event_id") not in self.processed_ids:
                                    event = parser_func(event_data)
                                    if event:
                                        await self.pending_events.put(event)
                                        self.processed_ids.add(event_data.get("event_id"))
                            except json.JSONDecodeError as e:
                                self.agent.logger.warning(f"Failed to parse JSON line: {line} - {e}")

        except Exception as e:
            self.agent.logger.error(f"Error reading file {file_path}: {e}")

    def _parse_game_event(self, data: dict) -> Optional[dict]:
        """解析游戏事件日志行"""
        return {
            "event_id": data.get("event_id", f"evt_{uuid.uuid4().hex[:8]}"),
            "event_type": data.get("event_type", "unknown"),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "data": {
                **data,
                "day": data.get("metadata", {}).get("day", self.agent.game_state.get("day", 1))
            }
        }

    def _parse_speech_event(self, data: dict) -> Optional[dict]:
        """解析发言日志行"""
        return {
            "event_id": f"speech_{uuid.uuid4().hex[:8]}",
            "event_type": "player_speech",
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "data": {
                "player_id": data.get("player_id"),
                "player_name": data.get("player_name", "未知玩家"),
                "content": data.get("text", ""),
                "sentiment": data.get("sentiment", 0.5),
                "keywords": data.get("keywords", []),
                "day": self.agent.game_state.get("day", 1)
            }
        }

    def _parse_vote_event(self, data: dict) -> Optional[dict]:
        """解析投票日志行"""
        return {
            "event_id": f"vote_{uuid.uuid4().hex[:8]}",
            "event_type": "vote_result",
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "data": {
                "round_id": data.get("round_id"),
                "day": data.get("day_number", self.agent.game_state.get("day", 1)),
                "candidates": data.get("candidates", []),
                "votes": data.get("votes", {}),
                "result": data.get("result"),
                "exiled_player": data.get("result")  # 兼容旧字段
            }
        }

    def _parse_state_event(self, data: dict) -> Optional[dict]:
        """解析游戏状态日志行"""
        # 游戏状态变化作为phase_change事件
        old_phase = self.agent.game_state.get("phase", "unknown")
        new_phase = "DAY" if data.get("phase") == "DAY" else "NIGHT"

        if old_phase != new_phase:
            return {
                "event_id": f"phase_{uuid.uuid4().hex[:8]}",
                "event_type": "phase_change",
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "data": {
                    "old_phase": old_phase,
                    "new_phase": new_phase,
                    "day": data.get("day_number", self.agent.game_state.get("day", 1)),
                    "alive_players": data.get("alive_players", []),
                    "dead_players": data.get("dead_players", [])
                }
            }
        return None

    def _parse_wolf_comm_event(self, data: dict) -> Optional[dict]:
        """解析狼人通信日志行"""
        return {
            "event_id": f"wolf_comm_{uuid.uuid4().hex[:8]}",
            "event_type": "wolf_communication",
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "data": {
                "player_id": data.get("player_id"),
                "message": data.get("message", ""),
                "day": self.agent.game_state.get("day", 1)
            }
        }

    def get_next_event(self) -> Optional[dict]:
        """获取下一个待处理事件"""
        try:
            return self.pending_events.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def query(self, query_type: str, data: Dict = None) -> Dict:
        """查询游戏信息"""
        if not self.connected:
            return {"success": False, "error": "Not connected"}

        try:
            if query_type == "get_config":
                return await self._query_config(data)
            elif query_type == "query_role_info":
                return await self._query_role_info(data)
            elif query_type == "query_public_state":
                return await self._query_public_state()
            elif query_type == "custom_query":
                return await self._custom_query(data)
            else:
                return {"success": False, "error": f"Unknown query type: {query_type}"}

        except Exception as e:
            self.agent.logger.error(f"Query failed: {e}")
            return {"success": False, "error": str(e)}

    async def _query_config(self, data: Dict) -> Dict:
        """查询配置信息"""
        metadata_file = self.game_dir / "config" / "metadata.json"

        if metadata_file.exists():
            async with aiofiles.open(metadata_file, 'r') as f:
                content = await f.read()
                metadata = json.loads(content) if content else {}
        else:
            metadata = {}

        return {
            "success": True,
            "data": {
                "game_metadata": metadata,
                "agent_settings": {
                    "speech_style": self.agent.config.speech_style,
                    "risk_tolerance": self.agent.config.risk_tolerance
                }
            }
        }

    async def _query_role_info(self, data: Dict) -> Dict:
        """查询角色信息"""
        info_type = data.get("info_type", "")

        if info_type == "my_role":
            # 从游戏元数据或角色文件获取角色
            role_file_map = {
                Role.WEREWOLF: self.werewolf_file,
                Role.SEER: self.seer_file,
                Role.WITCH: self.witch_file
            }

            # 检查角色文件，确定自己的角色
            for role, file_path in role_file_map.items():
                if file_path.exists():
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                        if content:
                            role_data = json.loads(content)
                            if role_data.get("role") == role.value:
                                # 检查自己是否在角色成员列表中
                                team_members = role_data.get("team_members", [])
                                if self.agent.config.agent_id in team_members:
                                    return {
                                        "success": True,
                                        "data": {"role": role.value}
                                    }

            # 如果没找到特定角色，默认为村民
            return {
                "success": True,
                "data": {"role": Role.VILLAGER.value}
            }

        elif info_type == "my_private_info":
            # 查询私有信息（如狼队成员、女巫药水状态等）
            return await self._query_private_info()

        else:
            return {"success": False, "error": f"Unknown info_type: {info_type}"}

    async def _query_private_info(self) -> Dict:
        """查询私有信息"""
        if self.agent.my_role == Role.WEREWOLF:
            if self.werewolf_file.exists():
                async with aiofiles.open(self.werewolf_file, 'r') as f:
                    content = await f.read()
                    if content:
                        return {
                            "success": True,
                            "data": json.loads(content)
                        }

        elif self.agent.my_role == Role.SEER:
            if self.seer_file.exists():
                async with aiofiles.open(self.seer_file, 'r') as f:
                    content = await f.read()
                    if content:
                        return {
                            "success": True,
                            "data": json.loads(content)
                        }

        elif self.agent.my_role == Role.WITCH:
            if self.witch_file.exists():
                async with aiofiles.open(self.witch_file, 'r') as f:
                    content = await f.read()
                    if content:
                        return {
                            "success": True,
                            "data": json.loads(content)
                        }

        return {
            "success": True,
            "data": {}  # 没有私有信息
        }

    async def _query_public_state(self) -> Dict:
        """查询公共游戏状态"""
        # 从game_state.log读取最新状态
        if self.game_state_log.exists():
            try:
                async with aiofiles.open(self.game_state_log, 'r') as f:
                    # 读取最后一行
                    lines = (await f.read()).strip().split('\n')
                    if lines:
                        last_line = lines[-1]
                        state_data = json.loads(last_line)

                        return {
                            "success": True,
                            "data": {
                                "alive_players": state_data.get("alive_players", []),
                                "dead_players": state_data.get("dead_players", []),
                                "phase": state_data.get("phase", "DAY"),
                                "day_number": state_data.get("day_number", 1),
                                "current_speaker": state_data.get("current_speaker"),
                                "vote_results": state_data.get("vote_results", {}),
                                "last_night_actions": state_data.get("last_night_actions", {})
                            }
                        }
            except Exception as e:
                self.agent.logger.error(f"Failed to read game state: {e}")

        # 返回默认状态
        return {
            "success": True,
            "data": {
                "alive_players": [],
                "dead_players": [],
                "phase": "DAY",
                "day_number": 1
            }
        }

    async def _custom_query(self, data: Dict) -> Dict:
        """自定义查询"""
        query_type = data.get("type", "")

        if query_type == "recent_speeches":
            # 查询最近的发言
            return await self._query_recent_speeches(data.get("limit", 10))
        elif query_type == "game_history":
            # 查询游戏历史
            return await self._query_game_history(data.get("limit", 50))
        else:
            return {"success": False, "error": f"Unknown custom query type: {query_type}"}

    async def _query_recent_speeches(self, limit: int) -> Dict:
        """查询最近的发言"""
        if not self.public_speech_log.exists():
            return {"success": True, "data": []}

        try:
            async with aiofiles.open(self.public_speech_log, 'r') as f:
                lines = (await f.read()).strip().split('\n')
                recent_speeches = []

                for line in reversed(lines[-limit:]):
                    if line.strip():
                        try:
                            speech_data = json.loads(line)
                            recent_speeches.append(speech_data)
                        except json.JSONDecodeError:
                            continue

                return {
                    "success": True,
                    "data": recent_speeches[::-1]  # 保持时间顺序
                }

        except Exception as e:
            self.agent.logger.error(f"Failed to query recent speeches: {e}")
            return {"success": False, "error": str(e)}

    async def _query_game_history(self, limit: int) -> Dict:
        """查询游戏历史"""
        if not self.game_events_log.exists():
            return {"success": True, "data": []}

        try:
            async with aiofiles.open(self.game_events_log, 'r') as f:
                lines = (await f.read()).strip().split('\n')
                history = []

                for line in reversed(lines[-limit:]):
                    if line.strip():
                        try:
                            event_data = json.loads(line)
                            history.append(event_data)
                        except json.JSONDecodeError:
                            continue

                return {
                    "success": True,
                    "data": history[::-1]  # 保持时间顺序
                }

        except Exception as e:
            self.agent.logger.error(f"Failed to query game history: {e}")
            return {"success": False, "error": str(e)}

    async def submit_action(self, action_data: Dict) -> Dict:
        """提交行动给法官系统"""
        if not self.connected:
            return {"success": False, "error": "Not connected"}

        try:
            # 构建完整的行动记录
            full_action = {
                "agent_id": self.agent.config.agent_id,
                "game_id": self.agent.config.game_id,
                "timestamp": datetime.now().isoformat(),
                **action_data
            }

            # 读取现有行动
            actions = []
            if self.action_file.exists():
                async with aiofiles.open(self.action_file, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        try:
                            existing_data = json.loads(content)
                            actions = existing_data.get("actions", [])
                        except json.JSONDecodeError:
                            actions = []

            # 添加新行动
            actions.append(full_action)

            # 写入文件
            async with aiofiles.open(self.action_file, 'w') as f:
                await f.write(json.dumps({
                    "actions": actions,
                    "last_updated": datetime.now().isoformat()
                }, indent=2, ensure_ascii=False))

            self.agent.logger.info(f"Action submitted: {action_data.get('action', 'unknown')}")

            return {
                "success": True,
                "action_id": f"act_{len(actions)}",
                "timestamp": full_action["timestamp"]
            }

        except Exception as e:
            self.agent.logger.error(f"Failed to submit action: {e}")
            return {"success": False, "error": str(e)}

    async def save_memory(self, memory_data: Dict):
        """保存Agent记忆到个人文件"""
        try:
            # 读取现有记忆
            existing_memory = {"entries": [], "last_updated": datetime.now().isoformat()}
            if self.memory_file.exists():
                async with aiofiles.open(self.memory_file, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        existing_memory = json.loads(content)

            # 添加新记忆条目
            if "entries" in memory_data:
                existing_memory["entries"].extend(memory_data["entries"])

            # 限制记忆条目数量
            max_entries = 1000  # 可配置
            if len(existing_memory["entries"]) > max_entries:
                existing_memory["entries"] = existing_memory["entries"][-max_entries:]

            # 更新最后修改时间
            existing_memory["last_updated"] = datetime.now().isoformat()

            # 写入文件
            async with aiofiles.open(self.memory_file, 'w') as f:
                await f.write(json.dumps(existing_memory, indent=2, ensure_ascii=False))

            self.agent.logger.debug(f"Memory saved to {self.memory_file}")

        except Exception as e:
            self.agent.logger.error(f"Failed to save memory: {e}")

    async def load_memory(self) -> Dict:
        """从个人文件加载Agent记忆"""
        try:
            if self.memory_file.exists():
                async with aiofiles.open(self.memory_file, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        return json.loads(content)
            return {"entries": [], "last_updated": datetime.now().isoformat()}

        except Exception as e:
            self.agent.logger.error(f"Failed to load memory: {e}")
            return {"entries": [], "last_updated": datetime.now().isoformat()}

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
        """决定投票目标 - 使用新的LLM接口"""
        # 构建上下文
        context = await self._build_decision_context()

        # 根据角色调用相应的LLM接口
        if self.agent.my_role == Role.WEREWOLF:
            decision = await self.agent.llm_client.decide_wolf_vote(context)
        elif self.agent.my_role == Role.SEER:
            decision = await self.agent.llm_client.decide_seer_vote(context)
        elif self.agent.my_role == Role.WITCH:
            decision = await self.agent.llm_client.decide_witch_vote(context)
        elif self.agent.my_role == Role.VILLAGER:
            decision = await self.agent.llm_client.decide_villager_vote(context)
        else:
            return None

        return decision.data.get("target_id")

    async def _build_decision_context(self) -> Dict:
        """构建决策上下文，集成记忆检索"""
        # 获取相关记忆（语义检索）
        relevant_info = ""
        if hasattr(self.agent.memory, 'get_relevant_context'):
            # 根据当前阶段构建不同的查询
            current_phase = self.agent.game_state.get("phase", "unknown")
            day = self.agent.game_state.get("day", 1)

            if current_phase == "daytime_discussion":
                query = "今天有哪些可疑行为和发言？谁可能是狼人？"
            elif current_phase == "daytime_voting":
                query = "谁最应该被投票出局？为什么？"
            else:  # 夜晚阶段
                query = "谁是最有价值的目标？"

            relevant_info = self.agent.memory.get_relevant_context(
                query=query,
                top_k=5,
                day_filter=day,
                max_chars=1000
            )

        # 获取最新的记忆摘要
        memory_summary = self.agent.get_memory_summary(10)

        return {
            "role": self.agent.my_role.value if self.agent.my_role else "unknown",
            "day": day,
            "phase": current_phase,
            "alive_players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "is_alive": p.is_alive,
                    "trust_score": self.agent._calculate_trust_score(p.id)  # 添加信任分
                }
                for p in self.agent.known_players.values() if p.is_alive
            ],
            "memory_summary": memory_summary,
            "semantic_memories": relevant_info,  # 语义检索结果
            "game_state": {
                "last_vote": self.agent.game_state.get("last_vote"),
                "last_night": self.agent.game_state.get("last_night"),
                "speech_round": self.agent.game_state.get("speech_round", 0),
                "phase_history": self.agent.game_phase_history[-5:]  # 最近5个阶段
            },
            "my_strategy": {
                "risk_tolerance": self.agent.config.risk_tolerance,
                "speech_style": self.agent.config.speech_style,
                "trust_threshold": self.agent.config.trust_threshold
            }
        }

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


class FileSystemMonitor:
    """文件系统监视器，用于检测文件变化"""

    def __init__(self, game_dir: Path):
        self.game_dir = game_dir
        self.file_mtimes = {}
        self.callbacks = {}

    def watch_file(self, file_path: Path, callback):
        """监视文件变化"""
        self.callbacks[str(file_path)] = callback
        if file_path.exists():
            self.file_mtimes[str(file_path)] = file_path.stat().st_mtime

    async def check_changes(self):
        """检查文件变化"""
        changed_files = []

        for file_path_str, last_mtime in self.file_mtimes.items():
            file_path = Path(file_path_str)
            if file_path.exists():
                current_mtime = file_path.stat().st_mtime
                if current_mtime > last_mtime:
                    changed_files.append(file_path)
                    self.file_mtimes[file_path_str] = current_mtime

                    # 调用回调函数
                    callback = self.callbacks.get(file_path_str)
                    if callback:
                        await callback(file_path)

        return changed_files


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
        """分析玩家发言 - 使用LLMClient现有接口"""
        # 构建分析上下文
        context = {
            "player_id": player_id,
            "content": content,
            "current_day": self.game_state.get("day", 1),
            "current_phase": self.game_state.get("phase", "unknown"),
            "my_role": self.my_role.value if self.my_role else "unknown"
        }

        # 使用现有的LLM接口进行分析
        if self.my_role == Role.WEREWOLF:
            # 狼人角度分析
            analysis_prompt = f"""
            作为狼人，分析以下发言：
            玩家{player_id}说：{content}

            请分析：
            1. 发言者的意图是什么？
            2. 是否有暴露狼人信息的风险？
            3. 这位玩家可能是好人还是狼人？

            输出JSON：{{"intent": "...", "risk_level": 0-1, "player_type_suspicion": "good/wolf/unknown", "notes": "..."}}
            """
        else:
            # 好人角度分析
            analysis_prompt = f"""
            作为{self.my_role.value if self.my_role else '玩家'}，分析以下发言：
            玩家{player_id}说：{content}

            请分析：
            1. 发言是否真诚？
            2. 逻辑是否合理？
            3. 这位玩家是否可疑？

            输出JSON：{{"sincerity": 0-1, "logic_score": 0-1, "suspicion_level": 0-1, "analysis": "..."}}
            """

        try:
            result = await self.llm_client._call_llm(analysis_prompt)
            return json.loads(result)
        except:
            return {"error": "analysis_failed", "player_id": player_id}

    async def formulate_strategy(self) -> Dict[str, Any]:
        """制定策略"""
        day = self.game_state.get("day", 1)
        alive_players = len([p for p in self.known_players.values() if p.is_alive])

        # 根据不同天数制定不同策略
        if day == 1:
            return {
                "strategy": "观察",
                "risk_tolerance": 0.3,
                "primary_goal": "收集信息",
                "speech_style": "谨慎",
                "voting_strategy": "跟随多数"
            }
        elif day == 2:
            return {
                "strategy": "积极推理",
                "risk_tolerance": 0.6,
                "primary_goal": "找出狼人",
                "speech_style": "适度激进",
                "voting_strategy": "自主判断"
            }
        else:
            return {
                "strategy": "决胜阶段",
                "risk_tolerance": 0.8,
                "primary_goal": "赢得胜利",
                "speech_style": "强力说服",
                "voting_strategy": "针对性投票"
            }


# ==================== 启动函数 ====================

async def main():
    """启动Agent示例"""
    # 创建完整的配置
    llm_config = LLMConfig(
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量获取
        model="gpt-3.5-turbo",  # 或 "gpt-4"
        temperature=0.7,
        max_tokens=500,
        timeout=30.0
    )

    config = AgentConfig(
        agent_id="werewolf_001",
        game_id="game_123",
        speech_style="moderate",
        risk_tolerance=0.5,
        trust_threshold=0.6,
        decision_delay=1.5,
        max_memory_entries=200,
        log_level="INFO",
        db_path="./memory_db/werewolf_001",  # 指定数据库路径
        llm=llm_config  # 添加LLM配置
    )

    # 确保数据库目录存在
    os.makedirs(config.db_path, exist_ok=True)

    # 确保游戏数据目录存在
    game_data_dir = Path(f"./game_data/game_{config.game_id}")
    game_data_dir.mkdir(parents=True, exist_ok=True)

    agent = ExampleWerewolfAgent(config)

    try:
        await agent.start()

        # 运行一段时间
        await asyncio.sleep(300)  # 运行5分钟

        await agent.stop()

    except KeyboardInterrupt:
        await agent.stop()
    except Exception as e:
        print(f"Agent运行出错: {e}")
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())