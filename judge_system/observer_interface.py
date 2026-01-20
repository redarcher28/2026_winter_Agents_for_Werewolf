"""
observer_interface.py
狼人杀多Agent系统 - 观察界面层完整实现
"""

import asyncio  # 异步IO核心库，用于构建异步应用
import json  # Web API标准数据格式
import logging
import time  # 传统时间同步
import uuid  # 生成唯一标识符（用于会话，任务ID）
from collections import defaultdict  # 带默认值的字典
from dataclasses import dataclass, field, asdict  # 数据类替代传统类，简化代码
from enum import Enum  # 枚举类可提升代码可读性
from pathlib import Path  # 现代路径操作
from typing import Dict, List, Optional, Any, Set, Callable  # 类型注解

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse


# ==================== 数据模型定义 ====================

class GamePhase(str, Enum):
    """游戏阶段枚举"""
    NIGHT = "night"
    DAY = "day"
    DISCUSSION = "discussion"
    VOTING = "voting"
    ENDED = "ended"


class PlayerType(str, Enum):
    """玩家类型"""
    HUMAN = "human"
    AI = "ai"


class Role(str, Enum):
    """
    角色枚举
    类型：狼人、预言家、女巫、村民、猎人
    """
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    VILLAGER = "villager"
    HUNTER = "hunter"  # 可选扩展


@dataclass
class PlayerStatus:
    """玩家状态"""
    player_id: str  # 玩家id
    player_name: str  # 玩家名称
    player_type: PlayerType  #
    role: Optional[Role] = None  # 观察者可见真实角色
    is_alive: bool = True  # 时候存活
    is_speaking: bool = False  # 是否正在发言
    vote_target: Optional[str] = None  # 投票对象
    suspicion_score: float = 0.0  # 怀疑指数 0-1
    speech_count: int = 0  # 统计该玩家的发言次数
    last_speech_time: Optional[float] = None  # 最近一次发言的时间戳


@dataclass
class GameState:
    """游戏状态快照"""
    game_id: str  # 游戏局ID
    phase: GamePhase  # 游戏流程阶段
    day_number: int  # 第几天
    alive_players: List[str]  # 存活玩家ID列表
    dead_players: List[str]  # 死亡玩家ID列表
    timestamp: float = field(default_factory=time.time)  # 时间戳，初始化为实例创建时的时间

    # 阶段特定信息
    current_speaker: Optional[str] = None  # 当前发言人
    vote_results: Optional[Dict[str, int]] = None  # 投票结果
    last_night_actions: Optional[Dict[str, str]] = None  # 角色 -> 行动目标


@dataclass
class SpeechItem:
    """发言记录"""
    speech_id: str
    player_id: str
    player_name: str
    text: str
    timestamp: float
    sentiment: Optional[str] = None  # positive/neutral/negative
    confidence: float = 1.0  # 发言置信度
    keywords: List[str] = field(default_factory=list)


# dataclasses.field 接受了一个名为 default_factory 的参数，它的作用是：如果在创建对象时没有赋值，则使用该方法初始化该字段。
# default_factory 必须是一个可以调用的无参数方法(通常为一个函数)。

@dataclass
class VoteRound:
    """投票轮次"""
    round_id: str
    day_number: int
    candidates: List[str]  # 候选人列表
    votes: Dict[str, str]  # 投票人 -> 被投人
    result: Optional[str] = None  # 被放逐玩家ID
    timestamp: float = field(default_factory=time.time)


@dataclass
class GameEvent:
    """游戏事件"""
    event_id: str  # 事件唯一标识
    event_type: str  # 事件类型（kill, vote, check, phase_change, etc.）
    player_id: Optional[str] = None  # 发起事件的玩家
    target_id: Optional[str] = None  # 事件目标 （如： 被投票的玩家）
    description: str = ""  # 事件描述（人类可读）
    timestamp: float = field(default_factory=time.time)  # 发生时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外信息


@dataclass
class AnalysisMetrics:
    """分析指标"""
    player_id: str
    wolf_probability: float = 0.0
    consistency_score: float = 0.0  # 发言一致性
    aggression_score: float = 0.0  # 攻击性
    defense_score: float = 0.0  # 防御性
    alliance_groups: List[str] = field(default_factory=list)  # 所属联盟


# ==================== WebSocket消息协议 ====================

class WSMessageType(str, Enum):
    """WebSocket消息类型"""
    GAME_STATE = "game_state"  # 游戏状态
    SPEECH = "speech"  # 发言
    VOTE_UPDATE = "vote_update"  # 投票更新
    GAME_EVENT = "game_event"  # 游戏事件
    SYSTEM_ALERT = "system_alert"  # 系统警告
    PLAYER_STATUS = "player_status"  # 玩家状态
    ANALYSIS_UPDATE = "analysis_update"  # 分析更新

@dataclass
class WSMessage:
    """
    WebSocket消息基类
    可根据实际的消息类型转换为投票类，动作类等
    """
    type: WSMessageType  # 消息类型
    channel: str  # 发送频道
    timestamp: float = field(default_factory=time.time)  # 时间戳
    data: Dict[str, Any] = field(default_factory=dict)  # 核心数据（默认空字典）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据（默认空字典）

    def to_dict(self) -> Dict[str, Any]:
        """
        将数据类对象的所有属性集合起来转换为字典
        """
        result = asdict(self)  # 遍历数据类实例的所有字段，生成一个 {字段名: 字段值} 的字典。
        result["type"] = self.type.value
        return result

    def to_json(self) -> str:
        """
        将数据类对象的所有属性集合起来转换为JSON字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ==================== 观察者权限管理 ====================

class ObserverRole(str, Enum):
    """观察者角色类型"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    VIEWER = "viewer"


@dataclass
class Observer:
    """观察者"""
    observer_id: str  # 观察员id
    username: str  # 观察员用户名
    role: ObserverRole  # 观察者角色类型
    websocket: Optional[Any] = None  # WebSocket连接对象，用于实时通信
    subscribed_channels: Set[str] = field(default_factory=set)  # 订阅的频道集合，存储频道名称
    connected_at: float = field(default_factory=time.time)  # 时间戳

    def has_permission(self, permission: str) -> bool:
        """检查权限"""
        permissions = {
            ObserverRole.ADMIN: {
                "view_all", "control_game", "reveal_roles",
                "view_wolf_chat", "export_data", "debug_tools"
            },
            ObserverRole.MODERATOR: {
                "view_all", "export_data", "annotate",
                "highlight_speech", "tag_players"
            },
            ObserverRole.VIEWER: {
                "view_public", "basic_analytics", "watch_live"
            }
        }
        return permission in permissions.get(self.role, set())
        # 注意：permissions以及他的元素本质都是集合，所以default自然返回空集合


# ==================== WebSocket连接管理器 ====================

class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, Observer] = {}  # 存储所有活跃连接：{观察者ID: Observer对象}
        self.channel_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # 频道订阅关系：{频道名: {观察者ID1, 观察者ID2, ...}}

    async def connect(self, observer: Observer, websocket: WebSocket):
        """
        连接观察者
        数据流向：
        观察者连接 → ConnectionManager.connect() → 存储连接 → 发送欢迎消息
            ↑                                        ↓
            WebSocket客户端   ←─   消息推送     ←─ 频道订阅管理
        """
        # 1. 接受WebSocket连接
        await websocket.accept()

        # 存储连接信息
        observer.websocket = websocket  # 关联WebSocket到观察者
        self.active_connections[observer.observer_id] = observer

        # 发送欢迎消息
        welcome_msg = WSMessage(
            type=WSMessageType.SYSTEM_ALERT,
            channel="system",
            data={
                "message": f"欢迎 {observer.username}! 角色: {observer.role.value}",
                "level": "info"
            }
        )
        # 向服务器发送个人信息
        await self.send_personal_message(welcome_msg, observer.observer_id)

        logging.info(f"观察者连接: {observer.username} ({observer.role})")

    def disconnect(self, observer_id: str):
        """断开连接"""
        if observer_id in self.active_connections:
            observer = self.active_connections[observer_id]
            # 从所有频道取消订阅
            for channel in observer.subscribed_channels:
                self.channel_subscriptions[channel].discard(observer_id)

            del self.active_connections[observer_id]
            logging.info(f"观察者断开: {observer.username}")

    async def send_personal_message(self, message: WSMessage, observer_id: str):
        """发送个人消息"""
        if observer_id in self.active_connections:
            observer = self.active_connections[observer_id]
            try:
                await observer.websocket.send_text(message.to_json())
            except Exception as e:
                logging.error(f"发送消息失败 {observer.username}: {e}")

    async def broadcast_to_channel(self, message: WSMessage, channel: str):
        """广播消息到频道"""
        if channel not in self.channel_subscriptions:
            return

        disconnected = []
        for observer_id in self.channel_subscriptions[channel]:
            if observer_id in self.active_connections:
                observer = self.active_connections[observer_id]
                # 检查权限过滤
                if self._check_permission_filter(message, observer):
                    try:
                        await observer.websocket.send_text(message.to_json())
                    except Exception as e:
                        logging.error(f"广播消息失败 {observer.username}: {e}")
                        disconnected.append(observer_id)
            else:
                disconnected.append(observer_id)

        # 清理断开连接
        for observer_id in disconnected:
            self.channel_subscriptions[channel].discard(observer_id)

    def _check_permission_filter(self, message: WSMessage, observer: Observer) -> bool:
        """检查权限过滤"""
        # 狼人私聊信息只对管理员可见
        if "wolf_chat" in message.channel and observer.role != ObserverRole.ADMIN:
            return False

        # AI思考过程需要调试模式
        if "ai_thoughts" in message.channel and not observer.has_permission("debug_tools"):
            return False

        return True

    def subscribe(self, observer_id: str, channel: str):
        """订阅频道"""
        if observer_id in self.active_connections:
            observer = self.active_connections[observer_id]
            observer.subscribed_channels.add(channel)  # 在观察员对象中添加他以订阅的频道
            self.channel_subscriptions[channel].add(observer_id)  # 在频道对象中添加订阅他的观察员

    def unsubscribe(self, observer_id: str, channel: str):
        """取消订阅频道"""
        if observer_id in self.active_connections:
            observer = self.active_connections[observer_id]
            observer.subscribed_channels.discard(channel)
            self.channel_subscriptions[channel].discard(observer_id)


# ==================== 游戏状态监视器 ====================

class GameStateMonitor:
    """游戏状态监视器"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 日志文件路径
        self.game_log_path = self.log_dir / "game_events.log"
        self.speech_log_path = self.log_dir / "public_speech.log"
        self.vote_log_path = self.log_dir / "vote_result.log"
        self.state_log_path = self.log_dir / "game_state.log"
        self.wolf_log_path = self.log_dir / "wolf_communication.log"

        # 状态缓存
        self.current_state: Optional[GameState] = None
        self.players: Dict[str, PlayerStatus] = {}
        self.speech_history: List[SpeechItem] = []
        self.vote_history: List[VoteRound] = []
        self.game_events: List[GameEvent] = []

        # 分析数据
        self.analysis_metrics: Dict[str, AnalysisMetrics] = {}

        # 回调函数
        # Callable 是 Python 类型注解，表示可调用对象（函数、方法、lambda表达式、实现了 __call__ 的类等）。
        # Callable[[参数类型1, 参数类型2, ...], 返回值类型]
        self.state_change_callbacks: List[Callable[[GameState], None]] = []
        self.speech_callbacks: List[Callable[[SpeechItem], None]] = []
        self.vote_callbacks: List[Callable[[VoteRound], None]] = []

        # 监控标志
        self.monitoring = False
        self.last_check_time = 0

    async def start_monitoring(self, interval: float = 1.0):
        """开始监控日志文件"""
        self.monitoring = True
        task = asyncio.create_task(self._monitor_loop(interval))
        return task

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False

    async def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                await self._check_log_updates()
                await asyncio.sleep(interval)
            except Exception as e:
                logging.error(f"监控循环错误: {e}")
                await asyncio.sleep(interval * 2)  # 错误时延长等待

    async def _check_log_updates(self):
        """检查日志更新"""
        current_time = time.time()

        # 检查游戏事件日志
        if self.game_log_path.exists():
            await self._process_game_log()

        # 检查发言日志
        if self.speech_log_path.exists():
            await self._process_speech_log()

        # 检查投票日志
        if self.vote_log_path.exists():
            await self._process_vote_log()

        # 检查状态日志
        if self.state_log_path.exists():
            await self._process_state_log()

        self.last_check_time = current_time

    async def _process_game_log(self):
        """处理游戏事件日志"""
        try:
            with open(self.game_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines[-20:]:  # 只处理最近20行（最近发生的20个事件？）

                # line.strip()：去除行首尾空白字符（空格、换行等）
                # if not ...：如果是空行或纯空白行
                # continue：跳过当前循环，处理下一行
                if not line.strip():
                    continue

                try:
                    event_data = json.loads(line)
                    event = GameEvent(
                        event_id=event_data.get("event_id", str(uuid.uuid4())),
                        event_type=event_data.get("event_type", ""),
                        player_id=event_data.get("player_id"),
                        target_id=event_data.get("target_id"),
                        description=event_data.get("description", ""),
                        timestamp=event_data.get("timestamp", time.time()),
                        metadata=event_data.get("metadata", {})
                    )

                    # 添加到历史（去重）
                    if not any(e.event_id == event.event_id for e in self.game_events):
                        self.game_events.append(event)
                        if len(self.game_events) > 100:  # 保持最近100条
                            self.game_events.pop(0)

                        # 触发回调
                        for callback in self.state_change_callbacks:
                            try:
                                callback(event)
                            except Exception as e:
                                logging.error(f"事件回调错误: {e}")

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logging.error(f"处理游戏日志错误: {e}")

    async def _process_speech_log(self):
        """处理发言日志"""
        try:
            with open(self.speech_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines[-10:]:  # 只处理最近10行
                if not line.strip():
                    continue

                try:
                    speech_data = json.loads(line)
                    speech = SpeechItem(
                        speech_id=speech_data.get("speech_id", str(uuid.uuid4())),
                        player_id=speech_data.get("player_id", ""),
                        player_name=speech_data.get("player_name", "Unknown"),
                        text=speech_data.get("text", ""),
                        timestamp=speech_data.get("timestamp", time.time()),
                        sentiment=speech_data.get("sentiment"),
                        confidence=speech_data.get("confidence", 1.0),
                        keywords=speech_data.get("keywords", [])
                    )

                    # 添加到历史（去重）
                    if not any(s.speech_id == speech.speech_id for s in self.speech_history):
                        self.speech_history.append(speech)
                        if len(self.speech_history) > 50:  # 保持最近50条
                            self.speech_history.pop(0)

                        # 更新玩家状态
                        if speech.player_id in self.players:
                            player = self.players[speech.player_id]
                            player.speech_count += 1
                            player.last_speech_time = speech.timestamp

                        # 触发回调
                        for callback in self.speech_callbacks:
                            try:
                                callback(speech)
                            except Exception as e:
                                logging.error(f"发言回调错误: {e}")

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logging.error(f"处理发言日志错误: {e}")

    async def _process_vote_log(self):
        """处理投票日志"""
        # 类似实现，处理投票日志
        pass

    async def _process_state_log(self):
        """处理状态日志"""
        try:
            with open(self.state_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                return

            # 读取最新状态
            latest_line = lines[-1].strip()
            if not latest_line:
                return

            state_data = json.loads(latest_line)

            # 创建游戏状态
            state = GameState(
                game_id=state_data.get("game_id", "default"),
                phase=GamePhase(state_data.get("phase", "night")),
                day_number=state_data.get("day_number", 1),
                alive_players=state_data.get("alive_players", []),
                dead_players=state_data.get("dead_players", []),
                timestamp=state_data.get("timestamp", time.time()),
                current_speaker=state_data.get("current_speaker"),
                vote_results=state_data.get("vote_results"),
                last_night_actions=state_data.get("last_night_actions")
            )

            # 更新状态
            if self.current_state is None or state.timestamp > self.current_state.timestamp:
                self.current_state = state

                # 触发回调
                for callback in self.state_change_callbacks:
                    try:
                        callback(state)
                    except Exception as e:
                        logging.error(f"状态回调错误: {e}")

        except Exception as e:
            logging.error(f"处理状态日志错误: {e}")

    # 对于回调函数的个人理解：callback本身是回调函数的一个等待队列。对于某个检测gamestate是否有更新的函数，
    # 一旦检测到有更新，就会把需要在更新后进行的函数加到callback清单里，清单上的函数会不断地依次执行，直到清单为零。
    # 正确理解：
    # 1. callback是函数引用的集合，不是任务队列
    # 2. 当检测到更新时，代码立即执行所有回调函数
    # _process_state_log() 内部的回调调用
    # for callback in self.state_change_callbacks:  # 遍历列表
    #     try:
    #         callback(state)  # ← **立即调用执行**，不会放入队列等待
    #     except Exception as e:
    #         logging.error(f"状态回调错误: {e}")
    def register_state_callback(self, callback: Callable[[GameState], None]):
        """注册状态变化回调"""
        self.state_change_callbacks.append(callback)

    def register_speech_callback(self, callback: Callable[[SpeechItem], None]):
        """注册发言回调"""
        self.speech_callbacks.append(callback)

    def register_vote_callback(self, callback: Callable[[VoteRound], None]):
        """注册投票回调"""
        self.vote_callbacks.append(callback)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        if not self.current_state:
            return {"error": "游戏未开始"}

        return {
            "game_state": asdict(self.current_state),
            "players": {pid: asdict(player) for pid, player in self.players.items()},
            "recent_speech": [asdict(s) for s in self.speech_history[-10:]],
            "recent_events": [asdict(e) for e in self.game_events[-20:]],
            "analysis": {pid: asdict(metrics) for pid, metrics in self.analysis_metrics.items()},
            "timestamp": time.time()
        }

    def update_player(self, player_id: str, **kwargs):
        """更新玩家信息"""
        if player_id not in self.players:
            self.players[player_id] = PlayerStatus(
                player_id=player_id,
                player_name=kwargs.get("player_name", f"Player_{player_id}"),
                player_type=PlayerType(kwargs.get("player_type", "ai"))
            )

        player = self.players[player_id]
        for key, value in kwargs.items():
            if hasattr(player, key):
                setattr(player, key, value)

    def calculate_analysis(self):
        """计算分析指标"""
        # 简化的分析计算
        for player_id, player in self.players.items():
            if player_id not in self.analysis_metrics:
                self.analysis_metrics[player_id] = AnalysisMetrics(player_id=player_id)

            metrics = self.analysis_metrics[player_id]

            # 基于发言次数和模式计算怀疑指数
            if player.speech_count > 0:
                # 这里可以添加更复杂的分析逻辑
                metrics.wolf_probability = min(0.3 + player.suspicion_score, 0.95)
                metrics.consistency_score = 0.7  # 简化


# ==================== WebSocket服务器 ====================

class WebSocketServer:
    """WebSocket服务器"""

    def __init__(self, manager: ConnectionManager, monitor: GameStateMonitor):
        self.manager = manager
        self.monitor = monitor
        self.app = FastAPI(title="狼人杀观察界面API")

        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境应该限制
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 设置路由
        self._setup_routes()

        # 注册回调
        self.monitor.register_state_callback(self._on_state_change)
        self.monitor.register_speech_callback(self._on_speech)

    # def _setup_static_files(self):
    #     """设置静态文件服务"""
    #     # 确保static目录存在
    #     static_dir = Path("static")
    #     static_dir.mkdir(exist_ok=True)
    #
    #     # 挂载静态文件目录
    #     self.app.mount("/static", StaticFiles(directory="static"), name="static")\

    def _setup_routes(self):
        """设置API路由"""

        @self.app.get("/")  # app是fast api对象
        async def root():
            """首页"""
            # 读取外部HTML文件
            html_file = Path("templates/index.html")
            if html_file.exists():
                return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
            else:
                # 如果文件不存在，返回一个简单的响应
                return HTMLResponse("""
                    <html>
                    <body>
                        <h1>狼人杀观察界面</h1>
                        <p>找不到模板文件，请确保 templates/index.html 存在</p>
                    </body>
                    </html>
                """)

        @self.app.get("/observer")
        async def observer_ui():
            """观察者界面"""
            # 读取外部HTML文件
            html_file = Path("templates/observer.html")
            if html_file.exists():
                return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
            else:
                # 如果文件不存在，返回一个简单的响应
                return HTMLResponse("""
                            <html>
                            <body>
                                <h1>观察者界面</h1>
                                <p>找不到模板文件，请确保 templates/observer.html 存在</p>
                            </body>
                            </html>
                        """)

        @self.app.websocket("/ws/observer")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket终端"""
            # 创建管理者（这里简化，实际应该从认证获取）
            observer = Observer(
                observer_id=str(uuid.uuid4()),
                username=f"Observer_{int(time.time())}",
                role=ObserverRole.ADMIN  # 默认作为管理者
            )

            await self.manager.connect(observer, websocket)

            try:
                while True:
                    # 接收消息
                    data = await websocket.receive_text()

                    try:
                        message = json.loads(data)  # 将接收到的消息用json字符串格式存储到message中
                        await self._handle_client_message(observer, message)  #
                    except json.JSONDecodeError:
                        error_msg = WSMessage(
                            type=WSMessageType.SYSTEM_ALERT,
                            channel="system",
                            data={"message": "无效的JSON格式", "level": "error"}
                        )
                        await self.manager.send_personal_message(error_msg, observer.observer_id)

            except WebSocketDisconnect:
                self.manager.disconnect(observer.observer_id)
            except Exception as e:
                logging.error(f"WebSocket错误: {e}")
                self.manager.disconnect(observer.observer_id)

        @self.app.get("/api/game/status")
        async def get_game_status():
            """获取游戏状态"""
            data = self.monitor.get_dashboard_data()
            return JSONResponse(data)

        @self.app.get("/api/players")
        async def get_players(alive_only: bool = False):
            """获取玩家列表"""
            players = self.monitor.players
            if alive_only:
                players = {pid: p for pid, p in players.items() if p.is_alive}
            return JSONResponse({"players": [asdict(p) for p in players.values()]})

        @self.app.get("/api/speech")
        async def get_speech_history(limit: int = 20):
            """获取发言历史"""
            speech = self.monitor.speech_history[-limit:]
            return JSONResponse({"speech": [asdict(s) for s in speech]})

        @self.app.get("/api/events")
        async def get_game_events(limit: int = 50):
            """获取游戏事件"""
            events = self.monitor.game_events[-limit:]
            return JSONResponse({"events": [asdict(e) for e in events]})

        @self.app.post("/api/control/next-phase")
        async def next_phase():
            """进入下一阶段（调试用）"""
            # 这里可以添加控制游戏的逻辑
            return JSONResponse({"success": True, "message": "进入下一阶段"})

        @self.app.get("/api/export/logs")
        async def export_logs():
            """导出日志"""
            # 这里可以添加导出逻辑
            return JSONResponse({"success": True, "message": "导出功能开发中"})

    # class WSMessageType(str, Enum):
    #     """WebSocket消息类型"""
    #     GAME_STATE = "game_state"  # 游戏状态
    #     SPEECH = "speech"  # 发言
    #     VOTE_UPDATE = "vote_update"  # 投票更新
    #     GAME_EVENT = "game_event"  # 游戏事件
    #     SYSTEM_ALERT = "system_alert"  # 系统警告
    #     PLAYER_STATUS = "player_status"  # 玩家状态
    #     ANALYSIS_UPDATE = "analysis_update"  # 分析更新
    async def _handle_client_message(self, observer: Observer, message: Dict[str, Any]):
        """处理客户端消息"""
        msg_type = message.get("type")

        if msg_type == "subscribe":
            channels = message.get("channels", [])
            for channel in channels:
                self.manager.subscribe(observer.observer_id, channel)

            response = WSMessage(
                type=WSMessageType.SYSTEM_ALERT,
                channel="system",
                data={"message": f"已订阅频道: {', '.join(channels)}", "level": "success"}
            )
            await self.manager.send_personal_message(response, observer.observer_id)

        elif msg_type == "unsubscribe":
            channels = message.get("channels", [])
            for channel in channels:
                self.manager.unsubscribe(observer.observer_id, channel)

        elif msg_type == "request_state":
            # 发送当前游戏状态
            if self.monitor.current_state:
                state_msg = WSMessage(
                    type=WSMessageType.GAME_STATE,
                    channel="game_state",
                    data=asdict(self.monitor.current_state)
                )
                await self.manager.send_personal_message(state_msg, observer.observer_id)

    def _on_state_change(self, state: GameState):
        """游戏状态变化回调"""
        asyncio.create_task(self._broadcast_state(state))

    def _on_speech(self, speech: SpeechItem):
        """新发言回调"""
        asyncio.create_task(self._broadcast_speech(speech))

    async def _broadcast_state(self, state: GameState):
        """广播游戏状态"""
        message = WSMessage(
            type=WSMessageType.GAME_STATE,
            channel="game_state",
            data=asdict(state)
        )
        await self.manager.broadcast_to_channel(message, "game_state")

    async def _broadcast_speech(self, speech: SpeechItem):
        """广播发言"""
        message = WSMessage(
            type=WSMessageType.SPEECH,
            channel="speech",
            data=asdict(speech)
        )
        await self.manager.broadcast_to_channel(message, "speech")

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """运行服务器"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="狼人杀观察界面服务器")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--log-dir", default="./logs", help="日志目录路径")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # 创建组件
        manager = ConnectionManager()
        monitor = GameStateMonitor(log_dir=args.log_dir)

        # 启动监控
        asyncio.run(monitor.start_monitoring())

        # 创建服务器
        server = WebSocketServer(manager, monitor)

        print(f"狼人杀观察界面服务器启动在: http://{args.host}:{args.port}")
        print(f"观察者界面: http://{args.host}:{args.port}/observer")
        print(f"API文档: http://{args.host}:{args.port}/docs")
        print("按 Ctrl+C 停止服务器")

        # 运行服务器
        server.run(host=args.host, port=args.port)

    except KeyboardInterrupt:
        print("\n服务器停止")
    except Exception as e:
        logging.error(f"服务器错误: {e}")
        raise


if __name__ == "__main__":
    main()
