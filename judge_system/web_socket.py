"""
WebSocket实时通信管理器
负责管理所有WebSocket连接、消息分发、状态同步和事件处理
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
from uuid import uuid4
from enum import Enum

# 日志配置
logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket消息类型枚举"""
    # 询问类消息
    SPEECH_QUERY = "speech_query"
    ACTION_QUERY = "action_query"
    VOTE_QUERY = "vote_query"

    # 响应类消息
    SPEECH_RESPONSE = "speech_response"
    ACTION_RESPONSE = "action_response"
    VOTE_RESPONSE = "vote_response"

    # 广播类消息
    PUBLIC_MESSAGE = "public_message"
    AGENT_MESSAGE = "agent_message"

    # 系统类消息
    HEARTBEAT = "heartbeat"
    CONNECTION_STATUS = "connection_status"
    ERROR = "error"
    TIMER_START = "timer_start"
    TIMER_END = "timer_end"


class GamePhase(str, Enum):
    """游戏阶段枚举"""
    NIGHT = "night"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTING = "day_voting"
    DAY_RESULT = "day_result"
    GAME_END = "game_end"


class PlayerRole(str, Enum):
    """玩家角色枚举"""
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    VILLAGER = "villager"


class WebSocketManager:
    """
    WebSocket管理器类
    单例模式，负责管理所有WebSocket连接和消息处理
    """

    # 单例实例
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化WebSocket管理器"""
        if self._initialized:
            return

        # 连接管理
        self.connections: Dict[str, Dict[str, Any]] = {}  # {game_id: {player_id: websocket}}
        self.player_info: Dict[str, Dict[str, Any]] = {}  # {player_id: {role, game_id, ...}}

        # 消息队列
        self.message_queues: Dict[str, asyncio.Queue] = {}  # {player_id: Queue}
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()

        # 计时器管理
        self.timers: Dict[str, Dict[str, Any]] = {}  # {timer_id: {task, callback, ...}}
        self.active_timers: Dict[str, Set[str]] = {}  # {game_id: {timer_id, ...}}

        # 回调函数注册
        self.callbacks: Dict[str, Callable] = {
            "on_action_response": None,
            "on_speech_response": None,
            "on_vote_response": None,
            "on_agent_disconnected": None,
            "on_game_state_updated": None,
        }

        # 游戏状态同步
        self.game_states: Dict[str, Dict[str, Any]] = {}  # {game_id: game_state}

        # 公示板连接
        public_display_connections: Set[Any] = set()

        # 心跳管理
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self.heartbeat_intervals: Dict[str, float] = {}  # {player_id: interval}

        # 锁和同步机制
        self.locks: Dict[str, asyncio.Lock] = {}

        self._initialized = True
        logger.info("WebSocketManager初始化完成")

    # ==================== 连接管理方法 ====================

    async def register_connection(
            self,
            game_id: str,
            player_id: str,
            websocket: Any,
            player_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        注册新的WebSocket连接

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            websocket: WebSocket连接对象
            player_info: 玩家信息

        Returns:
            注册结果
        """
        # 初始化游戏连接记录
        if game_id not in self.connections:
            self.connections[game_id] = {}
            self.active_timers[game_id] = set()

        # 存储连接
        self.connections[game_id][player_id] = {
            "websocket": websocket,
            "connected_at": datetime.now(),
            "last_heartbeat": datetime.now(),
            "status": "connected"
        }

        # 存储玩家信息
        self.player_info[player_id] = {
            **player_info,
            "game_id": game_id,
            "connected": True
        }

        # 初始化消息队列
        if player_id not in self.message_queues:
            self.message_queues[player_id] = asyncio.Queue()

        # 初始化锁
        if game_id not in self.locks:
            self.locks[game_id] = asyncio.Lock()

        # 启动心跳检测
        await self._start_heartbeat_check(player_id)

        # 发送连接成功消息
        await self.send_connection_status(player_id, "connected")

        logger.info(f"玩家 {player_id} 注册连接成功，游戏 {game_id}")
        return {"success": True, "player_id": player_id, "game_id": game_id}

    async def unregister_connection(self, game_id: str, player_id: str) -> Dict[str, Any]:
        """
        注销WebSocket连接

        Args:
            game_id: 游戏ID
            player_id: 玩家ID

        Returns:
            注销结果
        """
        try:
            # 停止心跳检测
            await self._stop_heartbeat_check(player_id)

            # 移除连接
            if game_id in self.connections and player_id in self.connections[game_id]:
                del self.connections[game_id][player_id]
                logger.info(f"玩家 {player_id} 连接已移除")

            # 更新玩家状态
            if player_id in self.player_info:
                self.player_info[player_id]["connected"] = False

            # 触发断开连接回调
            if self.callbacks["on_agent_disconnected"]:
                await self.callbacks["on_agent_disconnected"](game_id, player_id)

            # 如果游戏没有活跃连接，清理资源
            if game_id in self.connections and not self.connections[game_id]:
                await self._cleanup_game_resources(game_id)

            return {"success": True, "player_id": player_id}

        except Exception as e:
            logger.error(f"注销连接失败: {e}")
            return {"success": False, "error": str(e)}

    async def get_connection_status(self, game_id: str, player_id: str) -> Dict[str, Any]:
        """
        获取连接状态

        Args:
            game_id: 游戏ID
            player_id: 玩家ID

        Returns:
            连接状态信息
        """
        if (game_id in self.connections and
                player_id in self.connections[game_id]):
            conn = self.connections[game_id][player_id]
            return {
                "connected": True,
                "connected_at": conn["connected_at"].isoformat(),
                "last_heartbeat": conn["last_heartbeat"].isoformat(),
                "status": conn["status"]
            }
        return {"connected": False}

    # ==================== 消息发送方法 ====================

    async def send_message(
            self,
            game_id: str,
            player_id: str,
            message_type: str,
            data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        向指定玩家发送消息

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            message_type: 消息类型
            data: 消息数据

        Returns:
            发送结果
        """
        try:
            # 检查连接是否存在
            if (game_id not in self.connections or
                    player_id not in self.connections[game_id]):
                logger.warning(f"玩家 {player_id} 未连接，无法发送消息")
                return {"success": False, "error": "Player not connected"}

            # 获取连接
            connection = self.connections[game_id][player_id]
            websocket = connection["websocket"]

            # 构造消息
            message = {
                "type": message_type,
                "message_id": str(uuid4()),
                "timestamp": datetime.now().isoformat(),
                "data": data
            }

            # 发送消息
            await websocket.send_json(message)

            logger.debug(f"消息发送成功: {message_type} -> {player_id}")
            return {"success": True, "message_id": message["message_id"]}

        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return {"success": False, "error": str(e)}

    async def broadcast_to_game(
            self,
            game_id: str,
            message_type: str,
            data: Dict[str, Any],
            exclude_players: List[str] = None
    ) -> Dict[str, Any]:
        """
        向游戏内所有玩家广播消息

        Args:
            game_id: 游戏ID
            message_type: 消息类型
            data: 消息数据
            exclude_players: 排除的玩家列表

        Returns:
            广播结果
        """
        if exclude_players is None:
            exclude_players = []

        results = []
        if game_id in self.connections:
            for player_id, connection in self.connections[game_id].items():
                if player_id in exclude_players:
                    continue

                try:
                    result = await self.send_message(game_id, player_id, message_type, data)
                    results.append({
                        "player_id": player_id,
                        "success": result["success"],
                        "message_id": result.get("message_id")
                    })
                except Exception as e:
                    results.append({
                        "player_id": player_id,
                        "success": False,
                        "error": str(e)
                    })

        return {
            "success": all(r["success"] for r in results),
            "results": results
        }

    async def send_to_public_display(
            self,
            game_id: str,
            message_type: str,
            data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        向公示板发送消息

        Args:
            game_id: 游戏ID
            message_type: 消息类型
            data: 消息数据

        Returns:
            发送结果
        """
        # 这里假设公示板也有一个特殊的WebSocket连接
        # 实际实现中可能需要特殊处理
        return await self.broadcast_to_game(
            game_id,
            MessageType.PUBLIC_MESSAGE.value,
            {
                "phase": data.get("phase", "unknown"),
                "content": data
            }
        )

    # ==================== 询问接口方法 ====================

    async def ask_for_speech(
            self,
            game_id: str,
            player_id: str,
            time_limit: int = 60,
            prompt: str = "请发表你的观点"
    ) -> Dict[str, Any]:
        """
        询问玩家发言

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            time_limit: 时间限制（秒）
            prompt: 提示词

        Returns:
            询问结果
        """
        # 构造询问消息
        message_data = {
            "phase": GamePhase.DAY_DISCUSSION.value,
            "player_id": player_id,
            "time_limit": time_limit,
            "prompt": prompt,
            "current_time": datetime.now().isoformat()
        }

        # 发送询问
        result = await self.send_message(
            game_id,
            player_id,
            MessageType.SPEECH_QUERY.value,
            message_data
        )

        if result["success"]:
            # 启动计时器
            timer_id = await self._start_timer(
                game_id,
                player_id,
                "speech",
                time_limit,
                on_timeout=self._handle_speech_timeout
            )

            result["timer_id"] = timer_id

        return result

    async def ask_for_action(
            self,
            game_id: str,
            player_id: str,
            role: str,
            action_type: str,
            options: List[str],
            time_limit: int = 30,
            prompt: str = "请选择你的行动"
    ) -> Dict[str, Any]:
        """
        询问玩家行动

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            role: 玩家角色
            action_type: 行动类型
            options: 可用选项列表
            time_limit: 时间限制（秒）
            prompt: 提示词

        Returns:
            询问结果
        """
        # 构造询问消息
        message_data = {
            "phase": GamePhase.NIGHT.value,
            "role": role,
            "action_type": action_type,
            "options": options,
            "time_limit": time_limit,
            "prompt": prompt,
            "current_time": datetime.now().isoformat()
        }

        # 发送询问
        result = await self.send_message(
            game_id,
            player_id,
            MessageType.ACTION_QUERY.value,
            message_data
        )

        if result["success"]:
            # 启动计时器
            timer_id = await self._start_timer(
                game_id,
                player_id,
                "action",
                time_limit,
                on_timeout=self._handle_action_timeout
            )

            result["timer_id"] = timer_id

        return result

    async def ask_for_vote(
            self,
            game_id: str,
            player_id: str,
            candidates: List[str],
            time_limit: int = 30,
            prompt: str = "请投票选择要放逐的玩家"
    ) -> Dict[str, Any]:
        """
        询问玩家投票

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            candidates: 候选人列表
            time_limit: 时间限制（秒）
            prompt: 提示词

        Returns:
            询问结果
        """
        # 构造询问消息
        message_data = {
            "phase": GamePhase.DAY_VOTING.value,
            "candidates": candidates,
            "time_limit": time_limit,
            "prompt": prompt,
            "current_time": datetime.now().isoformat()
        }

        # 发送询问
        result = await self.send_message(
            game_id,
            player_id,
            MessageType.VOTE_QUERY.value,
            message_data
        )

        if result["success"]:
            # 启动计时器
            timer_id = await self._start_timer(
                game_id,
                player_id,
                "vote",
                time_limit,
                on_timeout=self._handle_vote_timeout
            )

            result["timer_id"] = timer_id

        return result

    # ==================== 接收接口方法 ====================

    async def handle_message(self, game_id: str, player_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理接收到的消息

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            message: 接收到的消息

        Returns:
            处理结果
        """
        try:
            message_type = message.get("type")
            data = message.get("data", {})

            # 更新心跳时间
            if message_type != MessageType.HEARTBEAT.value:
                await self._update_heartbeat(player_id)

            # 根据消息类型处理
            if message_type == MessageType.SPEECH_RESPONSE.value:
                return await self._handle_speech_response(game_id, player_id, data)

            elif message_type == MessageType.ACTION_RESPONSE.value:
                return await self._handle_action_response(game_id, player_id, data)

            elif message_type == MessageType.VOTE_RESPONSE.value:
                return await self._handle_vote_response(game_id, player_id, data)

            elif message_type == MessageType.HEARTBEAT.value:
                return await self._handle_heartbeat(game_id, player_id, data)

            else:
                logger.warning(f"未知消息类型: {message_type}")
                return {"success": False, "error": f"Unknown message type: {message_type}"}

        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            return {"success": False, "error": str(e)}

    async def _handle_speech_response(self, game_id: str, player_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理发言响应

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            data: 响应数据

        Returns:
            处理结果
        """
        # 停止计时器
        await self._stop_player_timer(game_id, player_id, "speech")

        # 验证响应数据
        if "content" not in data:
            return {"success": False, "error": "Missing content in speech response"}

        # 触发回调
        if self.callbacks["on_speech_response"]:
            result = await self.callbacks["on_speech_response"](game_id, player_id, data)
        else:
            result = {"success": True, "player_id": player_id, "content": data["content"]}

        # 广播发言内容
        if result["success"]:
            await self.broadcast_to_game(
                game_id,
                MessageType.AGENT_MESSAGE.value,
                {
                    "speaker": player_id,
                    "speech": data["content"],
                    "timestamp": datetime.now().isoformat()
                },
                exclude_players=[player_id]  # 不重复发送给发言者
            )

        return result

    async def _handle_action_response(self, game_id: str, player_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理行动响应

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            data: 响应数据

        Returns:
            处理结果
        """
        # 停止计时器
        await self._stop_player_timer(game_id, player_id, "action")

        # 验证响应数据
        required_fields = ["action", "target"]
        for field in required_fields:
            if field not in data:
                return {"success": False, "error": f"Missing {field} in action response"}

        # 触发回调
        if self.callbacks["on_action_response"]:
            return await self.callbacks["on_action_response"](game_id, player_id, data)
        else:
            return {"success": True, "player_id": player_id, "action": data}

    async def _handle_vote_response(self, game_id: str, player_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理投票响应

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            data: 响应数据

        Returns:
            处理结果
        """
        # 停止计时器
        await self._stop_player_timer(game_id, player_id, "vote")

        # 验证响应数据
        if "target" not in data:
            return {"success": False, "error": "Missing target in vote response"}

        # 触发回调
        if self.callbacks["on_vote_response"]:
            return await self.callbacks["on_vote_response"](game_id, player_id, data)
        else:
            return {"success": True, "player_id": player_id, "vote": data["target"]}

    # ==================== 游戏阶段流程方法 ====================

    async def process_night_phase(
            self,
            game_id: str,
            werewolf_players: List[str],
            seer_players: List[str],
            witch_players: List[str]
    ) -> Dict[str, Any]:
        """
        处理夜晚阶段

        Args:
            game_id: 游戏ID
            werewolf_players: 狼人玩家列表
            seer_players: 预言家玩家列表
            witch_players: 女巫玩家列表

        Returns:
            夜晚阶段结果
        """
        results = {
            "werewolf_actions": [],
            "seer_actions": [],
            "witch_actions": [],
            "errors": []
        }

        # 1. 狼人行动
        for werewolf_id in werewolf_players:
            try:
                # 获取存活非狼人玩家列表
                alive_non_werewolves = await self._get_alive_non_role_players(game_id, PlayerRole.WEREWOLF.value)

                if not alive_non_werewolves:
                    results["errors"].append(f"No valid targets for werewolf {werewolf_id}")
                    continue

                # 询问狼人行动
                action_result = await self.ask_for_action(
                    game_id=game_id,
                    player_id=werewolf_id,
                    role=PlayerRole.WEREWOLF.value,
                    action_type="kill",
                    options=alive_non_werewolves,
                    time_limit=30,
                    prompt="请选择你要杀害的目标"
                )

                if action_result["success"]:
                    results["werewolf_actions"].append({
                        "player_id": werewolf_id,
                        "timer_id": action_result.get("timer_id"),
                        "status": "asked"
                    })
                else:
                    results["errors"].append(f"Failed to ask werewolf {werewolf_id}: {action_result.get('error')}")

            except Exception as e:
                results["errors"].append(f"Error processing werewolf {werewolf_id}: {str(e)}")

        # 2. 预言家行动
        for seer_id in seer_players:
            try:
                # 获取存活玩家列表
                alive_players = await self._get_alive_players(game_id)

                if not alive_players:
                    results["errors"].append(f"No valid targets for seer {seer_id}")
                    continue

                # 询问预言家行动
                action_result = await self.ask_for_action(
                    game_id=game_id,
                    player_id=seer_id,
                    role=PlayerRole.SEER.value,
                    action_type="check",
                    options=alive_players,
                    time_limit=30,
                    prompt="请选择你要查验的玩家"
                )

                if action_result["success"]:
                    results["seer_actions"].append({
                        "player_id": seer_id,
                        "timer_id": action_result.get("timer_id"),
                        "status": "asked"
                    })
                else:
                    results["errors"].append(f"Failed to ask seer {seer_id}: {action_result.get('error')}")

            except Exception as e:
                results["errors"].append(f"Error processing seer {seer_id}: {str(e)}")

        # 3. 女巫行动
        for witch_id in witch_players:
            try:
                # 获取存活玩家列表
                alive_players = await self._get_alive_players(game_id)

                if not alive_players:
                    results["errors"].append(f"No valid targets for witch {witch_id}")
                    continue

                # 女巫有多个选项：救人、毒人、放弃
                action_result = await self.ask_for_action(
                    game_id=game_id,
                    player_id=witch_id,
                    role=PlayerRole.WITCH.value,
                    action_type="witch_action",
                    options=["save", "poison", "abandon"],
                    time_limit=45,
                    prompt="请选择你的行动：救人、毒人、或放弃"
                )

                if action_result["success"]:
                    results["witch_actions"].append({
                        "player_id": witch_id,
                        "timer_id": action_result.get("timer_id"),
                        "status": "asked"
                    })
                else:
                    results["errors"].append(f"Failed to ask witch {witch_id}: {action_result.get('error')}")

            except Exception as e:
                results["errors"].append(f"Error processing witch {witch_id}: {str(e)}")

        return results

    async def process_day_speech_phase(
            self,
            game_id: str,
            speech_order: List[str],
            speech_time_limit: int = 60
    ) -> Dict[str, Any]:
        """
        处理白天发言阶段

        Args:
            game_id: 游戏ID
            speech_order: 发言顺序列表
            speech_time_limit: 发言时间限制

        Returns:
            发言阶段结果
        """
        results = {
            "speeches": [],
            "errors": []
        }

        # 依次询问每个玩家发言
        for player_id in speech_order:
            try:
                # 检查玩家是否存活
                if not await self._is_player_alive(game_id, player_id):
                    logger.info(f"玩家 {player_id} 已死亡，跳过发言")
                    continue

                # 询问发言
                speech_result = await self.ask_for_speech(
                    game_id=game_id,
                    player_id=player_id,
                    time_limit=speech_time_limit,
                    prompt=f"玩家 {player_id}，请发表你的观点"
                )

                if speech_result["success"]:
                    results["speeches"].append({
                        "player_id": player_id,
                        "timer_id": speech_result.get("timer_id"),
                        "status": "asked",
                        "order": speech_order.index(player_id) + 1
                    })
                else:
                    results["errors"].append(f"Failed to ask speech from {player_id}: {speech_result.get('error')}")

            except Exception as e:
                results["errors"].append(f"Error processing speech for {player_id}: {str(e)}")

        return results

    async def process_day_voting_phase(
            self,
            game_id: str,
            alive_players: List[str],
            vote_time_limit: int = 30
    ) -> Dict[str, Any]:
        """
        处理白天投票阶段

        Args:
            game_id: 游戏ID
            alive_players: 存活玩家列表
            vote_time_limit: 投票时间限制

        Returns:
            投票阶段结果
        """
        results = {
            "votes_asked": [],
            "errors": []
        }

        # 向所有存活玩家询问投票
        for player_id in alive_players:
            try:
                # 投票候选人（排除自己）
                candidates = [p for p in alive_players if p != player_id]

                if not candidates:
                    results["errors"].append(f"No valid candidates for {player_id}")
                    continue

                # 询问投票
                vote_result = await self.ask_for_vote(
                    game_id=game_id,
                    player_id=player_id,
                    candidates=candidates,
                    time_limit=vote_time_limit,
                    prompt="请投票选择要放逐的玩家"
                )

                if vote_result["success"]:
                    results["votes_asked"].append({
                        "player_id": player_id,
                        "timer_id": vote_result.get("timer_id"),
                        "status": "asked"
                    })
                else:
                    results["errors"].append(f"Failed to ask vote from {player_id}: {vote_result.get('error')}")

            except Exception as e:
                results["errors"].append(f"Error asking vote from {player_id}: {str(e)}")

        return results

    # ==================== 计时器管理方法 ====================

    async def _start_timer(
            self,
            game_id: str,
            player_id: str,
            timer_type: str,
            duration: int,
            on_timeout: Callable[[str, str, str], Awaitable[Dict[str, Any]]]
    ) -> str:
        """
        启动计时器

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            timer_type: 计时器类型
            duration: 持续时间（秒）
            on_timeout: 超时回调函数

        Returns:
            计时器ID
        """
        timer_id = f"{game_id}_{player_id}_{timer_type}_{uuid4().hex[:8]}"

        async def timer_task():
            try:
                await asyncio.sleep(duration)

                # 超时处理
                if timer_id in self.timers:
                    timeout_result = await on_timeout(game_id, player_id, timer_type)

                    # 发送超时通知
                    await self.send_message(
                        game_id,
                        player_id,
                        MessageType.TIMER_END.value,
                        {
                            "timer_id": timer_id,
                            "timer_type": timer_type,
                            "reason": "timeout",
                            "result": timeout_result
                        }
                    )

                    # 清理计时器
                    if timer_id in self.timers:
                        del self.timers[timer_id]
                    if game_id in self.active_timers and timer_id in self.active_timers[game_id]:
                        self.active_timers[game_id].remove(timer_id)

            except asyncio.CancelledError:
                # 计时器被取消
                pass
            except Exception as e:
                logger.error(f"计时器任务异常: {e}")

        # 创建并启动计时器任务
        task = asyncio.create_task(timer_task())

        # 存储计时器信息
        self.timers[timer_id] = {
            "task": task,
            "game_id": game_id,
            "player_id": player_id,
            "timer_type": timer_type,
            "started_at": datetime.now(),
            "duration": duration,
            "on_timeout": on_timeout
        }

        # 添加到活跃计时器集合
        if game_id not in self.active_timers:
            self.active_timers[game_id] = set()
        self.active_timers[game_id].add(timer_id)

        # 发送计时器开始消息
        await self.send_message(
            game_id,
            player_id,
            MessageType.TIMER_START.value,
            {
                "timer_id": timer_id,
                "timer_type": timer_type,
                "duration": duration,
                "started_at": datetime.now().isoformat()
            }
        )

        logger.debug(f"计时器启动: {timer_id}, 持续时间: {duration}秒")
        return timer_id

    async def _stop_timer(self, timer_id: str) -> bool:
        """
        停止计时器

        Args:
            timer_id: 计时器ID

        Returns:
            是否成功停止
        """
        if timer_id in self.timers:
            timer_info = self.timers[timer_id]
            timer_info["task"].cancel()

            # 清理计时器记录
            game_id = timer_info["game_id"]
            if game_id in self.active_timers and timer_id in self.active_timers[game_id]:
                self.active_timers[game_id].remove(timer_id)

            del self.timers[timer_id]
            logger.debug(f"计时器停止: {timer_id}")
            return True

        return False

    async def _stop_player_timer(self, game_id: str, player_id: str, timer_type: str) -> bool:
        """
        停止玩家的特定类型计时器

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            timer_type: 计时器类型

        Returns:
            是否成功停止
        """
        timer_ids_to_stop = []

        # 查找匹配的计时器
        for timer_id, timer_info in self.timers.items():
            if (timer_info["game_id"] == game_id and
                    timer_info["player_id"] == player_id and
                    timer_info["timer_type"] == timer_type):
                timer_ids_to_stop.append(timer_id)

        # 停止所有匹配的计时器
        results = []
        for timer_id in timer_ids_to_stop:
            result = await self._stop_timer(timer_id)
            results.append(result)

        return all(results)

    async def _handle_action_timeout(self, game_id: str, player_id: str, timer_type: str) -> Dict[str, Any]:
        """
        处理行动超时

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            timer_type: 计时器类型

        Returns:
            超时处理结果
        """
        logger.warning(f"玩家 {player_id} {timer_type} 行动超时")

        # 根据角色决定默认行动
        player_role = self.player_info.get(player_id, {}).get("role")

        if player_role == PlayerRole.WEREWOLF.value:
            # 狼人超时，随机选择目标
            alive_non_werewolves = await self._get_alive_non_role_players(game_id, PlayerRole.WEREWOLF.value)
            default_target = alive_non_werewolves[0] if alive_non_werewolves else None

            return {
                "action": "kill",
                "target": default_target,
                "is_timeout_default": True,
                "reason": "action_timeout"
            }

        elif player_role == PlayerRole.SEER.value:
            # 预言家超时，不验人
            return {
                "action": "check",
                "target": None,
                "is_timeout_default": True,
                "reason": "action_timeout"
            }

        elif player_role == PlayerRole.WITCH.value:
            # 女巫超时，放弃行动
            return {
                "action": "abandon",
                "target": None,
                "is_timeout_default": True,
                "reason": "action_timeout"
            }

        return {"action": "abandon", "target": None, "is_timeout_default": True}

    async def _handle_speech_timeout(self, game_id: str, player_id: str, timer_type: str) -> Dict[str, Any]:
        """
        处理发言超时

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            timer_type: 计时器类型

        Returns:
            超时处理结果
        """
        logger.warning(f"玩家 {player_id} 发言超时")

        # 默认发言内容
        default_speech = "（玩家发言超时）"

        return {
            "content": default_speech,
            "is_timeout_default": True,
            "reason": "speech_timeout"
        }

    async def _handle_vote_timeout(self, game_id: str, player_id: str, timer_type: str) -> Dict[str, Any]:
        """
        处理投票超时

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            timer_type: 计时器类型

        Returns:
            超时处理结果
        """
        logger.warning(f"玩家 {player_id} 投票超时")

        # 默认投票（弃权）
        return {
            "target": None,  # 弃权
            "is_timeout_default": True,
            "reason": "vote_timeout"
        }

    # ==================== 心跳管理方法 ====================

    async def _start_heartbeat_check(self, player_id: str, interval: float = 30.0):
        """
        启动心跳检测

        Args:
            player_id: 玩家ID
            interval: 检测间隔（秒）
        """
        self.heartbeat_intervals[player_id] = interval

        async def heartbeat_task():
            while player_id in self.heartbeat_intervals:
                try:
                    await asyncio.sleep(interval)

                    # 检查最后一次心跳时间
                    if player_id in self.player_info:
                        game_id = self.player_info[player_id].get("game_id")
                        if game_id and game_id in self.connections:
                            if player_id in self.connections[game_id]:
                                conn_info = self.connections[game_id][player_id]
                                last_heartbeat = conn_info["last_heartbeat"]

                                # 计算时间差
                                time_diff = (datetime.now() - last_heartbeat).total_seconds()

                                if time_diff > interval * 2:  # 超过2个间隔没有心跳
                                    logger.warning(f"玩家 {player_id} 心跳超时，断开连接")
                                    await self.unregister_connection(game_id, player_id)
                                    break

                except Exception as e:
                    logger.error(f"心跳检测异常: {e}")
                    break

        # 启动心跳检测任务
        task = asyncio.create_task(heartbeat_task())
        self.heartbeat_tasks[player_id] = task

    async def _stop_heartbeat_check(self, player_id: str):
        """
        停止心跳检测

        Args:
            player_id: 玩家ID
        """
        if player_id in self.heartbeat_tasks:
            self.heartbeat_tasks[player_id].cancel()
            del self.heartbeat_tasks[player_id]

        if player_id in self.heartbeat_intervals:
            del self.heartbeat_intervals[player_id]

    async def _update_heartbeat(self, player_id: str):
        """
        更新心跳时间

        Args:
            player_id: 玩家ID
        """
        if player_id in self.player_info:
            game_id = self.player_info[player_id].get("game_id")
            if game_id and game_id in self.connections:
                if player_id in self.connections[game_id]:
                    self.connections[game_id][player_id]["last_heartbeat"] = datetime.now()

    async def _handle_heartbeat(self, game_id: str, player_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理心跳消息

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            data: 心跳数据

        Returns:
            处理结果
        """
        await self._update_heartbeat(player_id)
        return {"success": True, "player_id": player_id, "type": "heartbeat_ack"}

    # ==================== 回调注册方法 ====================

    def register_callback(self, callback_name: str, callback_func: Callable) -> bool:
        """
        注册回调函数

        Args:
            callback_name: 回调名称
            callback_func: 回调函数

        Returns:
            是否注册成功
        """
        if callback_name in self.callbacks:
            self.callbacks[callback_name] = callback_func
            logger.info(f"回调函数注册成功: {callback_name}")
            return True
        else:
            logger.error(f"未知的回调名称: {callback_name}")
            return False

    def unregister_callback(self, callback_name: str) -> bool:
        """
        注销回调函数

        Args:
            callback_name: 回调名称

        Returns:
            是否注销成功
        """
        if callback_name in self.callbacks:
            self.callbacks[callback_name] = None
            logger.info(f"回调函数注销成功: {callback_name}")
            return True
        else:
            logger.error(f"未知的回调名称: {callback_name}")
            return False

    # ==================== 辅助方法 ====================

    async def _get_alive_players(self, game_id: str) -> List[str]:
        """
        获取游戏中的存活玩家列表

        Args:
            game_id: 游戏ID

        Returns:
            存活玩家ID列表
        """
        # 这里需要从游戏状态中获取存活玩家
        # 简化实现，假设所有连接的玩家都是存活的
        if game_id in self.connections:
            return list(self.connections[game_id].keys())
        return []

    async def _get_alive_non_role_players(self, game_id: str, role: str) -> List[str]:
        """
        获取非指定角色的存活玩家列表

        Args:
            game_id: 游戏ID
            role: 要排除的角色

        Returns:
            非指定角色的存活玩家列表
        """
        alive_players = await self._get_alive_players(game_id)

        # 过滤掉指定角色的玩家
        non_role_players = []
        for player_id in alive_players:
            player_role = self.player_info.get(player_id, {}).get("role")
            if player_role != role:
                non_role_players.append(player_id)

        return non_role_players

    async def _is_player_alive(self, game_id: str, player_id: str) -> bool:
        """
        检查玩家是否存活

        Args:
            game_id: 游戏ID
            player_id: 玩家ID

        Returns:
            玩家是否存活
        """
        alive_players = await self._get_alive_players(game_id)
        return player_id in alive_players

    async def _cleanup_game_resources(self, game_id: str):
        """
        清理游戏资源

        Args:
            game_id: 游戏ID
        """
        try:
            # 停止所有计时器
            if game_id in self.active_timers:
                timer_ids = list(self.active_timers[game_id])
                for timer_id in timer_ids:
                    await self._stop_timer(timer_id)
                del self.active_timers[game_id]

            # 清理游戏状态
            if game_id in self.game_states:
                del self.game_states[game_id]

            # 清理连接
            if game_id in self.connections:
                del self.connections[game_id]

            # 清理锁
            if game_id in self.locks:
                del self.locks[game_id]

            logger.info(f"游戏资源清理完成: {game_id}")

        except Exception as e:
            logger.error(f"清理游戏资源失败: {e}")

    async def send_connection_status(self, player_id: str, status: str) -> Dict[str, Any]:
        """
        发送连接状态消息

        Args:
            player_id: 玩家ID
            status: 连接状态

        Returns:
            发送结果
        """
        if player_id in self.player_info:
            game_id = self.player_info[player_id].get("game_id")

            if game_id and game_id in self.connections:
                return await self.send_message(
                    game_id,
                    player_id,
                    MessageType.CONNECTION_STATUS.value,
                    {
                        "status": status,
                        "player_id": player_id,
                        "game_id": game_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

        return {"success": False, "error": "Player not found"}

    async def send_error_message(
            self,
            game_id: str,
            player_id: str,
            error_code: str,
            error_message: str,
            original_message_id: str = None
    ) -> Dict[str, Any]:
        """
        发送错误消息

        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            error_code: 错误码
            error_message: 错误消息
            original_message_id: 原始消息ID

        Returns:
            发送结果
        """
        return await self.send_message(
            game_id,
            player_id,
            MessageType.ERROR.value,
            {
                "error_code": error_code,
                "error_message": error_message,
                "original_message_id": original_message_id,
                "timestamp": datetime.now().isoformat()
            }
        )

    async def update_game_state(self, game_id: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新游戏状态

        Args:
            game_id: 游戏ID
            game_state: 游戏状态

        Returns:
            更新结果
        """
        self.game_states[game_id] = game_state

        # 触发游戏状态更新回调
        if self.callbacks["on_game_state_updated"]:
            await self.callbacks["on_game_state_updated"](game_id, game_state)

        return {"success": True, "game_id": game_id}

    # ==================== 统计和监控方法 ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        获取WebSocket管理器统计信息

        Returns:
            统计信息
        """
        total_connections = 0
        for game_connections in self.connections.values():
            total_connections += len(game_connections)

        return {
            "total_games": len(self.connections),
            "total_connections": total_connections,
            "active_timers": sum(len(timers) for timers in self.active_timers.values()),
            "message_queues": len(self.message_queues),
            "heartbeat_tasks": len(self.heartbeat_tasks),
            "connected_players": len([p for p in self.player_info.values() if p.get("connected")])
        }

    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """
        获取游戏统计信息

        Args:
            game_id: 游戏ID

        Returns:
            游戏统计信息
        """
        if game_id not in self.connections:
            return {"error": "Game not found"}

        return {
            "game_id": game_id,
            "connections": len(self.connections[game_id]),
            "active_timers": len(self.active_timers.get(game_id, set())),
            "connected_players": list(self.connections[game_id].keys()),
            "game_state": self.game_states.get(game_id, {}),
            "player_info": {
                player_id: info
                for player_id, info in self.player_info.items()
                if info.get("game_id") == game_id
            }
        }

    async def close_all_connections(self):
        """
        关闭所有连接
        """
        logger.info("开始关闭所有WebSocket连接")

        # 停止所有心跳检测
        for player_id in list(self.heartbeat_tasks.keys()):
            await self._stop_heartbeat_check(player_id)

        # 停止所有计时器
        for timer_id in list(self.timers.keys()):
            await self._stop_timer(timer_id)

        # 发送断开连接消息
        for game_id, connections in self.connections.items():
            for player_id in connections.keys():
                await self.send_connection_status(player_id, "disconnected")

        # 清理所有资源
        self.connections.clear()
        self.player_info.clear()
        self.message_queues.clear()
        self.timers.clear()
        self.active_timers.clear()
        self.game_states.clear()
        self.heartbeat_tasks.clear()
        self.heartbeat_intervals.clear()
        self.locks.clear()

        logger.info("所有WebSocket连接已关闭")