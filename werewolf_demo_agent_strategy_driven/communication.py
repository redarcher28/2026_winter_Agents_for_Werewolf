# communication.py
import asyncio
import aiofiles
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from config import Role, AgentConfig
from core_agent import BaseWerewolfAgent

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