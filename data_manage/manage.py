# -*- coding: utf-8 -*-
import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

from interfaces import GameStorageInterface, EventType, GamePhase, StorageDirectoryType
from logging_config import game_logger

class GameStorageManager(GameStorageInterface):
    """游戏数据存储管理器"""
    
    def __init__(self, game_id: str, base_dir: str = "./game_data"):
        """
        初始化存储管理器
        
        Args:
            game_id: 游戏ID
            base_dir: 基础存储目录
        """
        self.game_id = game_id
        self.base_dir = base_dir
        self.game_dir = f"{base_dir}/game_{game_id}/"
        
        # 转换为Path对象以便使用pathlib语法
        self.game_dir_path = Path(self.game_dir)
        
        # 日志文件路径设置
        self.log_dir = self.game_dir_path / StorageDirectoryType.LOGS.value
        self.public_dir = self.game_dir_path / StorageDirectoryType.PUBLIC.value
        
        # 具体日志文件路径
        self.game_log_path = self.log_dir / "game_events.log"
        self.speech_log_path = self.log_dir / "public_speech.log"
        self.vote_log_path = self.log_dir / "vote_result.log"
        self.state_log_path = self.log_dir / "game_state.log"
        self.wolf_log_path = self.log_dir / "wolf_communication.log"
        
        # 公共事件日志
        self.public_log_path = self.public_dir / "events.jsonl"
        
        self._game_metadata = None
        
        # 初始化日志器
        self.logger = game_logger.get_game_logger(game_id, "storage")
        
        # 创建目录结构
        self._ensure_directories()
        
        # 初始化游戏元数据
        self._init_game_metadata()
    
    def _init_game_metadata(self):
        """初始化游戏元数据"""
        # 使用pathlib构建元数据文件路径
        metadata_file = self.game_dir_path / StorageDirectoryType.CONFIG.value / "metadata.json"
        if not os.path.exists(metadata_file):
            self._game_metadata = {
                "game_id": self.game_id,
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            self._save_metadata()
        else:
            with open(metadata_file, "r", encoding="utf-8") as f:
                self._game_metadata = json.load(f)
    
    def _ensure_directories(self):
        """确保所有必要的目录都存在"""
        # 使用pathlib构建所有必要的目录路径
        directories = [
            Path(self.base_dir),  # 基础目录
            self.game_dir_path,  # 游戏目录
            self.game_dir_path / StorageDirectoryType.PUBLIC.value,  # 公开数据目录
            self.game_dir_path / StorageDirectoryType.AGENTS.value,  # 智能体数据目录
            self.game_dir_path / StorageDirectoryType.LOGS.value,  # 日志数据目录
            self.game_dir_path / StorageDirectoryType.BACKUPS.value,  # 备份数据目录
            self.game_dir_path / StorageDirectoryType.CONFIG.value,  # 配置数据目录
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _save_metadata(self):
        """保存游戏元数据"""
        if self._game_metadata:
            try:
                # 使用pathlib构建元数据文件路径
                metadata_file = self.game_dir_path / StorageDirectoryType.CONFIG.value / "metadata.json"
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(self._game_metadata, f, ensure_ascii=False, indent=2)
                self.logger.debug(f"Game metadata saved for game {self.game_id}")
            except Exception as e:
                self.logger.error(f"Error saving game metadata for game {self.game_id}: {e}")
    
    def _update_last_modified(self):
        """更新最后修改时间"""
        if self._game_metadata:
            self._game_metadata["last_modified"] = datetime.now().isoformat()
            self._save_metadata()
    
    # ============ 公共数据存储 ============
    
    def save_public_event(self, event_data: Dict[str, Any]) -> bool:
        """
        保存公共事件（法官发言、遗言等）
        
        Args:
            event_data: 事件数据
            
        Returns:
            保存是否成功
        """
        try:
            # 确保事件有ID和时间戳
            event = event_data.copy()
            if "event_id" not in event:
                event["event_id"] = f"evt_{uuid.uuid4().hex[:8]}"
            if "timestamp" not in event:
                event["timestamp"] = datetime.now().isoformat()
            
            # 追加到公共事件日志
            # 使用JSON Lines格式（每行一个JSON对象）
            with open(self.public_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
            
            # 同时保存到game_events.log以支持WebSocket API网关
            self.save_game_event(event)
            
            # 更新最后修改时间
            self._update_last_modified()
            self.logger.debug(f"Saved public event: {event.get('event_type')}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving public event: {e}")
            return False
    
    def get_public_events(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取公共事件
        
        Args:
            event_type: 过滤事件类型
            limit: 返回的最大事件数
            
        Returns:
            事件列表
        """
        events = []
        
        if not os.path.exists(self.public_log_path):
            return []
        
        try:
            with open(self.public_log_path, 'r', encoding='utf-8') as f:
                # 读取最后limit行
                lines = f.readlines()
                # 从最新的事件开始遍历，直到达到limit
                count = 0
                for line in reversed(lines):
                    if count >= limit:
                        break
                    try:
                        event = json.loads(line.strip())
                        if event_type is None or event.get("event_type") == event_type:
                            events.append(event)
                            count += 1
                    except json.JSONDecodeError as e:
                        continue
        except Exception as e:
            self.logger.error(f"Error reading public events: {e}")
        
        return events
    
    def save_game_state_snapshot(self, state: Dict[str, Any]):
        """
        保存游戏状态快照
        
        Args:
            state: 游戏状态
        """
        snapshot = {
            "game_id": self.game_id,
            "snapshot_id": f"snap_{uuid.uuid4().hex[:12]}",
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "event_count": len(self.get_public_events(limit=1000))
        }
        
        # 使用pathlib构建快照文件路径
        snapshot_file = self.public_dir / "state_snapshots.jsonl"
        
        with open(snapshot_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
        
        # 更新最后修改时间
        self._update_last_modified()
    
    def save_game_event(self, event_data: Dict[str, Any]) -> bool:
        """
        保存游戏事件到game_events.log
        
        Args:
            event_data: 游戏事件数据
            
        Returns:
            保存是否成功
        """
        try:
            # 确保事件数据包含必要字段
            event = {
                "event_id": event_data.get("event_id", f"evt_{uuid.uuid4().hex[:8]}"),
                "event_type": event_data.get("event_type", EventType.UNKNOWN.value),
                "player_id": event_data.get("player_id"),
                "target_id": event_data.get("target_id"),
                "description": event_data.get("description", ""),
                "timestamp": event_data.get("timestamp", datetime.now().timestamp()),
                "metadata": event_data.get("metadata", {})
            }
            
            with open(self.game_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
            
            self._update_last_modified()
            return True
        except Exception as e:
            self.logger.error(f"Error saving game event: {e}")
            return False
    
    def save_speech(self, speech_data: Dict[str, Any]) -> bool:
        """
        保存发言记录到public_speech.log
        
        Args:
            speech_data: 发言数据
            
        Returns:
            保存是否成功
        """
        try:
            # 确保发言数据包含必要字段
            speech = {
                "speech_id": speech_data.get("speech_id", f"spch_{uuid.uuid4().hex[:8]}"),
                "player_id": speech_data.get("player_id", ""),
                "player_name": speech_data.get("player_name", "Unknown"),
                "text": speech_data.get("text", ""),
                "timestamp": speech_data.get("timestamp", datetime.now().timestamp()),
                "sentiment": speech_data.get("sentiment"),
                "confidence": speech_data.get("confidence", 1.0),
                "keywords": speech_data.get("keywords", [])
            }
            
            with open(self.speech_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(speech, ensure_ascii=False) + '\n')
            
            self._update_last_modified()
            return True
        except Exception as e:
            self.logger.error(f"Error saving speech: {e}")
            return False
    
    def save_vote(self, vote_data: Dict[str, Any]) -> bool:
        """
        保存投票结果到vote_result.log
        
        Args:
            vote_data: 投票数据
            
        Returns:
            保存是否成功
        """
        try:
            # 确保投票数据包含必要字段
            vote = {
                "round_id": vote_data.get("round_id", f"vote_{uuid.uuid4().hex[:8]}"),
                "day_number": vote_data.get("day_number", 1),
                "candidates": vote_data.get("candidates", []),
                "votes": vote_data.get("votes", {}),
                "result": vote_data.get("result"),
                "timestamp": vote_data.get("timestamp", datetime.now().timestamp())
            }
            
            with open(self.vote_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(vote, ensure_ascii=False) + '\n')
            
            self._update_last_modified()
            return True
        except Exception as e:
            self.logger.error(f"Error saving vote: {e}")
            return False
    
    def save_game_state(self, state_data: Dict[str, Any]) -> bool:
        """
        保存游戏状态到game_state.log
        
        Args:
            state_data: 游戏状态数据
            
        Returns:
            保存是否成功
        """
        try:
            # 确保状态数据包含必要字段
            phase_value = state_data.get("phase", "UNKNOWN")
            
            # 验证phase值是否有效
            if phase_value != "UNKNOWN":
                try:
                    # 转换为GamePhase枚举
                    phase = GamePhase(phase_value)
                    phase_value = phase.value
                except ValueError:
                    # 如果无效，使用默认值
                    phase_value = GamePhase.DAY.value
            
            game_state = {
                "game_id": state_data.get("game_id", self.game_id),
                "phase": phase_value,
                "day_number": state_data.get("day_number", 1),
                "alive_players": state_data.get("alive_players", []),
                "dead_players": state_data.get("dead_players", []),
                "timestamp": state_data.get("timestamp", datetime.now().timestamp()),
                "current_speaker": state_data.get("current_speaker"),
                "vote_results": state_data.get("vote_results"),
                "last_night_actions": state_data.get("last_night_actions")
            }
            
            with open(self.state_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(game_state, ensure_ascii=False) + '\n')
            
            self._update_last_modified()
            return True
        except Exception as e:
            self.logger.error(f"Error saving game state: {e}")
            return False
    
    def save_wolf_communication(self, communication_data: Dict[str, Any]) -> bool:
        """
        保存狼人通信到wolf_communication.log
        
        Args:
            communication_data: 狼人通信数据
            
        Returns:
            保存是否成功
        """
        try:
            # 确保通信数据包含必要字段
            communication = {
                "communication_id": communication_data.get("communication_id", f"wolf_{uuid.uuid4().hex[:8]}"),
                "player_id": communication_data.get("player_id", ""),
                "message": communication_data.get("message", ""),
                "timestamp": communication_data.get("timestamp", datetime.now().timestamp()),
                "metadata": communication_data.get("metadata", {})
            }
            
            with open(self.wolf_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(communication, ensure_ascii=False) + '\n')
            
            self._update_last_modified()
            return True
        except Exception as e:
            self.logger.error(f"Error saving wolf communication: {e}")
            return False

    # ============ Agent私有数据存储 ============
    
    def save_agent_memory(self, agent_id: str, memory_data: Dict[str, Any]) -> bool:
        """
        保存Agent记忆
        
        Args:
            agent_id: Agent ID
            memory_data: 记忆数据
            
        Returns:
            保存是否成功
        """
        try:
            # 使用pathlib构建agent目录路径
            agent_dir_path = self.game_dir_path / StorageDirectoryType.AGENTS.value / agent_id
            agent_dir = str(agent_dir_path)
            os.makedirs(agent_dir, exist_ok=True)
            
            # 使用pathlib构建记忆文件路径
            memory_file = agent_dir_path / "memory.json"
            
            # 确保记忆数据有基本结构
            if "entries" not in memory_data:
                memory_data["entries"] = []
            if "last_updated" not in memory_data:
                memory_data["last_updated"] = datetime.now().isoformat()
            else:
                memory_data["last_updated"] = datetime.now().isoformat()
            
            # 创建备份（保留最后5个版本）
            self._backup_file(memory_file, max_backups=5)
            
            # 保存记忆
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            # 更新最后修改时间
            self._update_last_modified()
            self.logger.debug(f"Saved agent memory for {agent_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving agent memory for {agent_id}: {e}")
            return False
    
    def load_agent_memory(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        加载Agent记忆
        
        Args:
            agent_id: Agent ID
            
        Returns:
            记忆数据，如果不存在则返回None
        """
        # 使用pathlib构建记忆文件路径
        memory_file = self.game_dir_path / StorageDirectoryType.AGENTS.value / agent_id / "memory.json"
        
        if not os.path.exists(memory_file):
            return None
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading agent memory: {e}")
            return None
    
    def append_agent_memory(self, agent_id: str, memory_item: Dict[str, Any]) -> bool:
        """
        追加智能体记忆
        
        Args:
            agent_id: 智能体ID
            memory_item: 记忆项
            
        Returns:
            追加是否成功
        """
        try:
            # 加载现有记忆
            existing_memory = self.load_agent_memory(agent_id)
            
            if existing_memory:
                # 确保entries字段存在
                if "entries" not in existing_memory:
                    existing_memory["entries"] = []
                
                # 确保记忆项有时间戳
                if "timestamp" not in memory_item:
                    memory_item["timestamp"] = datetime.now().isoformat()
                
                # 追加记忆项
                existing_memory["entries"].append(memory_item)
                existing_memory["last_updated"] = datetime.now().isoformat()
                
                # 保存更新后的记忆
                return self.save_agent_memory(agent_id, existing_memory)
            else:
                # 如果不存在，创建新记忆结构
                new_memory = {
                    "entries": [memory_item],
                    "last_updated": datetime.now().isoformat()
                }
                if "timestamp" not in memory_item:
                    memory_item["timestamp"] = datetime.now().isoformat()
                
                return self.save_agent_memory(agent_id, new_memory)
        except Exception as e:
            self.logger.error(f"Error appending agent memory for {agent_id}: {e}")
            return False
    
    def save_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """
        保存Agent性能指标
        
        Args:
            agent_id: Agent ID
            metrics: 指标数据
        """
        # 使用pathlib构建agent目录和指标文件路径
        agent_dir_path = self.game_dir_path / StorageDirectoryType.AGENTS.value / agent_id
        os.makedirs(agent_dir_path, exist_ok=True)
        
        # 确保指标有时间戳
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()
        if "agent_id" not in metrics:
            metrics["agent_id"] = agent_id
        
        metrics_file = agent_dir_path / "metrics.jsonl"
        
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        
        # 更新最后修改时间
        self._update_last_modified()
    
    def get_agent_metrics(self, agent_id: str, 
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None) -> List[Dict]:
        """
        获取Agent性能指标
        
        Args:
            agent_id: Agent ID
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            指标列表
        """
        # 使用pathlib构建指标文件路径
        metrics_file = self.game_dir_path / StorageDirectoryType.AGENTS.value / agent_id / "metrics.jsonl"
        
        if not os.path.exists(metrics_file):
            return []
        
        metrics = []
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        metric = json.loads(line.strip())
                        
                        # 时间过滤
                        timestamp = metric.get("timestamp")
                        if timestamp:
                            if start_time and timestamp < start_time:
                                continue
                            if end_time and timestamp > end_time:
                                continue
                        
                        metrics.append(metric)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.error(f"Error reading agent metrics: {e}")
        
        return metrics
    
    # ============ 日志存储 ============
    
    def save_agent_log(self, agent_id: str, log_data: Dict[str, Any]) -> bool:
        """
        保存Agent日志
        
        Args:
            agent_id: Agent ID
            log_data: 日志数据
            
        Returns:
            保存是否成功
        """
        try:
            # 使用pathlib构建日志文件路径
            log_file = self.log_dir / f"agent_{agent_id}.log"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                log_line = f"[{log_data.get('timestamp', datetime.now().isoformat())}] "
                log_line += f"[{log_data.get('level', 'INFO')}] "
                log_line += f"{log_data.get('message', '')}\n"
                f.write(log_line)
            
            # 更新最后修改时间
            self._update_last_modified()
            self.logger.debug(f"Saved agent log for {agent_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving agent log for {agent_id}: {e}")
            return False
    
    def get_agent_logs(self, agent_id: str, log_type: Optional[str] = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取智能体日志列表
        
        Args:
            agent_id: 智能体ID
            log_type: 日志类型
            limit: 日志数量限制
            
        Returns:
            日志列表
        """
        logs = []
        # 使用pathlib构建日志文件路径
        log_file = self.log_dir / f"agent_{agent_id}.log"
        
        if not os.path.exists(log_file):
            return logs
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析日志行
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 解析日志格式: [timestamp] [level] message
                try:
                    timestamp_end = line.index(']')
                    timestamp = line[1:timestamp_end].strip()
                    
                    level_start = line.index('[', timestamp_end) + 1
                    level_end = line.index(']', level_start)
                    level = line[level_start:level_end].strip()
                    
                    message = line[level_end + 1:].strip()
                    
                    log_entry = {
                        "timestamp": timestamp,
                        "level": level,
                        "message": message
                    }
                    
                    # 如果指定了日志类型（level），则过滤
                    if log_type and log_entry["level"] != log_type:
                        continue
                    
                    logs.append(log_entry)
                    
                except ValueError:
                    # 如果日志格式不符合预期，跳过该行
                    continue
            
            # 限制返回数量
            if limit > 0:
                logs = logs[-limit:]
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Error getting agent logs: {e}")
            return []
    
    # ============ 实用方法 ============
    
    def _backup_file(self, filepath: str, max_backups: int = 5):
        """
        备份文件
        
        Args:
            filepath: 文件路径
            max_backups: 最大备份数量
        """
        if not os.path.exists(filepath):
            return
        
        # 创建备份目录
        backup_dir = f"{self.game_dir}{StorageDirectoryType.BACKUPS.value}/"
        os.makedirs(backup_dir, exist_ok=True)
        
        # 生成备份文件名
        filename = os.path.basename(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 包含毫秒
        backup_file = f"{backup_dir}/{filename}.{timestamp}.bak"
        
        try:
            # 复制文件
            shutil.copy2(filepath, backup_file)
            
            # 清理旧备份
            self._cleanup_old_backups(backup_dir, filename, max_backups)
        except Exception as e:
            self.logger.error(f"Error backing up file {filepath}: {e}")
    
    def _cleanup_old_backups(self, backup_dir: str, 
                            filename_prefix: str, 
                            max_backups: int):
        """
        清理旧备份
        
        Args:
            backup_dir: 备份目录
            filename_prefix: 文件名前缀
            max_backups: 最大备份数量
        """
        if not os.path.exists(backup_dir):
            return
            
        try:
            # 获取所有匹配的备份文件
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.startswith(f"{filename_prefix}.") and file.endswith(".bak"):
                    filepath = os.path.join(backup_dir, file)
                    if os.path.isfile(filepath):
                        backup_files.append((filepath, os.path.getmtime(filepath)))
            
            # 按修改时间排序（旧的在前面）
            backup_files.sort(key=lambda x: x[1])
            
            # 删除多余的备份
            deleted_count = 0
            while len(backup_files) > max_backups:
                old_file, _ = backup_files.pop(0)
                try:
                    os.remove(old_file)
                    deleted_count += 1
                except Exception as delete_e:
                    self.logger.error(f"Error deleting old backup {old_file}: {delete_e}")
            
            if deleted_count > 0:
                pass  # 可以添加日志记录
                        
        except Exception as e:
            self.logger.error(f"Error cleaning up backups in {backup_dir}: {e}")
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        获取存储摘要
        
        Returns:
            存储统计信息
        """
        summary = {
            "game_id": self.game_id,
            "base_dir": self.base_dir,
            "game_dir": self.game_dir,
            "total_size": 0,
            "file_counts": {},
            "directory_counts": {},
            "created_at": self._game_metadata.get("created_at") if self._game_metadata else None,
            "last_modified": self._game_metadata.get("last_modified") if self._game_metadata else None,
            "metadata": self._game_metadata.copy() if self._game_metadata else None
        }
        
        try:
            # 计算总大小和文件/目录数量
            total_size = 0
            file_counts = {}
            directory_counts = {}
            
            for dirpath, dirnames, filenames in os.walk(self.game_dir):
                # 计算当前目录的文件数量
                dir_name = os.path.basename(dirpath)
                if dir_name in [d.value for d in StorageDirectoryType]:
                    file_counts[dir_name] = len(filenames)
                    directory_counts[dir_name] = len(dirnames)
                
                # 累加文件大小
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            summary["total_size"] = total_size
            summary["total_size_mb"] = total_size / (1024 * 1024)
            summary["file_counts"] = file_counts
            summary["directory_counts"] = directory_counts
            
            # 添加事件和快照数量
            summary["public_events_count"] = len(self.get_public_events(limit=10000))
            
        except Exception as e:
            summary["error"] = str(e)
        
        return summary
    
    def get_game_metadata(self) -> Optional[Dict[str, Any]]:
        """
        获取游戏元数据
        
        Returns:
            游戏元数据，如果不存在则返回None
        """
        return self._game_metadata.copy() if self._game_metadata else None
    
    def update_game_metadata(self, metadata_updates: Dict[str, Any]) -> bool:
        """
        更新游戏元数据
        
        Args:
            metadata_updates: 要更新的元数据字段
            
        Returns:
            更新是否成功
        """
        try:
            if self._game_metadata:
                self._game_metadata.update(metadata_updates)
                self._game_metadata["last_modified"] = datetime.now().isoformat()
                self._save_metadata()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating game metadata: {e}")
            return False