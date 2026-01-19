# -*- coding: utf-8 -*-
import os
import json
import shutil
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

from interfaces import GameStorageInterface
from logging_config import game_logger

class StorageDirectoryType(Enum):
    """存储目录类型枚举"""
    PUBLIC = "public"      # 公开数据
    AGENTS = "agents"      # 智能体数据
    LOGS = "logs"          # 日志数据
    BACKUPS = "backups"    # 备份数据
    CONFIG = "config"      # 配置数据

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
        self._game_metadata = None
        
        # 初始化日志器
        self.logger = game_logger.get_game_logger(game_id, "storage")
        
        # 创建目录结构
        self._ensure_directories()
        
        # 初始化游戏元数据
        self._init_game_metadata()
    
    def _init_game_metadata(self):
        """初始化游戏元数据"""
        metadata_file = f"{self.game_dir}config/metadata.json"
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
        directories = [
            self.base_dir,
            self.game_dir,
            f"{self.game_dir}{StorageDirectoryType.PUBLIC.value}/",
            f"{self.game_dir}{StorageDirectoryType.AGENTS.value}/",
            f"{self.game_dir}{StorageDirectoryType.LOGS.value}/",
            f"{self.game_dir}{StorageDirectoryType.BACKUPS.value}/",
            f"{self.game_dir}{StorageDirectoryType.CONFIG.value}/",
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _save_metadata(self):
        """保存游戏元数据"""
        if self._game_metadata:
            try:
                metadata_file = f"{self.game_dir}{StorageDirectoryType.CONFIG.value}/metadata.json"
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
            public_log_file = f"{self.game_dir}{StorageDirectoryType.PUBLIC.value}/events.jsonl"
            
            # 使用JSON Lines格式（每行一个JSON对象）
            with open(public_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
            
            # 更新最后修改时间
            self._update_last_modified()
            self.logger.debug(f"Saved public event: {event.get('event_type')}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving public event: {e}")
            return False
    
    def get_public_events(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[Dict]:
        """
        获取公共事件
        
        Args:
            event_type: 过滤事件类型
            limit: 返回的最大事件数
            
        Returns:
            事件列表
        """
        events = []
        public_log_file = f"{self.game_dir}{StorageDirectoryType.PUBLIC.value}/events.jsonl"
        
        if not os.path.exists(public_log_file):
            return []
        
        try:
            with open(public_log_file, 'r', encoding='utf-8') as f:
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
            print(f"Error reading public events: {e}")
        
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
        
        snapshot_file = f"{self.game_dir}{StorageDirectoryType.PUBLIC.value}/state_snapshots.jsonl"
        
        with open(snapshot_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
        
        # 更新最后修改时间
        self._update_last_modified()
    
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
            agent_dir = f"{self.game_dir}{StorageDirectoryType.AGENTS.value}/{agent_id}/"
            os.makedirs(agent_dir, exist_ok=True)
            
            memory_file = f"{agent_dir}/memory.json"
            
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
        memory_file = f"{self.game_dir}{StorageDirectoryType.AGENTS.value}/{agent_id}/memory.json"
        
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
        agent_dir = f"{self.game_dir}{StorageDirectoryType.AGENTS.value}/{agent_id}/"
        os.makedirs(agent_dir, exist_ok=True)
        
        # 确保指标有时间戳
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()
        if "agent_id" not in metrics:
            metrics["agent_id"] = agent_id
        
        metrics_file = f"{agent_dir}/metrics.jsonl"
        
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
        metrics_file = f"{self.game_dir}{StorageDirectoryType.AGENTS.value}/{agent_id}/metrics.jsonl"
        
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
            print(f"Error reading agent metrics: {e}")
        
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
            log_file = f"{self.game_dir}{StorageDirectoryType.LOGS.value}/agent_{agent_id}.log"
            
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
        log_file = f"{self.game_dir}{StorageDirectoryType.LOGS.value}/agent_{agent_id}.log"
        
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
            print(f"Error backing up file {filepath}: {e}")
    
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
                    print(f"Error deleting old backup {old_file}: {delete_e}")
            
            if deleted_count > 0:
                pass  # 可以添加日志记录
                        
        except Exception as e:
            print(f"Error cleaning up backups in {backup_dir}: {e}")
    
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
            print(f"Error updating game metadata: {e}")
            return False