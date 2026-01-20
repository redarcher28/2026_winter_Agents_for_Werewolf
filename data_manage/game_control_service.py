# -*- coding: utf-8 -*-
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from judge_system.manage import GameStorageManager
from logging_config import game_logger
from interfaces import EventType

class GameControlService:
    """游戏控制服务（仅保留数据存储管理相关逻辑）"""
    
    def __init__(self, base_data_dir: str = "./game_data"):
        """
        初始化游戏数据存储服务
        
        Args:
            base_data_dir: 基础数据存储目录
        """
        self.base_data_dir = base_data_dir
        self._active_games = {}  # 活跃游戏存储管理器缓存
        
        # 初始化日志器
        self.logger = game_logger.get_service_logger("game_control")
        self.logger.info(f"GameControlService initialized with base_dir: {base_data_dir}")
    
    def _get_storage_manager(self, game_id: str) -> GameStorageManager:
        """
        获取游戏存储管理器（核心存储工具类）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            GameStorageManager实例
        """
        if game_id not in self._active_games:
            self._active_games[game_id] = GameStorageManager(game_id, self.base_data_dir)
        return self._active_games[game_id]
    
    async def create_new_game(self, game_config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        创建新游戏（仅保留配置存储逻辑）
        
        Args:
            game_config: 游戏配置
            
        Returns:
            (成功标志, 游戏ID, 结果数据)
        """
        try:
            # 生成游戏ID
            game_id = f"{uuid.uuid4().hex[:12]}"
            
            # 创建存储管理器
            storage_manager = GameStorageManager(game_id, self.base_data_dir)
            
            # 初始化游戏配置
            game_config["game_id"] = game_id
            game_config["created_at"] = datetime.now().isoformat()
            
            # 保存游戏配置（核心存储操作）
            config_file = f"{storage_manager.game_dir}config/game_config.json"
            os.makedirs(os.path.dirname(config_file), exist_ok=True)  # 确保目录存在
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(game_config, f, ensure_ascii=False, indent=2)
            
            # 保存游戏创建事件（数据持久化）
            storage_manager.save_public_event({
                "event_type": EventType.GAME_CREATED.value,
                "game_config": game_config,
                "timestamp": datetime.now().isoformat()
            })
            
            # 更新游戏元数据
            storage_manager.update_game_metadata({
                "game_config": game_config
            })
            
            # 缓存存储管理器
            self._active_games[game_id] = storage_manager
            
            return True, game_id, {
                "game_id": game_id,
                "created_at": game_config["created_at"],
                "game_config": game_config
            }
            
        except Exception as e:
            return False, "", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def save_game_snapshot(self, game_id: str, snapshot_name: str = "auto") -> Tuple[bool, str]:
        """
        保存游戏数据快照（核心数据备份逻辑）
        
        Args:
            game_id: 游戏ID
            snapshot_name: 快照名称
            
        Returns:
            (成功标志, 快照ID)
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 生成快照ID
            snapshot_id = f"snap_{uuid.uuid4().hex[:16]}"
            
            # 读取当前游戏数据（配置+事件）
            current_events = storage_manager.get_public_events(limit=10000)
            config_file = f"{storage_manager.game_dir}config/game_config.json"
            game_config = {}
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    game_config = json.load(f)
            
            # 构建快照元数据
            snapshot_metadata = {
                "snapshot_id": snapshot_id,
                "game_id": game_id,
                "snapshot_name": snapshot_name,
                "timestamp": datetime.now().isoformat(),
                "event_count": len(current_events),
                "game_config": game_config
            }
            
            # 保存快照（核心存储操作）
            snapshot_dir = f"{storage_manager.game_dir}backups/snapshots/{snapshot_id}/"
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # 存储快照元数据
            metadata_file = f"{snapshot_dir}metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_metadata, f, ensure_ascii=False, indent=2)
            
            # 存储事件数据（JSON Lines格式）
            events_file = f"{snapshot_dir}events.jsonl"
            with open(events_file, "w", encoding="utf-8") as f:
                for event in current_events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            
            # 备份游戏配置
            config_backup_file = f"{snapshot_dir}game_config.json"
            with open(config_backup_file, "w", encoding="utf-8") as f:
                json.dump(game_config, f, ensure_ascii=False, indent=2)
            
            # 记录快照备份事件
            storage_manager.save_public_event({
                "event_type": EventType.SNAPSHOT_SAVED.value,
                "snapshot_id": snapshot_id,
                "snapshot_name": snapshot_name,
                "timestamp": datetime.now().isoformat()
            })
            
            return True, snapshot_id
            
        except Exception as e:
            self.logger.error(f"Error saving game snapshot for game {game_id}: {e}")
            return False, ""
    
    async def load_game_snapshot(self, game_id: str, snapshot_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        加载游戏数据快照（核心数据恢复逻辑）
        
        Args:
            game_id: 游戏ID
            snapshot_id: 快照ID
            
        Returns:
            (成功标志, 快照数据)
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 检查快照目录是否存在
            snapshot_dir = f"{storage_manager.game_dir}backups/snapshots/{snapshot_id}/"
            if not os.path.exists(snapshot_dir):
                return False, {"error": "Snapshot not found"}
            
            # 读取快照元数据
            metadata_file = f"{snapshot_dir}metadata.json"
            with open(metadata_file, "r", encoding="utf-8") as f:
                snapshot_metadata = json.load(f)
            
            # 读取快照事件数据
            events_file = f"{snapshot_dir}events.jsonl"
            events = []
            with open(events_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
            
            # 读取快照中的游戏配置
            config_file = f"{snapshot_dir}game_config.json"
            with open(config_file, "r", encoding="utf-8") as f:
                game_config = json.load(f)
            
            # 恢复游戏配置（覆盖当前配置）
            current_config_file = f"{storage_manager.game_dir}config/game_config.json"
            with open(current_config_file, "w", encoding="utf-8") as f:
                json.dump(game_config, f, ensure_ascii=False, indent=2)
            
            # 恢复事件数据（覆盖当前事件）
            current_events_file = f"{storage_manager.game_dir}public/events.jsonl"
            os.makedirs(os.path.dirname(current_events_file), exist_ok=True)
            with open(current_events_file, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            
            # 记录快照恢复事件
            storage_manager.save_public_event({
                "event_type": "snapshot_loaded",
                "snapshot_id": snapshot_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return True, {
                "snapshot_metadata": snapshot_metadata,
                "event_count": len(events),
                "game_config": game_config
            }
            
        except Exception as e:
            self.logger.error(f"Error loading game snapshot {snapshot_id} for game {game_id}: {e}")
            return False, {"error": str(e)}
    
    async def get_game_status(self, game_id: str) -> Dict[str, Any]:
        """
        获取游戏数据存储状态（核心数据读取逻辑）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            游戏数据存储状态
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 读取游戏配置文件
            config_file = f"{storage_manager.game_dir}config/game_config.json"
            game_config = {}
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    game_config = json.load(f)
            
            # 获取存储摘要（大小、事件数等）
            storage_summary = storage_manager.get_storage_summary()
            
            return {
                "game_id": game_id,
                "event_count": storage_summary.get("public_events_count", 0),
                "total_size": storage_summary.get("total_size", 0),
                "created_at": game_config.get("created_at"),
                "last_modified": storage_summary.get("last_modified"),
                "game_config": game_config
            }
            
        except Exception as e:
            self.logger.error(f"Error getting game storage status: {e}")
            return {
                "game_id": game_id,
                "error": str(e)
            }
    
    async def get_all_games(self) -> List[Dict[str, Any]]:
        """
        获取所有游戏的存储信息（批量数据读取）
        
        Returns:
            游戏存储信息列表
        """
        games = []
        
        try:
            # 遍历数据目录下所有游戏文件夹
            if not os.path.exists(self.base_data_dir):
                return games
            
            for dir_name in os.listdir(self.base_data_dir):
                if dir_name.startswith("game_"):
                    game_id = dir_name[5:]  # 提取游戏ID
                    try:
                        # 获取单游戏存储状态
                        game_status = await self.get_game_status(game_id)
                        games.append(game_status)
                    except Exception as e:
                        self.logger.error(f"Error getting storage status for game {game_id}: {e}")
                        continue
            
            # 按创建时间排序
            games.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting all game storage info: {e}")
        
        return games