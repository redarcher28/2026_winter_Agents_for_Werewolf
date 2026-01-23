# -*- coding: utf-8 -*-
import os
import json
import shutil
import zipfile
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

from judge_system.data_manage.manage import GameStorageManager
from judge_system.observer_interface import PlayerType, Role, PlayerStatus
from interfaces import DataStorageInterface, DataBackupType, ExportFormat
from judge_system.data_manage.logging_config import game_logger

class DataStorageService(DataStorageInterface):
    """数据存储服务"""
    
    def __init__(self, base_data_dir: str = "./game_data"):
        """
        初始化数据存储服务
        
        Args:
            base_data_dir: 基础数据存储目录
        """
        self.base_data_dir = base_data_dir
        self.backup_dir = f"{base_data_dir}/backups/"
        self._storage_managers = {}  # 存储管理器缓存
        self._storage_manager_last_used = {}  # 存储管理器最后使用时间
        self._max_cache_size = 50  # 最大缓存大小
        
        # 初始化日志器
        self.logger = game_logger.get_service_logger("data_storage")
        self.logger.info(f"DataStorageService initialized with base_dir: {base_data_dir}")
    
    def _cleanup_cache(self):
        """
        清理缓存，保持最大缓存大小
        """
        if len(self._storage_managers) <= self._max_cache_size:
            return
        
        # 按最后使用时间排序，删除最久未使用的实例
        sorted_keys = sorted(self._storage_manager_last_used.keys(), 
                           key=lambda k: self._storage_manager_last_used[k])
        
        # 删除超过最大大小的实例
        for key in sorted_keys[:-self._max_cache_size]:
            del self._storage_managers[key]
            del self._storage_manager_last_used[key]
        
        self.logger.info(f"Cleaned up storage manager cache, remaining: {len(self._storage_managers)}")
    
    def _get_storage_manager(self, game_id: str) -> GameStorageManager:
        """
        获取游戏存储管理器
        
        Args:
            game_id: 游戏ID
            
        Returns:
            GameStorageManager实例
        """
        # 清理缓存
        self._cleanup_cache()
        
        if game_id not in self._storage_managers:
            self._storage_managers[game_id] = GameStorageManager(game_id, self.base_data_dir)
        
        # 更新最后使用时间
        self._storage_manager_last_used[game_id] = datetime.now().timestamp()
        
        return self._storage_managers[game_id]
    
    def _get_directory_stats(self, directory_path: str) -> Dict[str, int]:
        """
        获取目录的文件统计信息
        
        Args:
            directory_path: 目录路径
            
        Returns:
            包含文件数量和总大小的字典
        """
        file_count = 0
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
                file_count += 1
        return {
            "file_count": file_count,
            "total_size": total_size
        }
    
    def _safe_path_join(self, base_dir: str, *path_components: str) -> str:
        """
        安全的路径拼接，防止路径遍历攻击
        
        Args:
            base_dir: 基础目录
            *path_components: 路径组件
            
        Returns:
            安全的绝对路径
            
        Raises:
            ValueError: 如果生成的路径不在基础目录内
        """
        # 确保基础目录是绝对路径
        base_dir = os.path.abspath(base_dir)
        
        # 拼接路径
        combined_path = os.path.abspath(os.path.join(base_dir, *path_components))
        
        # 检查路径是否在基础目录内
        if not combined_path.startswith(base_dir + os.path.sep):
            raise ValueError(f"Path traversal attempt detected: {combined_path}")
        
        return combined_path
    
    def create_backup(self, game_id: str, backup_type: DataBackupType = DataBackupType.FULL, 
                     backup_name: str = "") -> Tuple[bool, str, Dict[str, Any]]:
        """
        创建游戏数据备份
        
        Args:
            game_id: 游戏ID
            backup_type: 备份类型
            backup_name: 备份名称
            
        Returns:
            (成功标志, 备份ID, 备份信息)
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 生成备份ID
            backup_id = f"backup_{uuid.uuid4().hex[:16]}"
            
            # 生成备份名称
            if not backup_name:
                backup_name = f"{game_id}_{backup_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 创建备份目录
            try:
                backup_dir = self._safe_path_join(self.backup_dir, game_id, f"{backup_id}/")
                os.makedirs(backup_dir, exist_ok=True)
                
                # 执行备份
                backup_info = {
                    "backup_id": backup_id,
                    "game_id": game_id,
                    "backup_type": backup_type.value,
                    "backup_name": backup_name,
                    "timestamp": datetime.now().isoformat(),
                    "file_count": 0,
                    "total_size": 0
                }
                
                if backup_type == DataBackupType.FULL:
                    # 完整备份：复制整个游戏目录
                    src_dir = storage_manager.game_dir
                    dst_dir = self._safe_path_join(backup_dir, "game_data/")
                    shutil.copytree(src_dir, dst_dir)
                    
                    # 计算备份统计信息
                    stats = self._get_directory_stats(dst_dir)
                    backup_info["file_count"] = stats["file_count"]
                    backup_info["total_size"] = stats["total_size"]
                    backup_info["total_size_mb"] = stats["total_size"] / (1024 * 1024)
            except ValueError as e:
                self.logger.error(f"Security violation when creating backup for game {game_id}: {e}")
                return False, "", {"error": "Security violation", "details": str(e)}
                
            # 保存备份元数据
            metadata_file = f"{backup_dir}/metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(backup_info, f, ensure_ascii=False, indent=2)
            
            # 压缩备份（可选）
            zip_file = f"{self.backup_dir}{game_id}/{backup_id}.zip"
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(backup_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, backup_dir)
                        zf.write(file_path, arcname)
            
            # 清理临时目录
            shutil.rmtree(backup_dir)
            
            return True, backup_id, backup_info
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when creating backup for game {game_id}: {e}")
            return False, "", {"error": "File not found", "details": str(e)}
        except PermissionError as e:
            self.logger.error(f"Permission error when creating backup for game {game_id}: {e}")
            return False, "", {"error": "Permission denied", "details": str(e)}
        except zipfile.BadZipfile as e:
            self.logger.error(f"Zip file error when creating backup for game {game_id}: {e}")
            return False, "", {"error": "Zip file error", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error creating backup for game {game_id}: {e}")
            return False, "", {"error": "Backup failed", "details": str(e)}
    
    def restore_backup(self, game_id: str, backup_id: str, 
                      overwrite: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        恢复游戏数据备份
        
        Args:
            game_id: 游戏ID
            backup_id: 备份ID
            overwrite: 是否覆盖现有数据
            
        Returns:
            (成功标志, 恢复信息)
        """
        try:
            # 检查备份文件是否存在
            try:
                backup_zip = self._safe_path_join(self.backup_dir, game_id, f"{backup_id}.zip")
                
                if not os.path.exists(backup_zip):
                    return False, {"error": "Backup file not found"}
                
                # 检查目标游戏目录是否存在
                storage_manager = self._get_storage_manager(game_id)
                game_dir = storage_manager.game_dir
                
                if os.path.exists(game_dir) and not overwrite:
                    return False, {"error": "Game directory already exists, use overwrite=True to replace"}
                
                # 临时解压目录
                temp_dir = self._safe_path_join(self.backup_dir, game_id, f"temp_restore_{uuid.uuid4().hex[:8]}/")
                os.makedirs(temp_dir, exist_ok=True)
            except ValueError as e:
                self.logger.error(f"Security violation when restoring backup for game {game_id}: {e}")
                return False, {"error": "Security violation", "details": str(e)}
            
            # 解压备份文件
            with zipfile.ZipFile(backup_zip, 'r') as zf:
                zf.extractall(temp_dir)
            
            # 恢复数据
            restore_info = {
                "game_id": game_id,
                "backup_id": backup_id,
                "restore_time": datetime.now().isoformat(),
                "file_count": 0,
                "total_size": 0
            }
            
            # 获取备份元数据
            metadata_file = f"{temp_dir}/metadata.json"
            with open(metadata_file, "r", encoding="utf-8") as f:
                backup_info = json.load(f)
            
            restore_info["backup_info"] = backup_info
            
            # 复制恢复的数据到游戏目录
            if os.path.exists(game_dir):
                shutil.rmtree(game_dir)
            
            src_data_dir = f"{temp_dir}/game_data/"
            shutil.copytree(src_data_dir, game_dir)
            
            # 计算恢复统计信息
            stats = self._get_directory_stats(game_dir)
            restore_info["file_count"] = stats["file_count"]
            restore_info["total_size"] = stats["total_size"]
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
            # 更新存储管理器缓存
            if game_id in self._storage_managers:
                del self._storage_managers[game_id]
            
            return True, restore_info
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when restoring backup {backup_id} for game {game_id}: {e}")
            # 清理临时目录
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, {"error": "File not found", "details": str(e)}
        except PermissionError as e:
            self.logger.error(f"Permission error when restoring backup {backup_id} for game {game_id}: {e}")
            # 清理临时目录
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, {"error": "Permission denied", "details": str(e)}
        except zipfile.BadZipfile as e:
            self.logger.error(f"Invalid zip file when restoring backup {backup_id} for game {game_id}: {e}")
            # 清理临时目录
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, {"error": "Invalid backup file", "details": str(e)}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid metadata file when restoring backup {backup_id} for game {game_id}: {e}")
            # 清理临时目录
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, {"error": "Invalid backup metadata", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error restoring backup {backup_id} for game {game_id}: {e}")
            # 清理临时目录
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False, {"error": "Restore failed", "details": str(e)}
    
    def clean_data(self, game_id: str, days_to_keep: int = 30, 
                  clean_old_events: bool = True, 
                  clean_old_backups: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        清理游戏数据
        
        Args:
            game_id: 游戏ID
            days_to_keep: 保留天数
            clean_old_events: 是否清理旧事件
            clean_old_backups: 是否清理旧备份
            
        Returns:
            (成功标志, 清理信息)
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            clean_info = {
                "game_id": game_id,
                "timestamp": datetime.now().isoformat(),
                "days_to_keep": days_to_keep,
                "events_removed": 0,
                "backups_removed": 0,
                "total_size_freed": 0
            }
            
            # 清理旧事件
            if clean_old_events:
                events_file = f"{storage_manager.game_dir}logs/game_events.log"
                if os.path.exists(events_file):
                    # 读取所有事件
                    events = []
                    with open(events_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                event = json.loads(line.strip())
                                events.append(event)
                            except json.JSONDecodeError:
                                continue
                    
                    # 计算保留时间点
                    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
                    
                    # 过滤事件
                    kept_events = []
                    removed_events = 0
                    for event in events:
                        event_time = event.get("timestamp")
                        if event_time:
                            try:
                                # 处理不同格式的时间戳
                                if isinstance(event_time, str):
                                    event_time = datetime.fromisoformat(event_time).timestamp()
                                if event_time >= cutoff_time:
                                    kept_events.append(event)
                                else:
                                    removed_events += 1
                            except ValueError:
                                # 如果时间戳格式无效，保留事件
                                kept_events.append(event)
                        else:
                            # 如果没有时间戳，保留事件
                            kept_events.append(event)
                    
                    # 保存过滤后的事件
                    if removed_events > 0:
                        with open(events_file, "w", encoding="utf-8") as f:
                            for event in kept_events:
                                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                    
                    clean_info["events_removed"] = removed_events
            
            # 清理旧备份
            if clean_old_backups:
                backup_dir = f"{self.backup_dir}{game_id}/"
                if os.path.exists(backup_dir):
                    # 计算保留时间点
                    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
                    
                    removed_backups = 0
                    total_size_freed = 0
                    
                    for backup_id in os.listdir(backup_dir):
                        backup_zip = f"{backup_dir}/{backup_id}"
                        if backup_zip.endswith(".zip"):
                            backup_time = os.path.getmtime(backup_zip)
                            if backup_time < cutoff_time:
                                total_size_freed += os.path.getsize(backup_zip)
                                os.remove(backup_zip)
                                removed_backups += 1
                    
                    clean_info["backups_removed"] = removed_backups
                    clean_info["total_size_freed"] = total_size_freed
            
            return True, clean_info
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when cleaning data for game {game_id}: {e}")
            return False, {"error": "File not found", "details": str(e)}
        except PermissionError as e:
            self.logger.error(f"Permission error when cleaning data for game {game_id}: {e}")
            return False, {"error": "Permission denied", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error cleaning data for game {game_id}: {e}")
            return False, {"error": "Clean data failed", "details": str(e)}
    
    def export_data(self, game_id: str, export_format: ExportFormat = ExportFormat.ZIP, 
                   export_path: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        导出游戏数据
        
        Args:
            game_id: 游戏ID
            export_format: 导出格式
            export_path: 导出路径
            
        Returns:
            (成功标志, 导出文件路径, 导出信息)
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 生成导出文件路径
            if not export_path:
                export_filename = f"game_{game_id}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.value}"
                export_path = f"{self.base_data_dir}/exports/{export_filename}"
            
            # 创建导出目录
            export_dir = os.path.dirname(export_path)
            os.makedirs(export_dir, exist_ok=True)
            
            export_info = {
                "game_id": game_id,
                "export_format": export_format.value,
                "export_path": export_path,
                "timestamp": datetime.now().isoformat(),
                "file_count": 0,
                "total_size": 0
            }
            
            if export_format == "zip":
                # 导出为ZIP文件
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # 添加游戏数据
                    src_dir = storage_manager.game_dir
                    for dirpath, dirnames, filenames in os.walk(src_dir):
                        for filename in filenames:
                            file_path = os.path.join(dirpath, filename)
                            arcname = os.path.relpath(file_path, self.base_data_dir)
                            zf.write(file_path, arcname)
                    
                # 计算导出统计信息
                stats = self._get_directory_stats(storage_manager.game_dir)
                export_info["file_count"] = stats["file_count"]
                export_info["total_size"] = stats["total_size"]
            
            # 添加MB格式的大小
            export_info["total_size_mb"] = export_info["total_size"] / (1024 * 1024)
            
            return True, export_path, export_info
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when exporting data for game {game_id}: {e}")
            return False, "", {"error": "File not found", "details": str(e)}
        except PermissionError as e:
            self.logger.error(f"Permission error when exporting data for game {game_id}: {e}")
            return False, "", {"error": "Permission denied", "details": str(e)}
        except zipfile.BadZipfile as e:
            self.logger.error(f"Zip file error when exporting data for game {game_id}: {e}")
            return False, "", {"error": "Zip file error", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error exporting data for game {game_id}: {e}")
            return False, "", {"error": "Export data failed", "details": str(e)}
    
    def import_data(self, import_path: str, game_id: Optional[str] = None, 
                   overwrite: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        """
        导入游戏数据
        
        Args:
            import_path: 导入文件路径
            game_id: 游戏ID
            overwrite: 是否覆盖现有数据
            
        Returns:
            (成功标志, 游戏ID, 导入信息)
        """
        try:
            # 检查导入文件是否存在
            if not os.path.exists(import_path):
                return False, "", {"error": "Import file not found"}
            
            # 生成游戏ID
            if not game_id:
                game_id = f"imported_{uuid.uuid4().hex[:12]}"
            
            # 检查游戏是否已存在
            storage_manager = self._get_storage_manager(game_id)
            game_dir = storage_manager.game_dir
            
            if os.path.exists(game_dir) and not overwrite:
                return False, "", {"error": "Game already exists, use overwrite=True to replace"}
            
            import_info = {
                "game_id": game_id,
                "import_path": import_path,
                "import_time": datetime.now().isoformat(),
                "file_count": 0,
                "total_size": 0
            }
            
            if import_path.endswith(".zip"):
                # 从ZIP文件导入
                try:
                    # 安全生成临时目录路径
                    temp_dir = self._safe_path_join(self.base_data_dir, f"temp_import_{uuid.uuid4().hex[:8]}/")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    with zipfile.ZipFile(import_path, 'r') as zf:
                        zf.extractall(temp_dir)
                    
                    # 查找游戏数据目录
                    game_data_dir = None
                    for root, dirs, files in os.walk(temp_dir):
                        for dir in dirs:
                            if dir.startswith("game_"):
                                game_data_dir = os.path.join(root, dir)
                                break
                        if game_data_dir:
                            break
                    
                    if not game_data_dir:
                        shutil.rmtree(temp_dir)
                        return False, "", {"error": "Game data not found in import file"}
                    
                    # 复制导入的数据到游戏目录
                    if os.path.exists(game_dir):
                        shutil.rmtree(game_dir)
                    
                    shutil.copytree(game_data_dir, game_dir)
                except ValueError as e:
                    self.logger.error(f"Security violation when importing data: {e}")
                    if 'temp_dir' in locals() and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    return False, "", {"error": "Security violation", "details": str(e)}
                
                # 计算导入统计信息
                stats = self._get_directory_stats(game_dir)
                import_info["file_count"] = stats["file_count"]
                import_info["total_size"] = stats["total_size"]
                
                # 清理临时目录
                shutil.rmtree(temp_dir)
            
            return True, game_id, import_info
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when importing data for game {game_id}: {e}")
            return False, "", {"error": "File not found", "details": str(e)}
        except PermissionError as e:
            self.logger.error(f"Permission error when importing data for game {game_id}: {e}")
            return False, "", {"error": "Permission denied", "details": str(e)}
        except zipfile.BadZipfile as e:
            self.logger.error(f"Invalid zip file when importing data: {e}")
            return False, "", {"error": "Invalid import file", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error importing data: {e}")
            return False, "", {"error": "Import data failed", "details": str(e)}
    
    def get_data_statistics(self, game_id: str) -> Dict[str, Any]:
        """
        获取游戏数据统计信息
        
        Args:
            game_id: 游戏ID
            
        Returns:
            数据统计信息
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 获取存储摘要
            storage_summary = storage_manager.get_storage_summary()
            
            # 获取备份统计
            backup_dir = f"{self.backup_dir}{game_id}/"
            backup_count = 0
            total_backup_size = 0
            
            if os.path.exists(backup_dir):
                for file in os.listdir(backup_dir):
                    if file.endswith(".zip"):
                        file_path = os.path.join(backup_dir, file)
                        total_backup_size += os.path.getsize(file_path)
                        backup_count += 1
            
            statistics = {
                "game_id": game_id,
                "timestamp": datetime.now().isoformat(),
                "storage_summary": storage_summary,
                "backup_statistics": {
                    "backup_count": backup_count,
                    "total_backup_size": total_backup_size,
                    "total_backup_size_mb": total_backup_size / (1024 * 1024)
                }
            }
            
            return statistics
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when getting statistics for game {game_id}: {e}")
            return {"error": "File not found", "details": str(e)}
        except PermissionError as e:
            self.logger.error(f"Permission error when getting statistics for game {game_id}: {e}")
            return {"error": "Permission denied", "details": str(e)}
        except Exception as e:
            self.logger.error(f"Unexpected error getting statistics for game {game_id}: {e}")
            return {"error": "Get statistics failed", "details": str(e)}
    
    def get_all_games(self) -> List[Dict[str, Any]]:
        """
        获取所有游戏数据信息
        
        Returns:
            游戏数据信息列表
        """
        try:
            games = []
            
            if not os.path.exists(self.base_data_dir):
                return games
            
            # 遍历所有游戏目录
            for dir_name in os.listdir(self.base_data_dir):
                if dir_name.startswith("game_"):
                    game_id = dir_name[5:]  # 提取游戏ID
                    
                    try:
                        # 获取游戏数据统计
                        statistics = self.get_data_statistics(game_id)
                        games.append(statistics)
                    except Exception as e:
                        self.logger.error(f"Error getting statistics for game {game_id}: {e}")
                        continue
            
            # 按创建时间排序
            games.sort(key=lambda x: x["storage_summary"]["created_at"], reverse=True)
            
            return games
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when getting all games: {e}")
            return []
        except PermissionError as e:
            self.logger.error(f"Permission error when getting all games: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error getting all games: {e}")
            return []
    
    def delete_game_data(self, game_id: str, delete_backups: bool = False) -> bool:
        """
        删除游戏数据
        
        Args:
            game_id: 游戏ID
            delete_backups: 是否删除备份
            
        Returns:
            操作是否成功
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            game_dir = storage_manager.game_dir
            
            # 删除游戏目录
            if os.path.exists(game_dir):
                shutil.rmtree(game_dir)
            
            # 删除备份
            if delete_backups:
                backup_dir = f"{self.backup_dir}{game_id}/"
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
            
            # 从缓存中移除
            if game_id in self._storage_managers:
                del self._storage_managers[game_id]
            
            return True
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found when deleting data for game {game_id}: {e}")
            return False
        except PermissionError as e:
            self.logger.error(f"Permission error when deleting data for game {game_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error deleting data for game {game_id}: {e}")
            return False
    
    # ============ 玩家状态管理接口 ============
    
    def save_player_status(self, game_id: str, player_status: Dict[str, Any]) -> bool:
        """
        保存单个玩家的状态
        
        Args:
            game_id: 游戏ID
            player_status: 玩家状态数据
            
        Returns:
            保存是否成功
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 保存玩家状态
            result = storage_manager.save_player_status(player_status)
            self.logger.debug(f"Saved player status for game {game_id}: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error saving player status for game {game_id}: {e}")
            return False
    
    def save_all_player_statuses(self, game_id: str, player_statuses: Dict[str, Dict[str, Any]]) -> bool:
        """
        保存所有玩家的状态
        
        Args:
            game_id: 游戏ID
            player_statuses: 玩家状态数据字典，键为玩家ID，值为玩家状态
            
        Returns:
            保存是否成功
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 保存所有玩家状态
            result = storage_manager.save_all_player_statuses(player_statuses)
            self.logger.debug(f"Saved all player statuses for game {game_id}: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error saving all player statuses for game {game_id}: {e}")
            return False
    
    def load_player_status(self, game_id: str, player_id: str) -> Optional[Dict[str, Any]]:
        """
        加载玩家状态
        
        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            
        Returns:
            玩家状态数据，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 加载玩家状态
            result = storage_manager.load_player_status(player_id)
            self.logger.debug(f"Loaded player status for game {game_id}, player {player_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error loading player status for game {game_id}, player {player_id}: {e}")
            return None
    
    def load_all_player_statuses(self, game_id: str) -> Dict[str, Dict[str, Any]]:
        """
        加载所有玩家的状态
        
        Args:
            game_id: 游戏ID
            
        Returns:
            玩家状态数据字典，键为玩家ID，值为玩家状态
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 加载所有玩家状态
            result = storage_manager.load_all_player_statuses()
            self.logger.debug(f"Loaded status for {len(result)} players in game {game_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error loading all player statuses for game {game_id}: {e}")
            return {}
    
    def save_witch_action(self, game_id: str, witch_id: str, action: Dict[str, Any]) -> bool:
        """
        保存女巫行动（私有数据）
        
        Args:
            game_id: 游戏ID
            witch_id: 女巫的Agent ID
            action: 女巫行动数据，包含是否使用解药、是否使用毒药、选择的目标等
            
        Returns:
            保存是否成功
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 保存女巫行动
            result = storage_manager.save_witch_action(witch_id, action)
            self.logger.debug(f"Saved witch action for game {game_id}, witch {witch_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error saving witch action for game {game_id}, witch {witch_id}: {e}")
            return False
    
    def get_witch_action(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        获取女巫行动（私有数据）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            女巫行动数据，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 获取女巫行动
            result = storage_manager.get_witch_action()
            self.logger.debug(f"Got witch action for game {game_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting witch action for game {game_id}: {e}")
            return None
    
    def save_seer_action(self, game_id: str, seer_id: str, action: Dict[str, Any]) -> bool:
        """
        保存预言家行动（私有数据）
        
        Args:
            game_id: 游戏ID
            seer_id: 预言家的Agent ID
            action: 预言家行动数据，包含验人选择、验人结果等
            
        Returns:
            保存是否成功
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 保存预言家行动
            result = storage_manager.save_seer_action(seer_id, action)
            self.logger.debug(f"Saved seer action for game {game_id}, seer {seer_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error saving seer action for game {game_id}, seer {seer_id}: {e}")
            return False
    
    def get_seer_action(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        获取预言家行动（私有数据）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            预言家行动数据，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 获取预言家行动
            result = storage_manager.get_seer_action()
            self.logger.debug(f"Got seer action for game {game_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting seer action for game {game_id}: {e}")
            return None
    
    def save_werewolf_action(self, game_id: str, werewolf_id: str, action: Dict[str, Any]) -> bool:
        """
        保存狼人行动（私有数据）
        
        Args:
            game_id: 游戏ID
            werewolf_id: 狼人的Agent ID
            action: 狼人行动数据，包含刀人选择等
            
        Returns:
            保存是否成功
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 保存狼人行动
            result = storage_manager.save_werewolf_action(werewolf_id, action)
            self.logger.debug(f"Saved werewolf action for game {game_id}, werewolf {werewolf_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error saving werewolf action for game {game_id}, werewolf {werewolf_id}: {e}")
            return False
    
    def get_werewolf_action(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        获取狼人行动（私有数据）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            狼人行动数据，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 获取狼人行动
            result = storage_manager.get_werewolf_action()
            self.logger.debug(f"Got werewolf action for game {game_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting werewolf action for game {game_id}: {e}")
            return None
    
    # ============ 业务逻辑层数据需求接口 ============
    
    def get_basic_env_data(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        获取基础环境数据（从 game_state.log 最后一行）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            基础环境数据，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 读取 game_state.log 的最后一行
            state_log_path = storage_manager.state_log_path
            
            if not os.path.exists(state_log_path):
                return None
            
            # 读取所有行并获取最后一行
            with open(state_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    return None
                
                last_line = lines[-1].strip()
                if not last_line:
                    return None
                
                # 解析最后一行
                game_state = json.loads(last_line)
                
                # 提取基础环境数据
                basic_env = {
                    "phase": game_state.get("phase", "UNKNOWN"),
                    "day_number": game_state.get("day_number", 1),
                    "alive_players": game_state.get("alive_players", [])
                }
                
                self.logger.debug(f"Got basic env data for game {game_id}: {basic_env}")
                return basic_env
        except Exception as e:
            self.logger.error(f"Error getting basic env data for game {game_id}: {e}")
            return None
    
    def get_role_permissions(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        获取角色权限快照（从 private/roles/*.json）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            角色权限快照，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            role_permissions = {
                "werewolf": {
                    "team_members": [],
                    "has_acted": False
                },
                "witch": {
                    "heal_used": False,
                    "poison_used": False
                },
                "seer": {
                    "has_inspected": False
                }
            }
            
            # 读取狼人数据
            werewolf_data = storage_manager.get_role_specific_data("werewolf")
            if werewolf_data:
                role_permissions["werewolf"]["team_members"] = werewolf_data.get("team_members", [])
                # 检查是否已行动
                kill_targets = werewolf_data.get("kill_targets", [])
                # 简单判断：如果有击杀目标记录，则认为已行动
                role_permissions["werewolf"]["has_acted"] = len(kill_targets) > 0
            
            # 读取女巫数据
            witch_data = storage_manager.get_role_specific_data("witch")
            if witch_data:
                role_permissions["witch"]["heal_used"] = witch_data.get("heal_used", False)
                role_permissions["witch"]["poison_used"] = witch_data.get("poison_used", False)
            
            # 读取预言家数据
            seer_data = storage_manager.get_role_specific_data("seer")
            if seer_data:
                inspections = seer_data.get("inspections", [])
                # 简单判断：如果有验人记录，则认为已行动
                role_permissions["seer"]["has_inspected"] = len(inspections) > 0
            
            self.logger.debug(f"Got role permissions for game {game_id}")
            return role_permissions
        except Exception as e:
            self.logger.error(f"Error getting role permissions for game {game_id}: {e}")
            return None
    
    def get_role_identities(self, game_id: str) -> Optional[Dict[str, str]]:
        """
        获取角色真实身份表（从 private/roles/ 下的所有文件）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            角色真实身份表，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            role_identities = {}
            
            # 构建角色数据目录路径
            roles_dir = os.path.join(storage_manager.game_dir, "private", "roles")
            
            if not os.path.exists(roles_dir):
                return role_identities
            
            # 读取所有角色文件
            for filename in os.listdir(roles_dir):
                if filename.endswith(".json") and filename != "wolf_communication.log":
                    role = filename[:-5]  # 移除 .json 后缀
                    role_data = storage_manager.get_role_specific_data(role)
                    
                    if role_data:
                        # 处理不同角色的数据结构
                        if role == "werewolf":
                            # 从狼队成员中提取身份
                            team_members = role_data.get("team_members", [])
                            for member in team_members:
                                if isinstance(member, dict):
                                    player_id = member.get("player_id")
                                    if player_id:
                                        role_identities[player_id] = "WEREWOLF"
                        elif role == "witch":
                            # 女巫数据中可能直接包含女巫ID
                            witch_id = role_data.get("witch_id") or role_data.get("player_id")
                            if witch_id:
                                role_identities[witch_id] = "WITCH"
                        elif role == "seer":
                            # 预言家数据中可能直接包含预言家ID
                            seer_id = role_data.get("seer_id") or role_data.get("player_id")
                            if seer_id:
                                role_identities[seer_id] = "SEER"
            
            self.logger.debug(f"Got role identities for game {game_id}: {role_identities}")
            return role_identities
        except Exception as e:
            self.logger.error(f"Error getting role identities for game {game_id}: {e}")
            return None
    
    def get_vote_data(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        获取决策依据/票池（从 vote_result.log 最后一行）
        
        Args:
            game_id: 游戏ID
            
        Returns:
            决策依据/票池，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self._get_storage_manager(game_id)
            
            # 读取 vote_result.log 的最后一行
            vote_log_path = storage_manager.vote_log_path
            
            if not os.path.exists(vote_log_path):
                return {
                    "votes": {},
                    "result": None
                }
            
            # 读取所有行并获取最后一行
            with open(vote_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    return {
                        "votes": {},
                        "result": None
                    }
                
                last_line = lines[-1].strip()
                if not last_line:
                    return {
                        "votes": {},
                        "result": None
                    }
                
                # 解析最后一行
                vote_data = json.loads(last_line)
                
                # 提取投票数据
                vote_info = {
                    "votes": vote_data.get("votes", {}),
                    "result": vote_data.get("result", None)
                }
                
                self.logger.debug(f"Got vote data for game {game_id}: {vote_info}")
                return vote_info
        except Exception as e:
            self.logger.error(f"Error getting vote data for game {game_id}: {e}")
            return {
                "votes": {},
                "result": None
            }
    
    def get_business_data(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        获取业务逻辑层所需的完整数据
        
        Args:
            game_id: 游戏ID
            
        Returns:
            业务逻辑层所需的完整数据，如果不存在则返回None
        """
        try:
            # 获取所有业务数据
            basic_env = self.get_basic_env_data(game_id)
            role_permissions = self.get_role_permissions(game_id)
            role_identities = self.get_role_identities(game_id)
            vote_data = self.get_vote_data(game_id)
            
            # 构建完整的业务数据
            business_data = {
                "basic_env": basic_env or {
                    "phase": "UNKNOWN",
                    "day_number": 1,
                    "alive_players": []
                },
                "role_permissions": role_permissions or {
                    "werewolf": {
                        "team_members": [],
                        "has_acted": False
                    },
                    "witch": {
                        "heal_used": False,
                        "poison_used": False
                    },
                    "seer": {
                        "has_inspected": False
                    }
                },
                "role_identities": role_identities or {},
                "vote_data": vote_data or {
                    "votes": {},
                    "result": None
                }
            }
            
            self.logger.debug(f"Got business data for game {game_id}")
            return business_data
        except Exception as e:
            self.logger.error(f"Error getting business data for game {game_id}: {e}")
            return None