# -*- coding: utf-8 -*-
import os
import json
import shutil
import zipfile
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

from judge_system.manage import GameStorageManager
from interfaces import DataStorageInterface, DataBackupType, ExportFormat
from logging_config import game_logger

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
                events_file = f"{storage_manager.game_dir}public/events.jsonl"
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
                        event_time = datetime.fromisoformat(event["timestamp"]).timestamp()
                        if event_time >= cutoff_time:
                            kept_events.append(event)
                        else:
                            removed_events += 1
                    
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