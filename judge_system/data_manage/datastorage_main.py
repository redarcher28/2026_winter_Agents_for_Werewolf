# -*- coding: utf-8 -*-
import os
import sys
import shutil
import tempfile
from datetime import datetime
import uuid
from enum import Enum

# ===== 模拟依赖模块 =====
# 首先创建模拟的interfaces模块
class MockDataStorageInterface:
    pass

class MockDataBackupType(Enum):
    FULL = "full"

class MockExportFormat(Enum):
    ZIP = "zip"

class MockGameStorageInterface:
    pass

class MockEventType(Enum):
    UNKNOWN = "unknown"

class MockGamePhase(Enum):
    DAY = "day"
    NIGHT = "night"

class MockStorageDirectoryType(Enum):
    LOGS = "logs"
    AGENTS = "agents"
    BACKUPS = "backups"
    CONFIG = "config"
    PRIVATE = "private"

# 模拟observer_interface模块
class MockPlayerType:
    pass

class MockRole:
    pass

class MockPlayerStatus:
    pass

# 将模拟的类注册为模块
sys.modules['interfaces'] = type('obj', (object,), {
    'DataStorageInterface': MockDataStorageInterface,
    'DataBackupType': MockDataBackupType,
    'ExportFormat': MockExportFormat,
    'GameStorageInterface': MockGameStorageInterface,
    'EventType': MockEventType,
    'GamePhase': MockGamePhase,
    'StorageDirectoryType': MockStorageDirectoryType
})

# 模拟judge_system.observer_interface模块
class MockObserverModule:
    PlayerType = MockPlayerType
    Role = MockRole
    PlayerStatus = MockPlayerStatus

sys.modules['judge_system.observer_interface'] = MockObserverModule

# ===== 现在可以导入原始模块 =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from judge_system.data_manage.manage import GameStorageManager

# 扩展GameStorageManager类，添加缺失的方法
class ExtendedGameStorageManager(GameStorageManager):
    def __init__(self, game_id, base_dir):
        super().__init__(game_id, base_dir)
        self._player_statuses = {}
    
    def save_player_status(self, player_status):
        """
        保存玩家状态
        """
        player_id = player_status.get('player_id')
        if player_id:
            self._player_statuses[player_id] = player_status
            return True
        return False
    
    def load_player_status(self, player_id):
        """
        加载玩家状态
        """
        return self._player_statuses.get(player_id)
    
    def save_all_player_statuses(self, player_statuses):
        """
        保存所有玩家状态
        """
        self._player_statuses.update(player_statuses)
        return True
    
    def load_all_player_statuses(self):
        """
        加载所有玩家状态
        """
        return self._player_statuses

# 替换原始的GameStorageManager
import judge_system.data_manage.manage
judge_system.data_manage.manage.GameStorageManager = ExtendedGameStorageManager

from judge_system.data_manage.data_storage_service import DataStorageService

# ===== 测试函数 =====
def test_data_storage_service():
    """
    测试数据管理与存储服务功能
    这个函数展示了如何使用DataStorageService进行数据的存储、备份、恢复、清理、导出和导入操作
    """
    # 创建临时目录作为测试数据存储位置
    temp_dir = tempfile.mkdtemp()
    print(f"测试临时目录: {temp_dir}")
    
    try:
        # 1. 初始化数据存储服务
        print("\n1. 初始化数据存储服务")
        storage_service = DataStorageService(base_data_dir=temp_dir)
        print(f"数据存储服务初始化成功，基础目录: {temp_dir}")
        
        # 2. 测试游戏数据管理
        print("\n2. 测试游戏数据管理")
        game_id = f"test_game_{uuid.uuid4().hex[:8]}"
        print(f"创建游戏ID: {game_id}")
        
        # 3. 测试玩家状态管理
        print("\n3. 测试玩家状态管理")
        player_id = "player_1"
        player_status = {
            "player_id": player_id,
            "name": "测试玩家",
            "role": "villager",
            "status": "alive",
            "vote": None,
            "speech": "Hello, everyone!"
        }
        
        # 保存玩家状态
        save_success = storage_service.save_player_status(game_id, player_status)
        print(f"保存玩家状态: {'成功' if save_success else '失败'}")
        
        # 加载玩家状态
        loaded_status = storage_service.load_player_status(game_id, player_id)
        print(f"加载玩家状态: {'成功' if loaded_status else '失败'}")
        if loaded_status:
            print(f"玩家名称: {loaded_status.get('name')}")
            print(f"玩家角色: {loaded_status.get('role')}")
        
        # 4. 测试角色行动管理
        print("\n4. 测试角色行动管理")
        
        # 保存女巫行动
        witch_id = "witch_1"
        witch_action = {
            "use_heal": True,
            "use_poison": False,
            "heal_target": player_id,
            "poison_target": None
        }
        witch_save_success = storage_service.save_witch_action(game_id, witch_id, witch_action)
        print(f"保存女巫行动: {'成功' if witch_save_success else '失败'}")
        
        # 获取女巫行动
        loaded_witch_action = storage_service.get_witch_action(game_id)
        print(f"获取女巫行动: {'成功' if loaded_witch_action else '失败'}")
        if loaded_witch_action:
            print(f"是否使用解药: {loaded_witch_action.get('use_heal')}")
        
        # 5. 测试数据备份
        print("\n5. 测试数据备份")
        backup_result = storage_service.create_backup(game_id, MockDataBackupType.FULL, "测试备份")
        backup_success, backup_id, backup_info = backup_result
        print(f"创建备份: {'成功' if backup_success else '失败'}")
        if backup_success:
            print(f"备份ID: {backup_id}")
            print(f"备份类型: {backup_info.get('backup_type')}")
            print(f"备份大小: {backup_info.get('total_size_mb', 0):.2f} MB")
        
        # 6. 测试数据统计信息
        print("\n6. 测试数据统计信息")
        statistics = storage_service.get_data_statistics(game_id)
        print(f"获取数据统计: {'成功' if 'error' not in statistics else '失败'}")
        if 'error' not in statistics:
            print(f"游戏ID: {statistics.get('game_id')}")
            print(f"存储摘要: {statistics.get('storage_summary', {}).get('total_size_mb', 0):.2f} MB")
            print(f"备份数量: {statistics.get('backup_statistics', {}).get('backup_count', 0)}")
        
        # 7. 测试数据清理
        print("\n7. 测试数据清理")
        clean_result = storage_service.clean_data(game_id, days_to_keep=1, clean_old_events=True, clean_old_backups=False)
        clean_success, clean_info = clean_result
        print(f"清理数据: {'成功' if clean_success else '失败'}")
        if clean_success:
            print(f"清理时间: {clean_info.get('timestamp')}")
            print(f"保留天数: {clean_info.get('days_to_keep')}")
            print(f"删除事件数: {clean_info.get('events_removed')}")
        
        # 8. 测试数据导出
        print("\n8. 测试数据导出")
        export_result = storage_service.export_data(game_id, MockExportFormat.ZIP)
        export_success, export_path, export_info = export_result
        print(f"导出数据: {'成功' if export_success else '失败'}")
        if export_success:
            print(f"导出文件路径: {export_path}")
            print(f"导出格式: {export_info.get('export_format')}")
            print(f"导出大小: {export_info.get('total_size_mb', 0):.2f} MB")
        
        # 9. 测试数据导入
        print("\n9. 测试数据导入")
        if export_success:
            import_game_id = f"imported_game_{uuid.uuid4().hex[:8]}"
            import_result = storage_service.import_data(export_path, import_game_id, overwrite=True)
            import_success, imported_game_id, import_info = import_result
            print(f"导入数据: {'成功' if import_success else '失败'}")
            if import_success:
                print(f"导入游戏ID: {imported_game_id}")
                print(f"导入文件路径: {import_info.get('import_path')}")
                print(f"导入大小: {import_info.get('total_size', 0) / (1024 * 1024):.2f} MB")
        
        # 10. 测试获取所有游戏
        print("\n10. 测试获取所有游戏")
        all_games = storage_service.get_all_games()
        print(f"获取所有游戏: 共 {len(all_games)} 个游戏")
        for game in all_games:
            print(f"  - 游戏ID: {game.get('game_id')}")
        
        # 11. 测试删除游戏数据
        print("\n11. 测试删除游戏数据")
        delete_success = storage_service.delete_game_data(game_id, delete_backups=True)
        print(f"删除游戏数据: {'成功' if delete_success else '失败'}")
        
        if 'imported_game_id' in locals():
            delete_import_success = storage_service.delete_game_data(imported_game_id, delete_backups=True)
            print(f"删除导入的游戏数据: {'成功' if delete_import_success else '失败'}")
        
        print("\n测试完成!")
        
    finally:
        # 清理临时目录
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n清理临时目录: {temp_dir}")

if __name__ == "__main__":
    test_data_storage_service()
