import os
import sys
import shutil
import tempfile
from datetime import datetime
import uuid
import types
from enum import Enum
from typing import Dict, Any, Optional
import json
from abc import ABC, abstractmethod


# ===== IDataManager 抽象基类 =====
class IDataManager(ABC):
    @abstractmethod
    async def save_game_state(self, game_id: str, state: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def load_game_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        pass


# ===== 导入实际依赖模块 =====
# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 导入实际的interfaces模块
from judge_system.data_manage.datamanage_interface import (
    DataStorageInterface,
    DataBackupType,
    ExportFormat,
    GameStorageInterface,
    EventType,
    GamePhase,
    StorageDirectoryType
)

# 导入实际的observer_interface模块
from judge_system.observer_interface import (
    PlayerType,
    Role,
    PlayerStatus
)


# ===== 现在可以导入原始模块 =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# route 0-3: 导入GameStorageManager类
from judge_system.data_manage.manage import GameStorageManager


# 现在GameStorageManager已经包含了所有必要的方法，不需要扩展

# route 0-5: 导入DataStorageService类
from judge_system.data_manage.data_storage_service import DataStorageService


# ===== DataManager 类实现 =====
# route 12: 实现DataManager类，继承自IDataManager接口
class DataManager(IDataManager):
    """
    数据管理类，实现IDataManager接口
    使用DataStorageService来管理游戏状态的保存和加载
    """
    
    def __init__(self, base_data_dir: str = "./game_data"):
        """
        初始化DataManager
        
        Args:
            base_data_dir: 基础数据存储目录
        """
        self.storage_service = DataStorageService(base_data_dir=base_data_dir)
    
    async def save_game_state(self, game_id: str, state: Dict[str, Any]) -> bool:
        """
        保存游戏状态
        
        Args:
            game_id: 游戏ID
            state: 游戏状态数据
            
        Returns:
            保存是否成功
        """
        try:
            # 获取存储管理器
            storage_manager = self.storage_service._get_storage_manager(game_id)
            # 保存游戏状态
            result = storage_manager.save_game_state(state)
            return result
        except Exception as e:
            print(f"保存游戏状态失败: {e}")
            return False
    
    async def load_game_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        加载游戏状态
        
        Args:
            game_id: 游戏ID
            
        Returns:
            游戏状态数据，如果不存在则返回None
        """
        try:
            # 获取存储管理器
            storage_manager = self.storage_service._get_storage_manager(game_id)
            # 读取最新的游戏状态
            if os.path.exists(storage_manager.state_log_path):
                with open(storage_manager.state_log_path, 'r', encoding='utf-8') as f:
                    # 读取所有状态记录
                    states = []
                    for line in f:
                        try:
                            state = json.loads(line.strip())
                            states.append(state)
                        except json.JSONDecodeError:
                            continue
                    # 返回最新的状态
                    if states:
                        return states[-1]
            return None
        except Exception as e:
            print(f"加载游戏状态失败: {e}")
            return None


# ===== 测试函数 =====
# route 1: 主测试函数入口
def test_data_storage_service():
    """
    测试数据管理与存储服务功能
    这个函数展示了如何使用DataStorageService进行数据的存储、备份、恢复、清理、导出和导入操作
    """
    # 创建临时目录作为测试数据存储位置
    # fixme: 在系统temp目录中生成一个临时文件夹，但不会自己删除，
    #  修改：import tempfile
    #  from pathlib import Path
    # def safe_temp_operation():
    #     """安全的临时目录使用示例"""
    #
    #     # 方法1：使用TemporaryDirectory（推荐）
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         temp_path = Path(temp_dir)
    #
    #         # 在临时目录中创建文件
    #         temp_file = temp_path / "data.txt"
    #         temp_file.write_text("Hello, World!")
    #
    #         # 处理文件...
    #         content = temp_file.read_text()
    #
    #         # 不需要手动清理，离开with块自动清理
    temp_dir = tempfile.mkdtemp()
    print(f"测试临时目录: {temp_dir}")

    try:
        # 1. 初始化数据存储服务
        print("\n1. 初始化数据存储服务")
        # route 1-1: 创建DataStorageService实例
        storage_service = DataStorageService(base_data_dir=temp_dir)
        print(f"数据存储服务初始化成功，基础目录: {temp_dir}")

        # 2. 测试游戏数据管理
        print("\n2. 测试游戏数据管理")
        game_id = f"test_game_{uuid.uuid4().hex[:8]}"
        print(f"创建游戏ID: {game_id}")

        # 3. 测试玩家状态管理
        print("\n3. 测试玩家状态管理")
        # 使用实际的PlayerStatus类
        player_id = "player_1"
        player_status = {
            "player_id": player_id,
            "name": "测试玩家",
            "role": "villager",
            "status": "alive",
            "vote": None,
            "speech": "Hello, everyone!"
        }

        # route 2: 保存玩家状态测试
        # 保存玩家状态
        save_success = storage_service.save_player_status(game_id, player_status)
        print(f"保存玩家状态: {'成功' if save_success else '失败'}")

        # route 3: 加载玩家状态测试
        # 加载玩家状态
        loaded_status = storage_service.load_player_status(game_id, player_id)
        print(f"加载玩家状态: {'成功' if loaded_status else '失败'}")
        if loaded_status:
            print(f"玩家名称: {loaded_status.get('name')}")
            print(f"玩家角色: {loaded_status.get('role')}")

        # 4. 测试角色行动管理
        print("\n4. 测试角色行动管理")

        # route 4: 女巫行动测试
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
        # route 5: 数据备份测试
        backup_result = storage_service.create_backup(game_id, DataBackupType.FULL, "测试备份")
        backup_success, backup_id, backup_info = backup_result
        print(f"创建备份: {'成功' if backup_success else '失败'}")
        if backup_success:
            print(f"备份ID: {backup_id}")
            print(f"备份类型: {backup_info.get('backup_type')}")
            print(f"备份大小: {backup_info.get('total_size_mb', 0):.2f} MB")

        # 6. 测试数据统计信息
        print("\n6. 测试数据统计信息")
        # route 6: 数据统计测试
        statistics = storage_service.get_data_statistics(game_id)
        print(f"获取数据统计: {'成功' if 'error' not in statistics else '失败'}")
        if 'error' not in statistics:
            print(f"游戏ID: {statistics.get('game_id')}")
            print(f"存储摘要: {statistics.get('storage_summary', {}).get('total_size_mb', 0):.2f} MB")
            print(f"备份数量: {statistics.get('backup_statistics', {}).get('backup_count', 0)}")

        # 7. 测试数据清理
        print("\n7. 测试数据清理")
        # route 7: 数据清理测试
        clean_result = storage_service.clean_data(game_id, days_to_keep=1, clean_old_events=True,
                                                  clean_old_backups=False)
        clean_success, clean_info = clean_result
        print(f"清理数据: {'成功' if clean_success else '失败'}")
        if clean_success:
            print(f"清理时间: {clean_info.get('timestamp')}")
            print(f"保留天数: {clean_info.get('days_to_keep')}")
            print(f"删除事件数: {clean_info.get('events_removed')}")

        # 8. 测试数据导出
        print("\n8. 测试数据导出")
        # route 8: 数据导出测试
        export_result = storage_service.export_data(game_id, ExportFormat.ZIP)
        export_success, export_path, export_info = export_result
        print(f"导出数据: {'成功' if export_success else '失败'}")
        if export_success:
            print(f"导出文件路径: {export_path}")
            print(f"导出格式: {export_info.get('export_format')}")
            print(f"导出大小: {export_info.get('total_size_mb', 0):.2f} MB")

        # 9. 测试数据导入
        print("\n9. 测试数据导入")
        # route 9: 数据导入测试
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
        # route 10: 获取所有游戏测试
        all_games = storage_service.get_all_games()
        print(f"获取所有游戏: 共 {len(all_games)} 个游戏")
        for game in all_games:
            print(f"  - 游戏ID: {game.get('game_id')}")

        # 11. 测试删除游戏数据
        print("\n11. 测试删除游戏数据")
        # route 11: 删除游戏数据测试
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


# ===== 主函数 =====
def main():
    """
    主函数，用于演示DataManager的使用
    """
    import asyncio
    
    async def test_data_manager():
        """
        测试DataManager类的功能
        """
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"测试临时目录: {temp_dir}")
        
        try:
            # 初始化DataManager
            data_manager = DataManager(base_data_dir=temp_dir)
            print("\n1. 初始化DataManager成功")
            
            # 测试游戏ID
            game_id = "test_game_123"
            
            # 测试保存游戏状态
            test_state = {
                "game_id": game_id,
                "phase": "day",
                "day_number": 1,
                "alive_players": ["player_1", "player_2"],
                "dead_players": [],
                "current_speaker": "player_1",
                "vote_results": {},
                "last_night_actions": {}
            }
            
            print("\n2. 测试保存游戏状态")
            save_result = await data_manager.save_game_state(game_id, test_state)
            print(f"保存游戏状态: {'成功' if save_result else '失败'}")
            
            # 测试加载游戏状态
            print("\n3. 测试加载游戏状态")
            loaded_state = await data_manager.load_game_state(game_id)
            print(f"加载游戏状态: {'成功' if loaded_state else '失败'}")
            if loaded_state:
                print(f"游戏阶段: {loaded_state.get('phase')}")
                print(f"天数: {loaded_state.get('day_number')}")
                print(f"存活玩家: {loaded_state.get('alive_players')}")
            
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"\n清理临时目录: {temp_dir}")
    
    # 运行异步测试
    asyncio.run(test_data_manager())


if __name__ == "__main__":
    # 运行DataManager测试
    main()
    
    # 运行完整的DataStorageService测试
    print("\n" + "="*50)
    print("运行完整的DataStorageService测试")
    print("="*50)
    test_data_storage_service()
