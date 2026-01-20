import json
import os
import uuid
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from judge_system.manage import GameStorageManager
from logging_config import game_logger
from interfaces import EventType

# ==================== 核心数据存储管理接口 ====================

# 初始化日志器
game_controller_logger = game_logger.get_service_logger("game_controller")

async def create_new_game(
    game_config: Dict[str, Any]  # 游戏配置
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    创建新游戏（仅保留配置存储逻辑）
    """
    try:
        # 生成游戏ID（与game_control_service.py保持一致）
        import uuid
        game_id = f"{uuid.uuid4().hex[:12]}"
        
        # 创建游戏存储管理器
        storage_manager = GameStorageManager(game_id=game_id)
        
        # 保存游戏配置（核心存储操作）
        storage_manager.save_public_event({
            "event_type": EventType.GAME_CREATED.value,
            "game_config": game_config,
            "timestamp": datetime.now().isoformat()
        })
        
        # 返回成功结果
        return True, game_id, {
            "game_id": game_id,
            "created_at": datetime.now().isoformat(),
            "game_config": game_config
        }
        
    except Exception as e:
        return False, "", {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def end_current_game(
    game_id: str,          # 游戏ID
    end_reason: str,       # 结束原因
    game_summary: Dict[str, Any]  # 游戏总结
) -> bool:
    """
    结束当前游戏（仅保留总结数据存储逻辑）
    """
    try:
        # 创建存储管理器
        storage_manager = GameStorageManager(game_id=game_id)
        
        # 保存游戏结束事件（数据持久化）
        storage_manager.save_public_event({
            "event_type": EventType.GAME_ENDED.value,
            "end_reason": end_reason,
            "game_summary": game_summary,
            "timestamp": datetime.now().isoformat()
        })
        
        # 保存游戏总结到文件（核心存储操作）
        summary_file = os.path.join(storage_manager.game_dir, "game_summary.json")
        # 确保目录存在
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(game_summary, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        game_controller_logger.error(f"Error ending game {game_id}: {e}")
        return False

async def save_game_snapshot(
    game_id: str,                # 游戏ID
    snapshot_name: str = "auto"  # 快照名称
) -> Tuple[bool, str]:
    """
    保存游戏快照（核心数据备份逻辑）
    """
    try:
        # 创建存储管理器
        storage_manager = GameStorageManager(game_id=game_id)
        
        # 生成快照ID
        snapshot_id = f"snap_{int(datetime.now().timestamp()*1000)}"
        
        # 获取当前游戏事件数据
        current_events = storage_manager.get_public_events(limit=1000)
        
        # 构建快照元数据
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "snapshot_name": snapshot_name,
            "game_id": game_id,
            "timestamp": datetime.now().isoformat(),
            "event_count": len(current_events)
        }
        
        # 保存快照元数据（核心存储操作）
        snapshot_dir = os.path.join(storage_manager.game_dir, "backups/")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_file = os.path.join(snapshot_dir, f"snapshot_{snapshot_id}.json")
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
        
        # 保存游戏状态快照
        state_snapshot = {
            "snapshot_id": snapshot_id,
            "public_events": current_events
        }
        state_file = os.path.join(snapshot_dir, f"state_{snapshot_id}.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_snapshot, f, ensure_ascii=False, indent=2)
        
        return True, snapshot_id
        
    except Exception as e:
        print(f"Error saving game snapshot: {e}")
        return False, ""

async def load_game_snapshot(
    game_id: str,      # 游戏ID
    snapshot_id: str   # 快照ID
) -> Tuple[bool, Dict[str, Any]]:
    """
    加载游戏快照（核心数据恢复逻辑）
    """
    try:
        # 创建存储管理器
        storage_manager = GameStorageManager(game_id=game_id)
        
        # 加载快照文件（核心读取操作）
        state_file = os.path.join(storage_manager.game_dir, "backups", f"state_{snapshot_id}.json")
        
        if not os.path.exists(state_file):
            return False, {"error": "Snapshot not found"}
        
        with open(state_file, 'r', encoding='utf-8') as f:
            snapshot_data = json.load(f)
        
        # 返回快照数据
        return True, snapshot_data
        
    except Exception as e:
        print(f"Error loading game snapshot: {e}")
        return False, {"error": str(e)}