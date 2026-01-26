import json
import os
import uuid
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from .manage import GameStorageManager
from .logging_config import game_logger
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

