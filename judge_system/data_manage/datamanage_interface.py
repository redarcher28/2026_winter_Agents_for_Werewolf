from enum import Enum
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


# ==================== 数据存储相关 ====================

class DataStorageInterface(ABC):
    """数据存储接口"""
    
    @abstractmethod
    def save_game_event(self, event_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def save_speech(self, speech_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def save_vote(self, vote_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def save_game_state(self, state_data: Dict[str, Any]) -> bool:
        pass


class DataBackupType(Enum):
    """数据备份类型"""
    FULL = "full"
    INCREMENTAL = "incremental"


class ExportFormat(Enum):
    """导出格式类型"""
    ZIP = "zip"
    JSON = "json"


# ==================== 游戏存储相关 ====================

class GameStorageInterface(ABC):
    """游戏存储接口"""
    
    @abstractmethod
    def save_game_event(self, event_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def save_speech(self, speech_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def save_vote(self, vote_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def save_game_state(self, state_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def get_public_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        pass


class EventType(Enum):
    """事件类型枚举"""
    UNKNOWN = "unknown"
    KILL = "kill"
    VOTE = "vote"
    CHECK = "check"
    PHASE_CHANGE = "phase_change"
    SPEECH = "speech"
    ROLE_REVEAL = "role_reveal"
    GAME_START = "game_start"
    GAME_END = "game_end"


class GamePhase(Enum):
    """游戏阶段枚举"""
    DAY = "day"
    NIGHT = "night"
    DISCUSSION = "discussion"
    VOTING = "voting"
    ENDED = "ended"


class StorageDirectoryType(Enum):
    """存储目录类型"""
    LOGS = "logs"
    AGENTS = "agents"
    BACKUPS = "backups"
    CONFIG = "config"
    PRIVATE = "private"
