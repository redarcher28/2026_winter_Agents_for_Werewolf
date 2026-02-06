# config.py
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional

class GamePhase(Enum):
    """游戏阶段枚举"""
    WEREWOLF_NIGHT = "werewolf_night"
    SEER_NIGHT = "seer_night"
    WITCH_NIGHT = "witch_night"
    DAYTIME_DISCUSSION = "daytime_discussion"
    DAYTIME_VOTING = "daytime_voting"
    GAME_END = "game_end"
    DAILY_SUMMARY = "daily_summary"

class Role(Enum):
    """角色枚举"""
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    VILLAGER = "villager"

class AgentState(Enum):
    """Agent状态枚举"""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    READY = "ready"
    PLAYING = "playing"
    WAITING = "waiting"
    DEAD = "dead"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class PlayerInfo:
    """玩家信息"""
    id: str
    name: str
    is_ai: bool
    is_alive: bool = True
    role: Optional[Role] = None

@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    timestamp: str
    day: int
    phase: str
    event_type: str
    content: Dict[str, Any]
    text: str
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = field(default=None, repr=False)

@dataclass
class StrategyDecision:
    """策略决策标准结构"""
    decision_type: str  # "speech", "vote", "night_action", "no_op"
    data: Dict[str, Any]
    confidence: float
    debug: Dict[str, Any] = None

@dataclass
class LLMConfig:
    """LLM 模型配置"""
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 60.0
    extra: Dict[str, Any] = None  # 用于支持 base_url 等额外参数

@dataclass
class AgentConfig:
    """Agent 全局配置"""
    agent_id: str
    game_id: str
    speech_style: str = "moderate"
    risk_tolerance: float = 0.5
    trust_threshold: float = 0.6
    decision_delay: float = 2.0
    max_memory_entries: int = 200
    log_level: str = "INFO"
    db_path: str = "./memory_db"
    llm: Optional[LLMConfig] = None
    verbose: bool = False  # 是否显示详细日志