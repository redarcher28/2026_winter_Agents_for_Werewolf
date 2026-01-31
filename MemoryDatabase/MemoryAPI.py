#该文件仅用于参考组员D是否已经完善了接口，不作用于实际生产

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime

@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    timestamp: str
    day: int  # [新增] 记录是第几天发生的，方便按时间过滤
    phase: str  # [新增] 记录阶段 (day/night)
    event_type: str
    content: Dict[str, Any]
    text: str  # [新增] 自然语言描述 (给 LLM 阅读)
    embedding: List[float]  # [新增] 向量数据 (给 ChromaDB 搜索)
    importance: float  # 0.0-1.0，重要性评分
    tags: List[str]  # 标签，如["谎言", "投票模式", "可疑行为"]

@dataclass
class AgentConfig:
    """Agent配置"""
    max_memory_entries: int = 100  # 最大记忆条目数（定期清理旧记忆）


class BaseWerewolfAgent(ABC):
    def __init__(self, config: AgentConfig):
        #仅保留记忆模块相关，其余忽略
        self.memory = AgentMemory(max_entries=config.max_memory_entries)

class AgentMemory:
    """记忆管理类"""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.event_index: Dict[str, List[MemoryEntry]] = {}

    def add_event(self, event: Dict):
        """添加事件到记忆"""
        entry = MemoryEntry(
            id=event.get("event_id", f"evt_{len(self.entries)}"),
            timestamp=event.get("timestamp", datetime.now().isoformat()),
            event_type=event.get("event_type", "unknown"),
            content=event.get("data", {}),
            importance=self._calculate_importance(event),
            tags=self._generate_tags(event)
        )

        # 添加条目
        self.entries.append(entry)

        # 更新索引
        if entry.event_type not in self.event_index:
            self.event_index[entry.event_type] = []
        self.event_index[entry.event_type].append(entry)

        # 保持条目数量不超过限制
        if len(self.entries) > self.max_entries:
            self._remove_least_important()

    def add_phase_change(self, old_phase: str, new_phase: str):
        """添加阶段变更记忆"""
        event = {
            "event_id": f"phase_{len(self.entries)}",
            "event_type": "phase_change",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "old_phase": old_phase,
                "new_phase": new_phase
            }
        }
        self.add_event(event)

    def get_summary(self, limit: int = 10) -> List[Dict]:
        """获取记忆摘要，按重要性排序"""
        sorted_entries = sorted(self.entries,
                                key=lambda x: x.importance,
                                reverse=True)
        return [asdict(entry) for entry in sorted_entries[:limit]]

    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """按标签搜索记忆"""
        results = []
        for entry in self.entries:
            if tag in entry.tags:
                results.append(entry)
        return results

    def get_recent_events(self, event_type: str = None, limit: int = 5) -> List[MemoryEntry]:
        """获取最近事件"""
        if event_type and event_type in self.event_index:
            entries = self.event_index[event_type]
        else:
            entries = self.entries

        return entries[-limit:] if entries else []

    def _calculate_importance(self, event: Dict) -> float:
        """计算事件重要性"""
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        importance_scores = {
            "phase_change": 0.8,
            "vote_result": 0.9,
            "night_reveal": 0.9,
            "player_death": 0.95,
            "player_speech": 0.3,
        }

        base_score = importance_scores.get(event_type, 0.1)

        # 根据内容调整分数
        if "result" in data and data.get("result", {}).get("exiled_player"):
            base_score += 0.1

        return min(base_score, 1.0)

    def _generate_tags(self, event: Dict) -> List[str]:
        """生成事件标签"""
        tags = [event.get("event_type", "unknown")]
        data = event.get("data", {})

        if "player_id" in data:
            tags.append(f"player_{data['player_id']}")

        if event.get("event_type") == "player_speech":
            content = data.get("content", "").lower()
            if any(word in content for word in ["狼人", "狼", "werewolf"]):
                tags.append("mentions_werewolf")
            if any(word in content for word in ["预言家", "seer"]):
                tags.append("mentions_seer")

        return tags

    def _remove_least_important(self):
        """移除最不重要的条目"""
        if not self.entries:
            return

        # 按重要性排序
        self.entries.sort(key=lambda x: x.importance)

        # 移除最不重要的条目
        removed = self.entries.pop(0)

        # 更新索引
        if removed.event_type in self.event_index:
            self.event_index[removed.event_type].remove(removed)
            if not self.event_index[removed.event_type]:
                del self.event_index[removed.event_type]
