import chromadb
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

# ================= 1. 配置与数据结构 =================

@dataclass
class AgentConfig:
    """Agent配置"""
    agent_id: str = "default_agent"# 用户及Agent身份标识
    game_id: str = "default_game"  # 房间标识（用以避免多批次同时开始的混乱
    server_url: str = "ws://judge-server"  # 法官服务器的 WebSocket 地址
    llm_config: Dict[str, Any] = None  # 大语言模型的配置参数
    speech_style: str = "moderate"  # 发言风格：aggressive/moderate/conservative
    risk_tolerance: float = 0.5  # 决策参数：0.0-1.0，其中0.0为极度保守，表现为人云亦云隐藏身份，而1.0则为高度激进，狼人表现为直接悍跳带节奏，预言家发金水，村民大胆推理
    trust_threshold: float = 0.6  # 信任阈值：0.0-1.0，其中0.0为曹贼一般用人必疑，1.0为轻易信任，当信任参数超过预设的信任阈值时决定采信
    decision_delay: float = 2.0  # 模拟思考时间（秒）
    max_memory_entries: int = 100  # 最大记忆条目数（定期清理旧记忆）
    log_level: str = "INFO"  # 控制日志输出详细程度，"DEBUG": 最详细，用于开发和调试；"INFO": 一般信息，适合正常游戏；"WARNING": 警告信息；"ERROR": 错误信息；"CRITICAL": 严重错误
    db_path: str = "./memory_db"


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



class AgentMemory:
    """
    双层记忆管理类：
    Layer 1 (RAM): self.entries -> 处理最近事件、高频逻辑查询 (速度极快)
    Layer 2 (Disk): ChromaDB -> 处理长期回忆、语义检索 (容量无限)
    """

    def __init__(self, config_or_max_entries=None, db_path="./memory_db"):
        """兼容多种初始化方式"""

        # 1. 处理配置差异
        if isinstance(config_or_max_entries, AgentConfig):
            self.config = config_or_max_entries
        elif isinstance(config_or_max_entries, int):
            self.config = AgentConfig(max_memory_entries=config_or_max_entries, db_path=db_path)
        else:
            self.config = AgentConfig(max_memory_entries=100, db_path=db_path)

        print(f"初始化双层记忆系统 (Max Entries: {self.config.max_memory_entries})...")

        # --- Layer 1: 内存层初始化 ---
        self.entries: List[MemoryEntry] = []
        self.event_index: Dict[str, List[MemoryEntry]] = {}

        # --- Layer 2: 向量层初始化 ---
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        #为每个 Agent 生成独立的数据库路径
        agent_id = getattr(self.config, 'agent_id', 'default_agent')
        if self.config.db_path == "./memory_db":
            final_db_path = f"./memory_db/{agent_id}"
        else:
            final_db_path = self.config.db_path

        print(f"数据库路径: {final_db_path}")

        self.client = chromadb.PersistentClient(path=final_db_path)
        self.collection = self.client.get_or_create_collection(name="events")

        print("记忆模块就绪。")

    def add_event(self, event: Dict, text_description: str = ""):
        """
        [核心] 添加事件：同时写入内存和向量数据库
        """
        if not text_description:
            etype = event.get("event_type")
            data = event.get("data", {})

            if etype == "player_speech":
                # 提取发言内容
                pid = data.get("player_id", "?")
                content = data.get("content", "")
                text_description = f"{pid}号玩家发言说：{content}"

            elif etype == "vote_result":
                # 提取投票结果
                votes = data.get("votes", {})
                res = data.get("result", "")
                text_description = f"投票结束。结果：{res}。详细票型：{votes}"

            elif etype == "player_death":
                pid = data.get("player_id", "?")
                text_description = f"{pid}号玩家死亡。"

            else:
                text_description = f"事件: {etype} | 数据: {data}"


        # 2. 计算衍生属性
        importance = self._calculate_importance(event)
        tags = self._generate_tags(event)

        # 3. 生成向量
        vector = self.encoder.encode(text_description).tolist()

        # 4. 准备元数据
        event_id = event.get("event_id", str(uuid.uuid4()))
        day = event.get("data", {}).get("day", 0)
        phase = event.get("phase", "unknown")
        timestamp = event.get("timestamp", datetime.now().isoformat())

        # ===========================
        # 存入 Layer 2: ChromaDB (持久化)
        # ===========================
        metadata = {
            "type": event.get("event_type", "unknown"),
            "day": day,
            "phase": phase,
            "timestamp": timestamp,
            "importance": importance
        }
        self.collection.add(
            ids=[event_id],
            embeddings=[vector],
            documents=[text_description],
            metadatas=[metadata]
        )

        # ===========================
        # 存入 Layer 1: 内存列表 (快速访问)
        # ===========================
        new_entry = MemoryEntry(
            id=event_id,
            timestamp=timestamp,
            day=day,
            phase=phase,
            event_type=event.get("event_type", "unknown"),
            content=event.get("data", {}),
            text=text_description,
            importance=importance,
            tags=tags,
            embedding=vector
        )

        self.entries.append(new_entry)

        # 更新索引
        if new_entry.event_type not in self.event_index:
            self.event_index[new_entry.event_type] = []
        self.event_index[new_entry.event_type].append(new_entry)

        # 内存限制清理
        if len(self.entries) > self.config.max_memory_entries:
            self._remove_least_important()

        print(f"[存入] Day{day} | Imp={importance:.1f} | {text_description[:40]}...")

    # ================= 逻辑检索 =================

    def add_phase_change(self, old_phase: str, new_phase: str):
        """记录阶段变更"""
        phase_map = {
            "werewolf_night": "狼人行动阶段",
            "seer_night": "预言家行动阶段",
            "witch_night": "女巫行动阶段",
            "daytime_discussion": "白天讨论阶段",
            "daytime_voting": "投票阶段",
            "game_end": "游戏结束"
        }

        old_cn = phase_map.get(old_phase, old_phase)
        new_cn = phase_map.get(new_phase, new_phase)

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "phase_change",
            "timestamp": datetime.now().isoformat(),
            "data": {"old_phase": old_phase, "new_phase": new_phase}
        }

        text = f"【系统公告】游戏阶段从 {old_cn} 变更为 {new_cn}。"
        self.add_event(event, text_description=text)

    def get_summary(self, limit: int = 10) -> List[Dict]:
        """获取内存中最重要的几条记忆"""
        sorted_entries = sorted(self.entries, key=lambda x: x.importance, reverse=True)
        return [asdict(e) for e in sorted_entries[:limit]]

    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """按标签精确搜索 (内存层)"""
        results = []
        for entry in self.entries:
            if tag in entry.tags:
                results.append(entry)
        return results

    def get_recent_events(self, event_type: str = None, limit: int = 5) -> List[MemoryEntry]:
        """获取最近发生的事件 (内存层)"""
        if event_type and event_type in self.event_index:
            entries = self.event_index[event_type]
        else:
            entries = self.entries
        return entries[-limit:] if entries else []



    def get_relevant_context(self, query: str, top_k: int = 5, day_filter: int = None, type_filter: str = None,
                             max_chars: int = 2000) -> str:
        """
        语义检索 + 逻辑过滤 + 窗口长度控制
        :param query: 用户的提问
        :param top_k: 尝试检索出的最大条数
        :param day_filter: 按天过滤 (可选)
        :param type_filter: 按类型过滤 (可选)
        :param max_chars: [新增] 返回文本的最大字符数，防止撑爆 LLM 上下文
        """
        print(f"[正在回忆] 思考: {query} (过滤条件: Day={day_filter}, Type={type_filter})")

        # 1. 构造过滤条件
        where_filter = {}
        if day_filter is not None:
            where_filter["day"] = day_filter
        if type_filter is not None:
            where_filter["type"] = type_filter
        final_where = where_filter if where_filter else None

        # 2. 检索
        query_vector = self.encoder.encode(query).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=final_where
            )
        except Exception as e:
            return f"【记忆检索】: 关于“{query}”没有找到匹配记录。"

        # 3. 提取结果
        if not results['documents'] or not results['documents'][0]:
            return f"【记忆检索】: 关于“{query}”没有找到匹配记录。"

        context_str = f"【关于“{query}”的相关记忆】:\n"
        current_len = len(context_str)
        found_docs = results['documents'][0]
        found_metas = results['metadatas'][0]

        for i, doc in enumerate(found_docs):
            day = found_metas[i].get('day', '?')
            kind = found_metas[i].get('type', 'unk')
            entry_text = f"- [Day {day} | {kind}] {doc}\n"

            if current_len + len(entry_text) > max_chars:
                context_str += "...(略)...\n"
                break

            context_str += entry_text
            current_len += len(entry_text)

        return context_str

    # ================= 内部工具方法 (私有) =================

    def _calculate_importance(self, event: Dict) -> float:
        """计算重要性 (保留组长逻辑)"""
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        # 基础分
        scores = {
            "phase_change": 0.2,  # 阶段变化没那么重要
            "vote_result": 0.9,  # 投票结果很重要
            "night_reveal": 1.0,  # 昨晚死人了，由于很重要
            "player_death": 1.0,
            "speech": 0.6,
        }
        base = scores.get(event_type, 0.5)

        res = data.get("result")

        # 只有当 result 存在，且它真的是个字典时，才去查 exiled_player
        if isinstance(res, dict) and res.get("exiled_player"):
            base += 0.1

        return min(base, 1.0)

    def _generate_tags(self, event: Dict) -> List[str]:
        """打标签"""
        # 1. 基础标签
        tags = [event.get("event_type", "unknown")]
        data = event.get("data", {})

        if "player_id" in data:
            tags.append(f"player_{data['player_id']}")

        # 2. 关键词提取
        if event.get("event_type") == "player_speech":

            content = data.get("content", "").lower()

            keywords = {
                "狼人": "mentions_werewolf", "wolf": "mentions_werewolf",
                "预言家": "mentions_seer", "seer": "mentions_seer",
                "查杀": "mentions_check",
                "女巫": "mentions_witch",
                "银水": "mentions_save",
                "金水": "mentions_good",
                "自爆": "mentions_suicide",
                "投票": "mentions_vote"
            }

            for word, tag in keywords.items():
                if word in content:
                    tags.append(tag)

        return list(set(tags))

    def retrieve_day_events(self, day: int) -> str:
        """获取某日全量记录，用于生成总结"""
        try:
            results = self.collection.get(where={"day": day})
            if not results['documents']:
                return f"第 {day} 天无记录。"
            return "\n".join([f"- {doc}" for doc in results['documents']])
        except Exception:
            return "获取记录失败。"

    def save_summary(self, day: int, summary: str):
        """存入总结"""
        self.add_event({
            "event_id": f"summary_day_{day}",
            "event_type": "daily_summary",
            "timestamp": datetime.now().isoformat(),
            "data": {"day": day, "content": summary},
            "phase": "night"
        }, text_description=f"【第{day}天总结】：{summary}")

    def _remove_least_important(self):
        """内存清理"""
        if not self.entries: return
        self.entries.sort(key=lambda x: x.importance)
        removed = self.entries.pop(0)
        if removed.event_type in self.event_index:
            try:
                self.event_index[removed.event_type].remove(removed)
            except ValueError:
                pass


if __name__ == "__main__":
    memory = AgentMemory()

    try:
        memory.client.delete_collection("events")
        print("旧记忆已清除，正在重建...")
    except:
        pass
    memory.collection = memory.client.get_or_create_collection("events")

    # [修改] 测试数据现在与 add_event 逻辑完美对齐
    test_events = [
        {
            "text": "", # 故意留空，测试自动生成
            "data": {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "event_type": "player_speech",
                "phase": "day_discussion",
                "data": {"day": 1, "player_id": "3", "content": "我是预言家，昨晚查验了5号，他是狼人。"}
            }
        },
        {
            "text": "",
            "data": {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "event_type": "player_speech",
                "phase": "day_discussion",
                "data": {"day": 1, "player_id": "5", "content": "3号在胡说，我才是女巫，昨晚还没用药。"}
            }
        },
        {
            "text": "1号玩家投票给了5号。", # 手动指定文本
            "data": {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "event_type": "vote_result",
                "phase": "day_vote",
                "data": {"day": 1, "votes": {"1": "5"}}
            }
        }
    ]

    print("\n--- 开始存入测试数据 ---")
    for item in test_events:
        memory.add_event(item["data"], text_description=item["text"])

    print("\n--- 1. 验证标签生成 (Tag Search) ---")
    # 如果 logic 正确，3号的发言应该有 mentions_seer 和 mentions_werewolf 标签
    tags = memory.search_by_tag("mentions_seer")
    print(f"提到预言家的记录数: {len(tags)}")

    print("\n--- 2. 语义搜索 ---")
    print(memory.get_relevant_context("谁跳了预言家？"))

    print("\n--- 3. 过滤测试 ---")
    print(memory.get_relevant_context("5号", type_filter="vote_result"))
