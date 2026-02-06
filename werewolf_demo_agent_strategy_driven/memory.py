import uuid
import chromadb
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from config import AgentConfig, MemoryEntry
import json
from collections import defaultdict


# ==================== 记忆图结构 ====================

class MemoryGraph:
    """
    记忆关系图：用于存储记忆节点之间的关系
    
    关系类型：
    - temporal: 时间顺序关系（A发生在B之前）
    - causal: 因果关系（A导致了B）
    - reference: 引用关系（A提到了B）
    - player_related: 玩家关联（都涉及同一玩家）
    - contradiction: 矛盾关系（A和B互相矛盾）
    """
    
    def __init__(self):
        # 邻接表：{memory_id: {relation_type: [related_memory_ids]}}
        self.edges: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # 节点属性：{memory_id: {属性字典}}
        self.nodes: Dict[str, Dict] = {}
        # 玩家索引：{player_id: [memory_ids]}
        self.player_index: Dict[str, List[str]] = defaultdict(list)
        # 事件类型索引：{event_type: [memory_ids]}
        self.type_index: Dict[str, List[str]] = defaultdict(list)
    
    def add_node(self, memory_id: str, attributes: Dict):
        """添加记忆节点"""
        self.nodes[memory_id] = attributes
        
        # 更新索引
        if 'players' in attributes:
            for player in attributes['players']:
                self.player_index[player].append(memory_id)
        
        if 'event_type' in attributes:
            self.type_index[attributes['event_type']].append(memory_id)
    
    def add_edge(self, from_id: str, to_id: str, relation_type: str):
        """添加关系边"""
        if to_id not in self.edges[from_id][relation_type]:
            self.edges[from_id][relation_type].append(to_id)
    
    def get_neighbors(self, memory_id: str, relation_types: List[str] = None) -> List[str]:
        """获取相邻节点"""
        if memory_id not in self.edges:
            return []
        
        neighbors = []
        if relation_types is None:
            # 返回所有类型的邻居
            for rel_dict in self.edges[memory_id].values():
                neighbors.extend(rel_dict)
        else:
            # 返回指定类型的邻居
            for rel_type in relation_types:
                if rel_type in self.edges[memory_id]:
                    neighbors.extend(self.edges[memory_id][rel_type])
        
        return list(set(neighbors))  # 去重
    
    def get_related_by_player(self, player_id: str) -> List[str]:
        """获取与某玩家相关的所有记忆"""
        return self.player_index.get(player_id, [])
    
    def get_related_by_type(self, event_type: str) -> List[str]:
        """获取某类型的所有记忆"""
        return self.type_index.get(event_type, [])
    
    def find_path(self, start_id: str, end_id: str, max_depth: int = 3) -> List[List[str]]:
        """查找两个记忆之间的路径（BFS）"""
        if start_id not in self.edges or end_id not in self.nodes:
            return []
        
        queue = [(start_id, [start_id])]
        visited = {start_id}
        paths = []
        
        while queue and len(paths) < 5:  # 最多返回5条路径
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current == end_id:
                paths.append(path)
                continue
            
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return paths


class AgentMemory:
    """
    双层记忆管理类：
    Layer 1 (RAM): self.entries -> 处理最近事件、高频逻辑查询 (速度极快)
    Layer 2 (Disk): ChromaDB -> 处理长期回忆、语义检索 (容量无限)
    
    增强功能：
    - 完全兼容 decision.py 的所有查询需求
    - 提供多种访问接口（entries、get_recent_events、search_by_tag等）
    - 优雅降级（即使某些功能不可用也不会崩溃）
    """

    def __init__(self, config_or_max_entries=None, db_path="./memory_db"):
        """兼容多种初始化方式"""

        # 1. 处理配置差异
        if isinstance(config_or_max_entries, AgentConfig):
            self.config = config_or_max_entries
        elif isinstance(config_or_max_entries, int):
            self.config = AgentConfig(
                agent_id="default_agent",
                game_id="default_game",
                max_memory_entries=config_or_max_entries, 
                db_path=db_path
            )
        else:
            self.config = AgentConfig(
                agent_id="default_agent",
                game_id="default_game",
                max_memory_entries=100, 
                db_path=db_path
            )

        # 日志模式（从配置中读取，如果没有则默认关闭）
        self.verbose = getattr(config_or_max_entries, 'verbose', False) if isinstance(config_or_max_entries, AgentConfig) else False
        
        if self.verbose:
            print(f"初始化双层记忆系统 (Max Entries: {self.config.max_memory_entries})...")

        # --- Layer 1: 内存层初始化 ---
        self.entries: List[MemoryEntry] = []  # ✅ 必须暴露给decision
        self.event_index: Dict[str, List[MemoryEntry]] = {}

        # --- Layer 2: 向量层初始化 ---
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # 为每个 Agent 生成独立的数据库路径
        agent_id = getattr(self.config, 'agent_id', 'default_agent')
        if self.config.db_path == "./memory_db":
            final_db_path = f"./memory_db/{agent_id}"
        else:
            final_db_path = self.config.db_path

        if self.verbose:
            print(f"数据库路径: {final_db_path}")

        self.client = chromadb.PersistentClient(path=final_db_path)
        self.collection = self.client.get_or_create_collection(name="events")

        # --- Layer 3: 图结构层初始化 (记忆关系图) ---
        self.memory_graph = MemoryGraph()
        
        if self.verbose:
            print("记忆模块就绪（含图结构增强）。")

    # ==================== 核心接口（decision.py 依赖） ====================
    
    def add_event(self, event: Dict, text_description: str = ""):
        """
        [核心] 添加事件：同时写入内存和向量数据库
        """
        if not text_description:
            text_description = self._auto_generate_description(event)

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

        self.entries.append(new_entry)  # ✅ decision 需要访问

        # 更新索引
        if new_entry.event_type not in self.event_index:
            self.event_index[new_entry.event_type] = []
        self.event_index[new_entry.event_type].append(new_entry)

        # 更新图结构：建立记忆节点和关系
        self._update_memory_graph(new_entry, event)

        # 内存限制清理
        if len(self.entries) > self.config.max_memory_entries:
            self._remove_least_important()

        if self.verbose:
            print(f"[存入] Day{day} | Imp={importance:.1f} | {text_description[:40]}...")

    def _auto_generate_description(self, event: Dict) -> str:
        """自动生成事件描述 - 优化语义相似度"""
        etype = event.get("event_type")
        data = event.get("data", {})

        if etype == "player_speech":
            pid = data.get("player_id", "?")
            content = data.get("content", "")
            # 【关键优化】在描述中添加更多语义关键词，提高检索准确性
            return f"【玩家发言】{pid} 说：{content}。这是 {pid} 的观点和分析。"

        elif etype == "vote_result":
            votes = data.get("votes", {})
            res = data.get("result", "")
            vote_details = "、".join([f"{voter}投给{target}" for voter, target in votes.items()])
            return f"【投票结果】{res} 被放逐。投票详情：{vote_details}。"

        elif etype == "player_death":
            pid = data.get("player_id", "?")
            return f"【玩家死亡】{pid} 死亡，退出游戏。"

        elif etype == "phase_change":
            # 降低阶段变更描述的语义权重
            return f"【阶段变更】游戏进入新阶段。"
        
        elif etype == "speech_turn":
            # 降低发言顺序描述的语义权重
            return f"【发言顺序】轮到下一位玩家发言。"
        
        elif etype == "seer_check":
            target = data.get("target", "?")
            result = data.get("result", "?")
            return f"【预言家查验】查验了 {target}，结果是 {result}。这是重要的身份信息。"
        
        elif etype == "witch_save":
            target = data.get("target", "?")
            return f"【女巫救人】使用解药救了 {target}。{target} 是银水玩家。"
        
        elif etype == "witch_poison":
            target = data.get("target", "?")
            return f"【女巫毒人】使用毒药毒了 {target}。"

        else:
            return f"事件: {etype} | 数据: {data}"

    # ==================== Decision.py 需要的查询接口 ====================
    
    def get_recent_events(self, event_type: str = None, limit: int = 5) -> List[MemoryEntry]:
        """
        获取最近发生的事件 (内存层)
        ✅ decision.py 依赖此方法
        """
        if event_type and event_type in self.event_index:
            entries = self.event_index[event_type]
        else:
            entries = self.entries
        return entries[-limit:] if entries else []
    
    def search_by_tag(self, tag: str) -> List[MemoryEntry]:
        """
        按标签精确搜索 (内存层)
        ✅ decision.py 依赖此方法
        """
        results = []
        for entry in self.entries:
            if tag in entry.tags:
                results.append(entry)
        return results
    
    def get_summary(self, limit: int = 10) -> List[Dict]:
        """
        获取内存中最重要的几条记忆
        ✅ decision.py 依赖此方法
        """
        sorted_entries = sorted(self.entries, key=lambda x: x.importance, reverse=True)
        return [asdict(e) for e in sorted_entries[:limit]]

    def get_relevant_context(self, query: str, top_k: int = 5, day_filter: int = None, 
                            type_filter: str = None, max_chars: int = 2000, 
                            use_cot: bool = True) -> str:
        """
        增强版语义检索：CoT推理 + 图结构遍历 + 逻辑过滤
        ✅ decision.py 依赖此方法
        
        :param query: 用户的提问
        :param top_k: 尝试检索出的最大条数
        :param day_filter: 按天过滤 (可选)
        :param type_filter: 按类型过滤 (可选)
        :param max_chars: 返回文本的最大字符数，防止撑爆 LLM 上下文
        :param use_cot: 是否使用CoT推理链（默认开启）
        """
        if self.verbose:
            print(f"[正在回忆] 思考: {query} (过滤条件: Day={day_filter}, Type={type_filter}, CoT={use_cot})")

        if use_cot:
            return self._cot_retrieval(query, top_k, day_filter, type_filter, max_chars)
        else:
            return self._simple_retrieval(query, top_k, day_filter, type_filter, max_chars)

    def _simple_retrieval(self, query: str, top_k: int, day_filter: int, 
                         type_filter: str, max_chars: int) -> str:
        """原始的简单向量检索（保留兼容性）"""
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
            if self.verbose:
                print(f"语义检索失败: {e}")
            return f"【记忆检索失败】: 关于\"{query}\"没有找到匹配记录。"

        # 3. 提取结果
        if not results['documents'] or not results['documents'][0]:
            return f"【记忆检索】: 关于\"{query}\"没有找到匹配记录。"

        context_str = f"【关于\"{query}\"的相关记忆】:\n"
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

    def _cot_retrieval(self, query: str, top_k: int, day_filter: int, 
                      type_filter: str, max_chars: int) -> str:
        """
        CoT推理式检索：模拟人类的记忆回溯过程
        
        步骤：
        1. 分解查询意图（识别关键实体和关系）
        2. 初始向量检索（找到种子记忆）
        3. 图结构扩展（沿着关系链追溯相关记忆）
        4. 逻辑排序（按时间线和因果关系组织）
        5. 生成推理链文本
        """
        if self.verbose:
            print(f"[CoT检索] 开始推理式记忆检索...")
        
        # 步骤1: 分解查询意图
        query_intent = self._analyze_query_intent(query)
        if self.verbose:
            print(f"[CoT检索] 查询意图: {query_intent}")
        
        # 步骤2: 初始向量检索（种子记忆）
        seed_memories = self._vector_search(query, top_k * 2, day_filter, type_filter)
        if not seed_memories:
            return f"【记忆检索】: 关于\"{query}\"没有找到匹配记录。"
        
        if self.verbose:
            print(f"[CoT检索] 找到 {len(seed_memories)} 个种子记忆")
        
        # 步骤3: 图结构扩展（追溯相关记忆）
        expanded_memories = self._expand_via_graph(seed_memories, query_intent, max_expand=top_k * 3)
        if self.verbose:
            print(f"[CoT检索] 扩展后共 {len(expanded_memories)} 个相关记忆")
        
        # 步骤4: 逻辑排序和去重
        sorted_memories = self._sort_by_reasoning_chain(expanded_memories, query_intent)
        
        # 步骤5: 生成推理链文本
        context_str = self._build_reasoning_context(query, sorted_memories, query_intent, max_chars)
        
        return context_str

    # ==================== 辅助功能 ====================
    
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

    # ==================== 内部工具方法 (私有) ====================

    def _calculate_importance(self, event: Dict) -> float:
        """计算重要性"""
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        # 基础分 - 提高关键事件的重要性
        scores = {
            "phase_change": 0.1,        # 降低阶段变更的重要性
            "speech_turn": 0.05,        # 降低发言顺序的重要性
            "vote_result": 0.95,        # 提高投票结果
            "night_reveal": 1.0,        # 夜晚揭示（查验、救人等）
            "player_death": 1.0,        # 玩家死亡
            "player_speech": 0.85,      # 【关键】大幅提高玩家发言的重要性
            "seer_check": 1.0,          # 预言家查验
            "witch_save": 0.9,          # 女巫救人
            "witch_poison": 0.95,       # 女巫毒人
            "werewolf_discussion": 0.7, # 狼人讨论
            "werewolf_kill": 0.9,       # 狼人刀人
        }
        base = scores.get(event_type, 0.5)

        # 额外加分项
        res = data.get("result")
        if isinstance(res, dict) and res.get("exiled_player"):
            base += 0.05
        
        # 如果发言中包含关键词，提高重要性
        if event_type == "player_speech":
            content = data.get("content", "").lower()
            keywords = ["狼人", "预言家", "女巫", "查验", "银水", "金水", "查杀", "自爆", "跳", "身份"]
            for keyword in keywords:
                if keyword in content:
                    base = min(base + 0.05, 1.0)
                    break

        return min(base, 1.0)

    def _generate_tags(self, event: Dict) -> List[str]:
        """打标签"""
        tags = [event.get("event_type", "unknown")]
        data = event.get("data", {})

        if "player_id" in data:
            tags.append(f"player_{data['player_id']}")

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

    def _remove_least_important(self):
        """内存清理"""
        if not self.entries:
            return
        self.entries.sort(key=lambda x: x.importance)
        removed = self.entries.pop(0)
        if removed.event_type in self.event_index:
            try:
                self.event_index[removed.event_type].remove(removed)
            except ValueError:
                pass
    
    # ==================== 兼容性接口（支持各种访问模式） ====================
    
    def __len__(self):
        """支持 len(memory)"""
        return len(self.entries)
    
    def __getitem__(self, index):
        """支持 memory[0] 访问"""
        return self.entries[index]
    
    def __iter__(self):
        """支持 for entry in memory"""
        return iter(self.entries)
    
    def clear(self):
        """清空内存（保留ChromaDB）"""
        self.entries.clear()
        self.event_index.clear()
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """显式获取所有条目"""
        return self.entries.copy()
    
    def count_by_type(self, event_type: str) -> int:
        """统计某类事件数量"""
        return len(self.event_index.get(event_type, []))
    
    def get_latest(self) -> Optional[MemoryEntry]:
        """获取最新的一条记忆"""
        return self.entries[-1] if self.entries else None
    
    def filter_by_day(self, day: int) -> List[MemoryEntry]:
        """按天数筛选"""
        return [e for e in self.entries if e.day == day]
    
    def filter_by_importance(self, min_importance: float) -> List[MemoryEntry]:
        """按重要性筛选"""
        return [e for e in self.entries if e.importance >= min_importance]
    
    # ==================== CoT推理检索的核心方法 ====================
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        分析查询意图，提取关键信息
        
        返回：
        {
            'keywords': ['狼人', '预言家'],  # 关键词
            'players': ['player_1', 'player_2'],  # 涉及的玩家
            'event_types': ['player_speech', 'vote_result'],  # 关注的事件类型
            'temporal': 'recent' | 'all' | 'specific_day',  # 时间范围
            'reasoning_type': 'causal' | 'contradiction' | 'evidence'  # 推理类型
        }
        """
        intent = {
            'keywords': [],
            'players': [],
            'event_types': [],
            'temporal': 'all',
            'reasoning_type': 'evidence'
        }
        
        query_lower = query.lower()
        
        # 提取玩家信息
        import re
        player_patterns = [r'player[_\s]?(\d+)', r'(\d+)号', r'玩家(\d+)']
        for pattern in player_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                intent['players'].append(f"player_{match}")
        
        # 识别关键词和事件类型
        keyword_map = {
            '狼人': ('werewolf', ['player_speech', 'night_reveal']),
            '预言家': ('seer', ['player_speech', 'night_reveal']),
            '女巫': ('witch', ['player_speech', 'night_reveal']),
            '发言': ('speech', ['player_speech']),
            '投票': ('vote', ['vote_result']),
            '死': ('death', ['player_death']),
            '查验': ('check', ['night_reveal']),
            '可疑': ('suspicious', ['player_speech', 'vote_result']),
            '矛盾': ('contradiction', ['player_speech']),
        }
        
        for keyword, (tag, event_types) in keyword_map.items():
            if keyword in query:
                intent['keywords'].append(tag)
                intent['event_types'].extend(event_types)
        
        # 识别时间范围
        if '最近' in query or '刚才' in query or '昨晚' in query:
            intent['temporal'] = 'recent'
        elif '所有' in query or '全部' in query:
            intent['temporal'] = 'all'
        
        # 识别推理类型
        if '为什么' in query or '原因' in query or '导致' in query:
            intent['reasoning_type'] = 'causal'
        elif '矛盾' in query or '不一致' in query:
            intent['reasoning_type'] = 'contradiction'
        else:
            intent['reasoning_type'] = 'evidence'
        
        # 去重
        intent['event_types'] = list(set(intent['event_types']))
        intent['players'] = list(set(intent['players']))
        
        return intent
    
    def _vector_search(self, query: str, top_k: int, day_filter: int, 
                      type_filter: str) -> List[Tuple[MemoryEntry, float]]:
        """
        向量检索，返回记忆和相似度分数
        """
        where_filter = {}
        if day_filter is not None:
            where_filter["day"] = day_filter
        if type_filter is not None:
            where_filter["type"] = type_filter
        final_where = where_filter if where_filter else None
        
        query_vector = self.encoder.encode(query).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=final_where
            )
        except Exception as e:
            if self.verbose:
                print(f"向量检索失败: {e}")
            return []
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        # 构建结果列表
        memories_with_scores = []
        for i, mem_id in enumerate(results['ids'][0]):
            # 从内存中找到对应的MemoryEntry
            memory_entry = None
            for entry in self.entries:
                if entry.id == mem_id:
                    memory_entry = entry
                    break
            
            if memory_entry:
                # 距离转换为相似度分数（距离越小，相似度越高）
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = 1.0 / (1.0 + distance)
                memories_with_scores.append((memory_entry, similarity))
        
        return memories_with_scores
    
    def _expand_via_graph(self, seed_memories: List[Tuple[MemoryEntry, float]], 
                         query_intent: Dict, max_expand: int) -> List[Tuple[MemoryEntry, float, str]]:
        """
        通过图结构扩展记忆
        
        返回：[(MemoryEntry, score, reasoning_path)]
        """
        expanded = []
        visited = set()
        
        # 根据查询意图选择扩展策略
        reasoning_type = query_intent.get('reasoning_type', 'evidence')
        
        for memory, score in seed_memories:
            if memory.id in visited:
                continue
            
            visited.add(memory.id)
            expanded.append((memory, score, "直接匹配"))
            
            # 根据推理类型选择关系类型
            if reasoning_type == 'causal':
                relation_types = ['causal', 'temporal']
            elif reasoning_type == 'contradiction':
                relation_types = ['contradiction', 'reference']
            else:
                relation_types = ['reference', 'player_related', 'temporal']
            
            # 获取相邻记忆
            neighbors = self.memory_graph.get_neighbors(memory.id, relation_types)
            
            for neighbor_id in neighbors[:3]:  # 每个种子最多扩展3个邻居
                if neighbor_id in visited or len(expanded) >= max_expand:
                    break
                
                # 找到邻居的MemoryEntry
                neighbor_entry = None
                for entry in self.entries:
                    if entry.id == neighbor_id:
                        neighbor_entry = entry
                        break
                
                if neighbor_entry:
                    visited.add(neighbor_id)
                    # 邻居的分数衰减
                    neighbor_score = score * 0.7
                    reasoning_path = f"通过 {memory.event_type} 关联"
                    expanded.append((neighbor_entry, neighbor_score, reasoning_path))
        
        # 如果涉及特定玩家，添加该玩家的相关记忆
        for player_id in query_intent.get('players', []):
            player_memories = self.memory_graph.get_related_by_player(player_id)
            for mem_id in player_memories[:5]:  # 每个玩家最多5条
                if mem_id in visited or len(expanded) >= max_expand:
                    break
                
                for entry in self.entries:
                    if entry.id == mem_id:
                        visited.add(mem_id)
                        expanded.append((entry, 0.6, f"涉及玩家 {player_id}"))
                        break
        
        return expanded
    
    def _sort_by_reasoning_chain(self, memories: List[Tuple[MemoryEntry, float, str]], 
                                query_intent: Dict) -> List[Tuple[MemoryEntry, float, str]]:
        """
        按推理链排序：时间顺序 + 因果关系 + 重要性
        """
        # 先按天数和时间戳排序（时间线）
        sorted_memories = sorted(memories, key=lambda x: (x[0].day, x[0].timestamp))
        
        # 再按分数和重要性调整
        # 对于因果推理，保持时间顺序；对于证据收集，按相关性排序
        if query_intent.get('reasoning_type') == 'causal':
            # 因果推理：保持时间顺序
            return sorted_memories
        else:
            # 证据收集：按分数排序，但同一天的保持时间顺序
            result = []
            by_day = defaultdict(list)
            for item in sorted_memories:
                by_day[item[0].day].append(item)
            
            # 按天数倒序（最近的在前），但每天内部保持时间顺序
            for day in sorted(by_day.keys(), reverse=True):
                day_memories = sorted(by_day[day], key=lambda x: x[1], reverse=True)
                result.extend(day_memories)
            
            return result
    
    def _build_reasoning_context(self, query: str, 
                                 sorted_memories: List[Tuple[MemoryEntry, float, str]], 
                                 query_intent: Dict, max_chars: int) -> str:
        """
        构建推理链文本：以CoT的方式组织记忆
        """
        reasoning_type = query_intent.get('reasoning_type', 'evidence')
        
        # 构建标题
        if reasoning_type == 'causal':
            context_str = f"【关于\"{query}\"的因果推理链】:\n\n"
        elif reasoning_type == 'contradiction':
            context_str = f"【关于\"{query}\"的矛盾分析】:\n\n"
        else:
            context_str = f"【关于\"{query}\"的证据链】:\n\n"
        
        current_len = len(context_str)
        current_day = None
        step = 1
        
        for memory, score, reasoning_path in sorted_memories:
            # 添加天数分隔
            if current_day != memory.day:
                day_header = f"\n--- 第 {memory.day} 天 ---\n"
                if current_len + len(day_header) > max_chars:
                    context_str += "\n...(更多记忆已省略)...\n"
                    break
                context_str += day_header
                current_len += len(day_header)
                current_day = memory.day
            
            # 构建记忆条目（CoT格式）
            entry_text = f"{step}. [{memory.event_type}] {memory.text}\n"
            entry_text += f"   → 推理路径: {reasoning_path} | 相关度: {score:.2f} | 重要性: {memory.importance:.2f}\n"
            
            # 添加逻辑连接词
            if step > 1 and reasoning_type == 'causal':
                entry_text = f"   ↓ 因此...\n{entry_text}"
            
            if current_len + len(entry_text) > max_chars:
                context_str += "\n...(更多记忆已省略)...\n"
                break
            
            context_str += entry_text
            current_len += len(entry_text)
            step += 1
        
        # 添加推理总结
        if len(sorted_memories) > 0:
            summary = f"\n【推理总结】: 共检索到 {len(sorted_memories)} 条相关记忆，"
            if reasoning_type == 'causal':
                summary += "按时间因果顺序排列。"
            elif reasoning_type == 'contradiction':
                summary += "重点关注矛盾之处。"
            else:
                summary += "按相关性排序。"
            
            if current_len + len(summary) <= max_chars:
                context_str += summary
        
        return context_str
    
    def _update_memory_graph(self, new_entry: MemoryEntry, event: Dict):
        """
        更新记忆图：为新记忆建立关系
        """
        # 添加节点
        players = self._extract_players_from_event(event)
        self.memory_graph.add_node(new_entry.id, {
            'event_type': new_entry.event_type,
            'day': new_entry.day,
            'phase': new_entry.phase,
            'players': players,
            'importance': new_entry.importance
        })
        
        # 建立关系
        # 1. 时间关系：与前一条记忆建立temporal关系
        if len(self.entries) > 1:
            prev_entry = self.entries[-2]  # 倒数第二个（因为当前已经添加到entries）
            self.memory_graph.add_edge(prev_entry.id, new_entry.id, 'temporal')
        
        # 2. 玩家关联：与涉及相同玩家的记忆建立关系
        for player in players:
            related_memories = self.memory_graph.get_related_by_player(player)
            for related_id in related_memories[-5:]:  # 只关联最近5条
                if related_id != new_entry.id:
                    self.memory_graph.add_edge(related_id, new_entry.id, 'player_related')
        
        # 3. 因果关系：根据事件类型建立因果链
        if new_entry.event_type == 'vote_result':
            # 投票结果与之前的发言有因果关系
            for entry in self.entries[-10:]:
                if entry.event_type == 'player_speech':
                    self.memory_graph.add_edge(entry.id, new_entry.id, 'causal')
        
        elif new_entry.event_type == 'player_death':
            # 死亡与投票结果或夜晚行动有因果关系
            for entry in self.entries[-5:]:
                if entry.event_type in ['vote_result', 'night_reveal']:
                    self.memory_graph.add_edge(entry.id, new_entry.id, 'causal')
        
        # 4. 引用关系：如果发言中提到其他玩家
        if new_entry.event_type == 'player_speech':
            content = new_entry.text.lower()
            for other_player in ['player_1', 'player_2', 'player_3', 'player_4', 
                                'player_5', 'player_6', 'player_7', 'player_8']:
                if other_player in content:
                    # 找到该玩家的最近发言
                    for entry in reversed(self.entries[:-1]):
                        if entry.event_type == 'player_speech' and other_player in entry.text:
                            self.memory_graph.add_edge(new_entry.id, entry.id, 'reference')
                            break
    
    def _extract_players_from_event(self, event: Dict) -> List[str]:
        """从事件中提取涉及的玩家ID"""
        players = []
        data = event.get('data', {})
        
        # 1. 直接的player_id字段
        if 'player_id' in data:
            pid = data['player_id']
            if isinstance(pid, str) and pid.startswith('player_'):
                players.append(pid)
        
        # 2. 投票相关
        if 'votes' in data and isinstance(data['votes'], dict):
            for voter, target in data['votes'].items():
                if isinstance(voter, str) and voter.startswith('player_'):
                    players.append(voter)
                if isinstance(target, str) and target.startswith('player_'):
                    players.append(target)
        
        # 3. 结果字段
        if 'result' in data:
            result = data['result']
            if isinstance(result, str) and result.startswith('player_'):
                players.append(result)
        
        # 4. 目标字段（夜晚行动）
        if 'target' in data:
            target = data['target']
            if isinstance(target, str) and target.startswith('player_'):
                players.append(target)
        
        # 5. 发言者字段（狼人讨论）
        if 'speaker' in data:
            speaker = data['speaker']
            if isinstance(speaker, str) and speaker.startswith('player_'):
                players.append(speaker)
        
        # 去重并返回
        return list(set(players))
    
    def _extract_mentioned_players(self, text: str) -> List[str]:
        """从文本中提取提到的玩家"""
        import re
        players = []
        
        patterns = [r'player[_\s]?(\d+)', r'(\d+)号', r'玩家(\d+)']
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                players.append(f"player_{match}")
        
        return list(set(players))

    def retrieve_recent_memories(self, query: str, n_results: int = 10,
                                 day_filter: int = None, recency_weight: float = 0.3,
                                 exclude_system_events: bool = True) -> List[str]:
        """
        增强版近期记忆检索：结合语义相似度和时间因素

        Args:
            query: 查询文本
            n_results: 返回结果数量
            day_filter: 天数过滤
            recency_weight: 时间权重 (0-1)，越大表示越重视最近发生的事件
        """
        if not self.collection:
            return []

        try:
            # 1. 向量相似度检索
            query_vector = self.encoder.encode(query).tolist()

            # 增加检索数量，留出筛选空间
            vector_results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results * 2,  # 检索更多，以便筛选
                where={"day": day_filter} if day_filter is not None else None
            )

            if not vector_results or not vector_results['documents']:
                return []

            # 2. 综合排序：相似度 + 时间权重 + 重要性
            scored_items = []
            for i, (doc, metadata) in enumerate(zip(vector_results['documents'][0],
                                                    vector_results['metadatas'][0])):

                # 【关键修改】过滤发言顺序
                if exclude_system_events and self._is_system_message(doc):
                    continue  # 跳过发言顺序

                # 相似度分数（距离越小，相似度越高）
                distance = vector_results['distances'][0][i] if 'distances' in vector_results else 0
                similarity_score = 1.0 / (1.0 + distance)

                # 时间因素：越近的事件得分越高
                day = metadata.get('day', 0)
                current_day = day_filter if day_filter else self._get_current_day()
                recency_score = 1.0 / (1.0 + abs(current_day - day))

                # 重要性分数
                importance_score = metadata.get('importance', 0.5)

                # 综合得分
                combined_score = (
                        similarity_score * (1 - recency_weight) +
                        recency_score * recency_weight +
                        importance_score * 0.2
                )

                scored_items.append({
                    'text': doc,
                    'score': combined_score,
                    'day': day,
                    'similarity': similarity_score
                })

            # 3. 按综合得分排序，返回前n_results个
            scored_items.sort(key=lambda x: x['score'], reverse=True)

            # 4. 去重：相似的文本只保留一个
            unique_texts = []
            seen_texts = set()
            for item in scored_items:
                # 简单的文本去重（前50个字符相同视为重复）
                text_start = item['text'][:50]
                if text_start not in seen_texts:
                    seen_texts.add(text_start)
                    unique_texts.append(item['text'])
                    if len(unique_texts) >= n_results:
                        break

            # 5. 如果向量检索结果不足，补充按时间排序的最新记忆
            if len(unique_texts) < n_results and self.entries:
                recent_entries = sorted(self.entries, key=lambda x: (x.day, x.timestamp), reverse=True)
                for entry in recent_entries:
                    if entry.text not in unique_texts and entry.text not in seen_texts:
                        unique_texts.append(entry.text)
                        seen_texts.add(entry.text)
                        if len(unique_texts) >= n_results:
                            break

            return unique_texts[:n_results]

        except Exception as e:
            if getattr(self, 'verbose', False):
                print(f"检索记忆失败: {e}")
            return []

    def _is_system_message(self, text: str) -> bool:
        """
        判断是否为需要过滤的发言顺序信息
        """
        if not text:
            return False

        # 检查是否是发言顺序相关
        system_patterns = [
            "发言顺序",
        ]

        for pattern in system_patterns:
            if pattern in text:
                return True

        # 额外检查：如果文本很短且包含player_和数字，可能是发言顺序
        if len(text) < 80 and any(f"player_{i}" in text for i in range(1, 9)):
            # 但如果包含玩家实际发言内容，就不是系统消息
            if "说：" in text or "发言说：" in text:
                return False
            return True

        return False

    def add_memory(self, content: Dict, text: str, event_type: str, day: int, phase: str, importance: float = 0.5):
        """
        [新增] 兼容接口：手动添加记忆
        """
        # 构造 event 字典
        event = {
            "event_type": event_type,
            "data": content,
            "phase": phase,
            "timestamp": datetime.now().isoformat()
        }
        # 确保 day 存在于 data 中
        if "day" not in event["data"]:
            event["data"]["day"] = day
            
        # 调用核心方法
        self.add_event(event, text_description=text)
        
        if self.entries:
            self.entries[-1].importance = importance