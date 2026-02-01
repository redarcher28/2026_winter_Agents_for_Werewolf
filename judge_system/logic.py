import asyncio
import logging
import uuid
import time
import json
import os
import aiofiles
from typing import Dict, Any, List, Optional, Callable
from collections import Counter
from pathlib import Path

# ==========================================
# 0. 导入依赖
# ==========================================
# 假设 input_file_8.py 被重命名为 agent_framework.py
try:
    from agent_framework import (
        BaseWerewolfAgent, AgentConfig, LLMConfig, Role, GamePhase,
        AgentState
    )
except ImportError:
    # 如果找不到文件，说明环境未配置好，这里仅作提示
    print("Error: agent_framework.py (input_file_8) not found.")
    exit(1)

# 假设 input_file_1.py (DataStorageService) 和相关接口可用
# 这里为了代码独立运行，我们保留必要的 Local/Real DataManager 桥接逻辑
# 实际项目中应 import 真实的 DataStorageService

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [Server] %(message)s')
logger = logging.getLogger("WerewolfServer")


# ==========================================
# 1. 基础接口定义 (保持兼容)
# ==========================================
class IDataManager:
    async def save_game_state(self, game_id: str, state: Dict[str, Any]) -> bool: pass

    async def save_event(self, game_id: str, event_data: Dict[str, Any]) -> bool: pass

    async def save_vote(self, game_id: str, vote_data: Dict[str, Any]) -> bool: pass

    async def save_role_data(self, game_id: str, role: str, data: Dict[str, Any]) -> bool: pass

    async def save_player_status(self, game_id: str, player_data: Dict[str, Any]) -> bool: pass


class EventBus:
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self._listeners: self._listeners[event_type] = []
        self._listeners[event_type].append(handler)

    async def publish(self, event_type: str, payload: Any):
        if event_type in self._listeners:
            await asyncio.gather(*(h(payload) for h in self._listeners[event_type]))


bus = EventBus()


# ==========================================
# 2. 状态管理器 (GSM)
# ==========================================
class GameStateManager:
    def __init__(self, dm: IDataManager, game_id: str):
        self.dm = dm
        self.game_id = game_id
        self.lock = asyncio.Lock()

        # 初始化基础状态
        self.state = {
            "game_id": game_id,
            "phase": "night",
            "day_number": 1,
            "timestamp": time.time(),
            "game_over": False,
            "winner": None,
            "alive_players": [str(i) for i in range(1, 13)],
            "dead_players": [],
            "last_night_actions": {},
            "vote_results": {},
            "players": {},
            "private_data": {
                "witch": {"use_heal": False, "use_poison": False, "heal_target": None, "poison_target": None},
                "seer": {"inspections": []},
                "werewolf": {"team_members": [str(i) for i in range(1, 5)], "kill_targets": []}
            },
            "current_turn_actions": {"wolf_kill": None, "witch_action": None, "votes": {}}
        }

        # 初始化玩家 (1-4狼, 5女巫, 6预言家, 7-8神, 9-12民)
        self._init_players()

    def _init_players(self):
        for i in range(1, 13):
            pid = str(i)
            role = "villager"
            if 1 <= i <= 4:
                role = "werewolf"
            elif i == 5:
                role = "witch"
            elif i == 6:
                role = "seer"
            elif i <= 8:
                role = "god"  # 猎人/白痴等，暂统称神

            self.state["players"][pid] = {
                "id": pid, "name": f"Player_{pid}", "role": role, "status": "alive", "is_ai": True
            }

    async def init_storage(self):
        """初始化存储，将初始状态写入文件，供Agent读取"""
        # 保存全局状态
        await self.dm.save_game_state(self.game_id, self.state)
        # 保存玩家私有信息
        for pid, p in self.state["players"].items():
            await self.dm.save_player_status(self.game_id, p)
        # 保存角色私有信息
        await self.dm.save_role_data(self.game_id, "werewolf", self.state["private_data"]["werewolf"])
        await self.dm.save_role_data(self.game_id, "witch", self.state["private_data"]["witch"])
        await self.dm.save_role_data(self.game_id, "seer", self.state["private_data"]["seer"])

    async def commit_change(self, delta: Dict[str, Any]):
        """提交状态变更并持久化"""
        async with self.lock:
            # 简单处理 update_type 逻辑 (与原 logic.py 类似，略微简化)
            if "update_type" in delta:
                u_type = delta.pop("update_type")
                if u_type == "player_update":
                    pid = delta["player_id"]
                    self.state["players"][pid].update(delta["data"])
                    if delta["data"].get("status") == "dead":
                        if pid in self.state["alive_players"]: self.state["alive_players"].remove(pid)
                        if pid not in self.state["dead_players"]: self.state["dead_players"].append(pid)
                    await self.dm.save_player_status(self.game_id, self.state["players"][pid])

                elif u_type == "wolf_kill":
                    self.state["current_turn_actions"]["wolf_kill"] = delta["target_id"]
                    self.state["private_data"]["werewolf"]["kill_targets"].append(delta["target_id"])
                    await self.dm.save_role_data(self.game_id, "werewolf", self.state["private_data"]["werewolf"])

                elif u_type == "witch_action":
                    act = delta["data"]
                    w_data = self.state["private_data"]["witch"]
                    if act.get("use_heal"): w_data.update({"use_heal": True, "heal_target": act.get("heal_target")})
                    if act.get("use_poison"): w_data.update(
                        {"use_poison": True, "poison_target": act.get("poison_target")})
                    self.state["current_turn_actions"]["witch_action"] = act
                    await self.dm.save_role_data(self.game_id, "witch", w_data)

                elif u_type == "seer_check":
                    self.state["private_data"]["seer"]["inspections"].append(delta["target_id"])
                    await self.dm.save_role_data(self.game_id, "seer", self.state["private_data"]["seer"])

                elif u_type == "vote":
                    self.state["current_turn_actions"]["votes"][delta["voter"]] = delta["target"]
            else:
                self.state.update(delta)

            self.state["timestamp"] = time.time()
            await self.dm.save_game_state(self.game_id, self.state)
            return self.state


# ==========================================
# 3. 动作监听器 (服务端监听 Agent 文件)
# ==========================================
class AgentActionListener:
    """监听 agent_actions.json，获取 Agent 的行动"""

    def __init__(self, game_id: str, base_dir: str = "./game_data"):
        self.action_file = Path(base_dir) / f"game_{game_id}" / "agent_actions.json"
        self.processed_timestamps = set()

    async def wait_for_action(self, agent_id: str, expected_action_types: List[str], timeout: int = 30) -> Optional[
        Dict]:
        """
        轮询等待指定 Agent 的特定类型动作
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.action_file.exists():
                try:
                    async with aiofiles.open(self.action_file, 'r') as f:
                        content = await f.read()
                        if content.strip():
                            data = json.loads(content)
                            actions = data.get("actions", [])

                            # 倒序查找该 Agent 的最新行动
                            for action in reversed(actions):
                                act_ts = action.get("timestamp")
                                act_id = action.get("agent_id")
                                act_data = action.get("data", {})
                                act_type = action.get("action")  # submit_night_action, submit_vote

                                # 检查是否已处理
                                if act_ts in self.processed_timestamps:
                                    continue

                                if act_id == agent_id:
                                    # 检查具体类型 (Agent框架包装了一层 submit_*)
                                    real_type = act_data.get("action_type") or act_type

                                    # 宽松匹配：如果是 submit_night_action，检查 data 里的 action_type 是否匹配期望
                                    # 或者直接匹配顶层 action
                                    is_match = False
                                    if act_type in expected_action_types: is_match = True
                                    if real_type in expected_action_types: is_match = True

                                    if is_match:
                                        self.processed_timestamps.add(act_ts)
                                        return act_data
                except Exception as e:
                    logger.warning(f"Error reading action file: {e}")

            await asyncio.sleep(0.5)
        return None


# ==========================================
# 4. 游戏循环控制器 (GLC)
# ==========================================
class GameLoopController:
    def __init__(self, gsm: GameStateManager, action_listener: AgentActionListener):
        self.gsm = gsm
        self.listener = action_listener

    async def ask_agent_action(self, role: str, player_id: str, action_types: List[str], context: Dict = None) -> Dict:
        """
        向 Agent 发起行动请求（通过等待文件系统响应）
        """
        logger.info(f"⏳ 等待 Player_{player_id} ({role}) 行动: {action_types}...")

        # 真正等待 Agent 写入文件
        # 注意：Agent 框架中，action_type 可能是 'kill', 'check', 'vote' 等
        # 而 submit 的 action 名可能是 'submit_night_action', 'submit_vote'
        # 我们在 listener 中做了兼容处理

        # 映射 Agent 框架的 action type
        expected = action_types
        if "KILL" in action_types: expected.extend(["kill", "submit_night_action"])
        if "check" in action_types: expected.extend(["submit_night_action"])
        if "heal" in action_types or "poison" in action_types: expected.extend(
            ["save", "poison", "submit_night_action"])
        if "VOTE" in action_types: expected.extend(["vote", "submit_vote"])

        result = await self.listener.wait_for_action(player_id, expected, timeout=15)

        if result:
            logger.info(f"✅ 收到 Player_{player_id} 行动: {result}")
            # 标准化返回格式给 handle_action 使用
            return {
                "user_id": player_id,
                "action_type": result.get("action_type", "unknown"),
                "target_id": result.get("target_id"),
                "raw": result
            }
        else:
            logger.warning(f"❌ Player_{player_id} 行动超时 (SKIP)")
            return {"user_id": player_id, "action_type": "skip"}

    async def handle_agent_action(self, cmd: Dict):
        """执行接收到的指令"""
        act_type = cmd.get("action_type")

        # 简单校验
        if act_type == "skip": return

        # 提交到 GSM
        if act_type == "kill":
            await self.gsm.commit_change({"update_type": "wolf_kill", "target_id": str(cmd["target_id"])})
        elif act_type == "save":  # 女巫解药
            await self.gsm.commit_change(
                {"update_type": "witch_action", "data": {"use_heal": True, "heal_target": str(cmd["target_id"])}})
        elif act_type == "poison":  # 女巫毒药
            await self.gsm.commit_change(
                {"update_type": "witch_action", "data": {"use_poison": True, "poison_target": str(cmd["target_id"])}})
        elif act_type == "check":
            target = str(cmd["target_id"])
            await self.gsm.commit_change({"update_type": "seer_check", "target_id": target})
            # 这里需要给预言家私发结果，但在文件系统模式下，预言家会通过 query_role_info 读取 private/seer.json
            # 我们只需要确保 private 数据更新了即可 (GSM commit_change 已做)
        elif act_type == "vote":
            await self.gsm.commit_change(
                {"update_type": "vote", "voter": str(cmd["user_id"]), "target": str(cmd["target_id"])})

    async def run_game_loop(self):
        """游戏主循环"""
        await self.gsm.dm.save_event(self.gsm.game_id, {"event_type": "game_start", "description": "游戏开始"})

        while not self.gsm.state["game_over"]:
            day = self.gsm.state["day_number"]
            logger.info(f"\n======== 第 {day} 天 ========")

            # 1. 夜晚阶段
            await self.gsm.commit_change({"phase": "night"})
            await self.gsm.dm.save_event(self.gsm.game_id, {
                "event_type": "phase_change",
                "data": {"old_phase": "day", "new_phase": "night", "day": day}
            })

            # 1.1 狼人行动
            wolves = [p for p in self.gsm.state["players"].values() if
                      p["status"] == "alive" and p["role"] == "werewolf"]
            if wolves:
                # 简化：只等待第一个活着的狼人代表行动
                leader = wolves[0]
                action = await self.ask_agent_action("werewolf", leader["id"], ["KILL"])
                await self.handle_agent_action(action)

            # 1.2 女巫行动
            witches = [p for p in self.gsm.state["players"].values() if p["status"] == "alive" and p["role"] == "witch"]
            if witches:
                # 确保狼刀信息已更新，Agent 会通过 polling werewolf_kill 获取（需 GSM 支持 public state 更新）
                # 注意：实际 Agent 框架中，女巫可能需要通过 private query 获取今晚死讯(如果是 mock 环境)
                # 这里我们假设 Agent 足够聪明，或者我们不做特殊处理，Agent 策略决定是否盲毒
                action = await self.ask_agent_action("witch", witches[0]["id"], ["heal", "poison"])
                await self.handle_agent_action(action)

            # 1.3 预言家行动
            seers = [p for p in self.gsm.state["players"].values() if p["status"] == "alive" and p["role"] == "seer"]
            if seers:
                action = await self.ask_agent_action("seer", seers[0]["id"], ["check"])
                await self.handle_agent_action(action)

            # 2. 黎明结算
            await self.gsm.commit_change({"phase": "day"})
            await self.gsm.dm.save_event(self.gsm.game_id, {
                "event_type": "phase_change",
                "data": {"old_phase": "night", "new_phase": "day", "day": day}
            })

            await self._settle_night()
            if self.gsm.state["game_over"]: break

            # 3. 白天投票
            # 简化：并发等待所有活人投票
            alive = [p for p in self.gsm.state["players"].values() if p["status"] == "alive"]
            logger.info("🗳️ 开始投票环节...")

            # 并发请求
            tasks = []
            for p in alive:
                tasks.append(self.ask_agent_action(p["role"], p["id"], ["VOTE"]))

            results = await asyncio.gather(*tasks)
            for res in results:
                await self.handle_agent_action(res)

            await self._settle_votes()
            if self.gsm.state["game_over"]: break

            await self.gsm.commit_change({"day_number": day + 1})
            await asyncio.sleep(2)  # 稍作休息

    async def _settle_night(self):
        actions = self.gsm.state["current_turn_actions"]
        dead = []

        # 狼刀
        target = actions.get("wolf_kill")

        # 女巫解药
        witch_act = actions.get("witch_action") or {}
        if target and witch_act.get("use_heal") and witch_act.get("heal_target") == target:
            target = None  # 救活

        if target: dead.append(target)
        if witch_act.get("use_poison"): dead.append(witch_act.get("poison_target"))

        unique_dead = list(set([d for d in dead if d]))

        # 公告
        desc = f"昨晚死亡: {unique_dead}" if unique_dead else "昨晚是平安夜"
        await self.gsm.dm.save_event(self.gsm.game_id, {
            "event_type": "night_reveal",
            "data": {"dead_players": unique_dead, "announcement": desc}
        })

        for pid in unique_dead:
            await self.gsm.commit_change({"update_type": "player_update", "player_id": pid, "data": {"status": "dead"}})
            # 触发 Agent 的 player_death 事件
            await self.gsm.dm.save_event(self.gsm.game_id, {
                "event_type": "player_death", "data": {"player_id": pid}
            })

        await self.gsm.commit_change({"current_turn_actions": {"wolf_kill": None, "witch_action": None, "votes": {}}})
        await self._check_victory()

    async def _settle_votes(self):
        votes = self.gsm.state["current_turn_actions"]["votes"]
        if not votes: return

        # 统计
        counts = Counter(votes.values())
        if not counts: return

        max_v = max(counts.values())
        candidates = [k for k, v in counts.items() if v == max_v]
        out_pid = candidates[0]  # 简化：平票出第一个

        await self.gsm.dm.save_vote(self.gsm.game_id, {
            "day_number": self.gsm.state["day_number"],
            "votes": votes,
            "result": out_pid
        })

        await self.gsm.dm.save_event(self.gsm.game_id, {
            "event_type": "vote_result",
            "data": {"result": out_pid, "votes": votes}
        })

        await self.gsm.commit_change({"update_type": "player_update", "player_id": out_pid, "data": {"status": "dead"}})
        await self.gsm.dm.save_event(self.gsm.game_id, {
            "event_type": "player_death", "data": {"player_id": out_pid}
        })

        await self.gsm.commit_change({"current_turn_actions": {"votes": {}}})
        await self._check_victory()

    async def _check_victory(self):
        p = self.gsm.state["players"]
        wolves = [pid for pid, v in p.items() if v["status"] == "alive" and v["role"] == "werewolf"]
        good = [pid for pid, v in p.items() if v["status"] == "alive" and v["role"] != "werewolf"]

        winner = None
        if not wolves:
            winner = "GOOD_SIDE"
        elif len(wolves) >= len(good):
            winner = "WOLF_SIDE"  # 简化胜利条件

        if winner:
            await self.gsm.commit_change({"game_over": True, "winner": winner})
            await self.gsm.dm.save_event(self.gsm.game_id, {"event_type": "game_end", "data": {"winner": winner}})
            logger.info(f"🏆 游戏结束: {winner} 获胜!")


# ==========================================
# 5. 具体化的 Agent 实现
# ==========================================
class GamePlayerAgent(BaseWerewolfAgent):
    """
    继承自 Agent 框架的具体实现，连接游戏逻辑
    """

    async def on_game_start(self):
        self.logger.info(f"[{self.my_role.value}] 游戏开始，我是 {self.my_id}")

    async def on_night_action(self, phase: GamePhase):
        # 简化策略：使用 LLM Client 或 随机 fallback
        try:
            context = await self.decision_engine._build_decision_context()

            if phase == GamePhase.WEREWOLF_NIGHT and self.my_role == Role.WEREWOLF:
                # 狼人行动
                decision = await self.llm_client.decide_wolf_kill(context)
                target = decision.data.get("target_id")
                # Fallback
                if not target:
                    target = self._random_target(exclude_roles=[Role.WEREWOLF])

                await self.submit_action("submit_night_action", {
                    "action_type": "kill",
                    "target_id": target
                })

            elif phase == GamePhase.SEER_NIGHT and self.my_role == Role.SEER:
                # 预言家行动
                decision = await self.llm_client.decide_seer_check(context)
                target = decision.data.get("target_id")
                if not target: target = self._random_target(exclude_self=True)

                await self.submit_action("submit_night_action", {
                    "action_type": "check",
                    "target_id": target
                })

            elif phase == GamePhase.WITCH_NIGHT and self.my_role == Role.WITCH:
                # 女巫行动
                decision = await self.llm_client.decide_witch_action(context)
                act_type = decision.data.get("action_type", "no_potion")
                target = decision.data.get("target_id")

                if act_type == "save" and target:
                    await self.submit_action("submit_night_action", {"action_type": "save", "target_id": target})
                elif act_type == "poison" and target:
                    await self.submit_action("submit_night_action", {"action_type": "poison", "target_id": target})
                else:
                    await self.submit_action("submit_night_action", {"action_type": "skip"})

        except Exception as e:
            self.logger.error(f"决策错误: {e}")
            # 保底操作：跳过
            await self.submit_action("submit_night_action", {"action_type": "skip"})

    async def on_daytime_discussion(self):
        pass  # 暂不实现自由发言，简化流程

    async def on_voting_phase(self):
        context = await self.decision_engine._build_decision_context()
        # 调用各角色投票逻辑
        target = await self.decision_engine.decide_vote_target()

        if not target:
            target = self._random_target(exclude_self=True)

        await self.submit_action("submit_vote", {
            "action_type": "vote",
            "target_id": target
        })

    # --- 辅助方法 ---
    def _random_target(self, exclude_self=False, exclude_roles=[]):
        import random
        candidates = []
        for pid, info in self.known_players.items():
            if not info.is_alive: continue
            if exclude_self and pid == self.my_id: continue
            # Note: agent usually doesn't know others' roles, but for simulation simple fallback:
            candidates.append(pid)
        return random.choice(candidates) if candidates else None

    # 重写基类抽象方法以避免报错
    async def analyze_speech(self, player_id: str, content: str):
        return {}

    async def formulate_strategy(self):
        return {}


# ==========================================
# 6. Real Data Manager (文件存储适配器)
# ==========================================
class RealDataManager(IDataManager):
    """
    实际的文件存储实现，与 agent_framework 的路径约定一致
    """

    def __init__(self, base_dir="./game_data"):
        self.base_dir = Path(base_dir)

    def _get_game_dir(self, game_id):
        d = self.base_dir / f"game_{game_id}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "logs").mkdir(exist_ok=True)
        (d / "private" / "roles").mkdir(parents=True, exist_ok=True)
        (d / "private" / "players").mkdir(parents=True, exist_ok=True)
        return d

    async def save_game_state(self, game_id, state):
        p = self._get_game_dir(game_id) / "logs" / "game_state.log"
        async with aiofiles.open(p, "a") as f:
            await f.write(json.dumps(state, ensure_ascii=False) + "\n")
        return True

    async def save_event(self, game_id, event_data):
        p = self._get_game_dir(game_id) / "logs" / "game_events.log"
        # 补全 ID 和 时间戳
        if "event_id" not in event_data: event_data["event_id"] = uuid.uuid4().hex
        if "timestamp" not in event_data: event_data["timestamp"] = datetime.now().isoformat()
        async with aiofiles.open(p, "a") as f:
            await f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
        return True

    async def save_vote(self, game_id, vote_data):
        p = self._get_game_dir(game_id) / "logs" / "vote_result.log"
        if "timestamp" not in vote_data: vote_data["timestamp"] = datetime.now().isoformat()
        async with aiofiles.open(p, "a") as f:
            await f.write(json.dumps(vote_data, ensure_ascii=False) + "\n")
        return True

    async def save_role_data(self, game_id, role, data):
        p = self._get_game_dir(game_id) / "private" / "roles" / f"{role}.json"
        # 包装一下以符合 Agent 的 expect
        wrapped = {"role": role, **data}
        async with aiofiles.open(p, "w") as f:
            await f.write(json.dumps(wrapped, ensure_ascii=False, indent=2))
        return True

    async def save_player_status(self, game_id, player_data):
        pid = player_data["id"]
        p = self._get_game_dir(game_id) / "private" / "players" / f"{pid}.json"
        async with aiofiles.open(p, "w") as f:
            await f.write(json.dumps(player_data, ensure_ascii=False, indent=2))
        return True


# ==========================================
# 7. 主程序：启动服务端和 12 个 Agent
# ==========================================
async def main():
    GAME_ID = "ROOM_REAL_AGENT_001"

    # 1. 初始化服务端组件
    dm = RealDataManager()
    gsm = GameStateManager(dm, GAME_ID)
    action_listener = AgentActionListener(GAME_ID)
    glc = GameLoopController(gsm, action_listener)

    # 初始化存储结构
    await gsm.init_storage()

    # 2. 启动 12 个 Agent (作为并发 Task)
    logger.info("🚀 正在启动 12 个 AI Agent...")
    agent_tasks = []
    agents = []

    # 使用 Mock LLM Config 避免 API Key 报错
    mock_llm = LLMConfig(provider="openai", api_key="sk-mock", model="gpt-3.5-turbo")

    for pid in range(1, 13):
        agent_id = str(pid)
        config = AgentConfig(
            agent_id=agent_id,
            game_id=GAME_ID,
            max_memory_entries=50,
            db_path=f"./memory_db/{agent_id}",  # 独立记忆库
            llm=mock_llm
        )

        agent = GamePlayerAgent(config)
        agents.append(agent)
        # 启动 agent 生命周期
        task = asyncio.create_task(agent.start())
        agent_tasks.append(task)

    # 等待 Agent 连接就绪
    await asyncio.sleep(2)

    try:
        # 3. 启动游戏主循环
        logger.info("🎮 游戏主循环启动...")
        await glc.run_game_loop()

    except Exception as e:
        logger.error(f"Game loop error: {e}")
    finally:
        # 清理：停止所有 Agent
        logger.info("🛑 正在停止 Agent...")
        for agent in agents:
            await agent.stop()

        # 等待任务结束
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        logger.info("✅ 系统退出")


if __name__ == "__main__":
    from datetime import datetime

    asyncio.run(main())