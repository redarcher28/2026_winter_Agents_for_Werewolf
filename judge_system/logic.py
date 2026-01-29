import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Callable

# ==========================================
# 1. 基础配置 (保持原样)
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("WerewolfBLL")


class IDataManager(ABC):
    @abstractmethod
    async def save_game_state(self, game_id: str, state: Dict[str, Any]) -> bool: pass

    @abstractmethod
    async def load_game_state(self, game_id: str) -> Optional[Dict[str, Any]]: pass


class EventBus:
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)

    async def publish(self, event_type: str, payload: Any):
        if event_type in self._listeners:
            await asyncio.gather(*(h(payload) for h in self._listeners[event_type]))


bus = EventBus()


# ==========================================
# 4. 状态管理器 (GSM) - 扩展 12 人局神职
# ==========================================
class GameStateManager:
    def __init__(self, dm: IDataManager, game_id: str = "ROOM_888"):
        self.dm = dm
        self.game_id = game_id
        self.lock = asyncio.Lock()
        self.state = {
            "phase": "NIGHT",
            "day_count": 1,
            "players": {
                # 1-4狼，5预言家(SEER)，6女巫(WITCH)，7-8猎人白痴(GOD)，9-12民
                **{str(i): {"role": "WEREWOLF", "alive": True} for i in range(1, 5)},
                "5": {"role": "SEER", "alive": True},
                "6": {"role": "WITCH", "alive": True, "has_heal": True, "has_poison": True},
                "7": {"role": "GOD", "alive": True},
                "8": {"role": "GOD", "alive": True},
                **{str(i): {"role": "VILLAGER", "alive": True} for i in range(9, 13)},
            },
            "current_votes": {},
            "night_kill_target": None,
            "night_heal_target": None,  # 女巫救人目标
            "night_poison_target": None,  # 女巫毒人目标
            "game_over": False,
            "winner": None
        }

    async def commit_change(self, delta: Dict[str, Any]):
        async with self.lock:
            for key, value in delta.items():
                if key == "players":
                    for p_id, p_data in value.items():
                        if p_id in self.state["players"]:
                            self.state["players"][p_id].update(p_data)
                elif key == "kill":
                    self.state["night_kill_target"] = value
                elif key == "heal":
                    self.state["night_heal_target"] = value
                elif key == "poison":
                    self.state["night_poison_target"] = value
                elif key == "vote":
                    voter, target = value
                    self.state["current_votes"][voter] = target
                else:
                    self.state[key] = value
            await self.dm.save_game_state(self.game_id, self.state)
            return self.state


# ==========================================
# 5. 校验层 (PF & GRE) - 增加神职校验
# ==========================================
class PermissionFilter:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm

    async def validate(self, cmd_packet: Dict):
        state = self.gsm.state
        rid, data = cmd_packet["id"], cmd_packet["data"]
        uid, act = str(data.get("user_id")), data.get("action")
        user = state["players"].get(uid)

        ok, msg = False, "OK"
        if not user or not user["alive"]:
            msg = "玩家无效或已出局"
        elif state["phase"] == "NIGHT":
            role = user["role"]
            if act == "KILL" and role == "WEREWOLF":
                ok = True
            elif act == "VERIFY" and role == "SEER":
                ok = True
            elif act in ["HEAL", "POISON"] and role == "WITCH":
                ok = True
            else:
                msg = f"角色 {role} 夜晚无法执行 {act}"
        elif state["phase"] == "DAY":
            if act == "VOTE":
                ok = True
            else:
                msg = "白天只能投票"

        await bus.publish("PF_DONE", {"id": rid, "ok": ok, "msg": msg})


class RuleEngine:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm

    async def validate(self, cmd_packet: Dict):
        rid, data = cmd_packet["id"], cmd_packet["data"]
        uid, act = str(data.get("user_id")), data.get("action")
        tid = str(data.get("target_id"))
        target = self.gsm.state["players"].get(tid)
        user = self.gsm.state["players"].get(uid)

        ok, msg = True, "OK"
        if tid != "None" and (not target or not target["alive"]):
            ok, msg = False, "目标非法或已死亡"
        elif act == "HEAL" and not user.get("has_heal"):
            ok, msg = False, "解药已使用"

        await bus.publish("GRE_DONE", {"id": rid, "ok": ok, "msg": msg})


# ==========================================
# 6. 核心流程控制器 (GLC) - 修复死循环时序
# ==========================================
class GameLoopController:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm
        self.sync_registry = {}

    async def handle_agent_action(self, action_data: Dict):
        req_id = f"local_act_{uuid.uuid4().hex[:6]}"
        # 【修正】引入 Event 确保动作完成后主循环再继续
        done_ev = asyncio.Event()
        self.sync_registry[req_id] = {"pf": False, "gre": False, "raw": action_data, "event": done_ev}

        await bus.publish("INBOUND_ACTION", {"id": req_id, "data": action_data})
        await done_ev.wait()  # 阻塞主循环
        return req_id

    async def on_validation_callback(self, res: Dict, source: str):
        rid = res["id"]
        if rid not in self.sync_registry: return
        entry = self.sync_registry[rid]

        if not res["ok"]:
            logger.warning(f"行动拒绝 [{rid}]: {res['msg']}")
            entry["event"].set()
            del self.sync_registry[rid]
            return

        entry[source] = True
        if entry["pf"] and entry["gre"]:
            cmd = entry["raw"]
            uid, tid = str(cmd["user_id"]), str(cmd.get("target_id"))

            # 业务执行逻辑
            if cmd["action"] == "KILL":
                await self.gsm.commit_change({"kill": tid})
            elif cmd["action"] == "VERIFY":
                role = self.gsm.state["players"][tid]["role"]
                logger.info(f"🔮 [预言家验人] {tid} 号身份为: {role}")
            elif cmd["action"] == "HEAL":
                await self.gsm.commit_change({"heal": tid, "players": {uid: {"has_heal": False}}})
            elif cmd["action"] == "POISON":
                await self.gsm.commit_change({"poison": tid, "players": {uid: {"has_poison": False}}})
            elif cmd["action"] == "VOTE":
                await self.gsm.commit_change({"vote": (uid, tid)})

            logger.info(f"行动成功: {uid} {cmd['action']} -> {tid}")
            entry["event"].set()  # 释放主循环
            del self.sync_registry[rid]

    async def settle_night(self):
        s = self.gsm.state
        kill, heal, poison = s["night_kill_target"], s["night_heal_target"], s["night_poison_target"]

        dead_list = []
        if kill and kill != heal: dead_list.append(kill)
        if poison: dead_list.append(poison)

        for pid in set(dead_list):
            await self.gsm.commit_change({"players": {pid: {"alive": False}}})
            logger.info(f"📢 [公告] {pid} 号昨晚不幸遇害")

        if not dead_list: logger.info("📢 [公告] 昨晚是个平安夜")

        # 清理夜间临时状态
        await self.gsm.commit_change({"kill": None, "heal": None, "poison": None})
        await self._check_victory()

    async def settle_votes(self):
        votes = self.gsm.state["current_votes"]
        if not votes: return
        victim_id = Counter(votes.values()).most_common(1)[0][0]
        logger.info(f"📢 [公告] 玩家 {victim_id} 被放逐")
        await self.gsm.commit_change({"players": {victim_id: {"alive": False}}, "current_votes": {}})
        await self._check_victory()

    async def _check_victory(self):
        p = self.gsm.state["players"]
        wolves = [i for i, v in p.items() if v["alive"] and v["role"] == "WEREWOLF"]
        goods = [i for i, v in p.items() if v["alive"] and v["role"] != "WEREWOLF"]

        winner = None
        if not wolves:
            winner = "GOOD_SIDE"
        elif not goods:
            winner = "WOLF_SIDE"

        if winner:
            await self.gsm.commit_change({"game_over": True, "winner": winner})
            logger.info(f"🏆 游戏结束！获胜方: {winner}")

    async def run_game_loop(self):
        logger.info("🚀 游戏启动...")
        while not self.gsm.state["game_over"]:
            logger.info(f"\n--- 第 {self.gsm.state['day_count']} 天 ---")

            # 1. 夜晚 (顺序: 狼人 -> 女巫 -> 预言家)
            await self.gsm.commit_change({"phase": "NIGHT"})
            wolves = [i for i, p in self.gsm.state["players"].items() if p["alive"] and p["role"] == "WEREWOLF"]
            if wolves:
                await self.handle_agent_action({"user_id": wolves[0], "action": "KILL", "target_id": "9"})

            witch = self.gsm.state["players"]["6"]
            if witch["alive"] and self.gsm.state["night_kill_target"]:
                await self.handle_agent_action(
                    {"user_id": "6", "action": "HEAL", "target_id": self.gsm.state["night_kill_target"]})

            seer = self.gsm.state["players"]["5"]
            if seer["alive"]:
                await self.handle_agent_action({"user_id": "5", "action": "VERIFY", "target_id": "1"})

            # 2. 黎明结算
            await self.gsm.commit_change({"phase": "DAY"})
            await self.settle_night()
            if self.gsm.state["game_over"]: break

            # 3. 投票
            alives = [i for i, p in self.gsm.state["players"].items() if p["alive"]]
            for pid in alives:
                target = [i for i in alives if i != pid][0] if len(alives) > 1 else alives[0]
                await self.handle_agent_action({"user_id": pid, "action": "VOTE", "target_id": target})

            await self.settle_votes()
            await self.gsm.commit_change({"day_count": self.gsm.state["day_count"] + 1})


# ==========================================
# 7. 本地集成 (保持原样)
# ==========================================
class LocalDataManager(IDataManager):
    async def save_game_state(self, game_id, state): return True

    async def load_game_state(self, game_id): return None


async def main():
    dm = LocalDataManager()
    gsm = GameStateManager(dm)
    glc = GameLoopController(gsm)
    pf, gre = PermissionFilter(gsm), RuleEngine(gsm)

    bus.subscribe("INBOUND_ACTION", pf.validate)
    bus.subscribe("INBOUND_ACTION", gre.validate)
    bus.subscribe("PF_DONE", lambda r: glc.on_validation_callback(r, "pf"))
    bus.subscribe("GRE_DONE", lambda r: glc.on_validation_callback(r, "gre"))

    await glc.run_game_loop()


if __name__ == "__main__":
    asyncio.run(main())



