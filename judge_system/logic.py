import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Any, List, Optional

# ==========================================
# 1. 基础配置
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("WerewolfBLL")

# ==========================================
# 2. 接口契约：定义你对数据层（DM）的要求
# ==========================================
class IDataManager(ABC):
    """
    业务逻辑层要求本地数据层必须实现的接口
    """
    @abstractmethod
    async def save_game_state(self, game_id: str, state: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    async def load_game_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        pass

# ==========================================
# 3. 异步事件总线 (维持解耦)
# ==========================================
class EventBus:
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_type: str, handler):
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)

    async def publish(self, event_type: str, payload: Any):
        if event_type in self._listeners:
            await asyncio.gather(*(h(payload) for h in self._listeners[event_type]))

bus = EventBus()

# ==========================================
# 4. 状态管理器 (GSM)
# ==========================================
class GameStateManager:
    def __init__(self, dm: IDataManager, game_id: str = "ROOM_888"):
        self.dm = dm
        self.game_id = game_id
        self.lock = asyncio.Lock()
        self.state = {
            "phase": "NIGHT",
            "players": {
                "1": {"role": "WEREWOLF", "alive": True},
                "2": {"role": "SEER", "alive": True},
                "3": {"role": "VILLAGER", "alive": True},
                "4": {"role": "VILLAGER", "alive": True},
            },
            "current_votes": {},
            "game_over": False,
            "winner": None
        }

    async def commit_change(self, delta: Dict[str, Any]):
        """核心：业务计算完成后，强制同步到本地 DM"""
        async with self.lock:
            if "players" in delta:
                for p_id, p_data in delta["players"].items():
                    if p_id in self.state["players"]:
                        self.state["players"][p_id].update(p_data)
            elif "vote" in delta:
                voter, target = delta["vote"]
                self.state["current_votes"][voter] = target
            else:
                self.state.update(delta)

            # 调用本地持久化接口
            success = await self.dm.save_game_state(self.game_id, self.state)
            if not success:
                logger.error("本地数据保存失败")
            return self.state

# ==========================================
# 5. 校验逻辑 (PF & GRE)
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
        elif act in ["KILL", "VERIFY"] and state["phase"] != "NIGHT":
            msg = "非夜晚阶段"
        elif act == "KILL" and user["role"] != "WEREWOLF":
            msg = "无杀人权限"
        else:
            ok = True
        await bus.publish("PF_DONE", {"id": rid, "ok": ok, "msg": msg})

class RuleEngine:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm

    async def validate(self, cmd_packet: Dict):
        state = self.gsm.state
        rid, tid = cmd_packet["id"], str(cmd_packet["data"].get("target_id"))
        target = state["players"].get(tid)
        
        ok, msg = True, "OK"
        if tid and (not target or not target["alive"]):
            ok, msg = False, "目标非法"
        await bus.publish("GRE_DONE", {"id": rid, "ok": ok, "msg": msg})

# ==========================================
# 6. 核心流程控制器 (GLC)
# ==========================================
class GameLoopController:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm
        self.sync_registry = {}

    async def handle_agent_action(self, action_data: Dict):
        """
        供本地 AI Agent 调用的接口
        action_data: {"user_id": "1", "action": "KILL", "target_id": "3"}
        """
        req_id = f"local_act_{uuid.uuid4().hex[:6]}"
        self.sync_registry[req_id] = {"pf": False, "gre": False, "raw": action_data}
        await bus.publish("INBOUND_ACTION", {"id": req_id, "data": action_data})
        return req_id

    async def on_validation_callback(self, res: Dict, source: str):
        rid = res["id"]
        if rid not in self.sync_registry: return
        entry = self.sync_registry[rid]

        if not res["ok"]:
            logger.warning(f"行动拒绝 [{rid}]: {res['msg']}")
            del self.sync_registry[rid]
            return

        entry[source] = True
        if entry["pf"] and entry["gre"]:
            cmd = entry["raw"]
            await self.gsm.commit_change({"vote": (str(cmd["user_id"]), str(cmd["target_id"]))})
            logger.info(f"行动成功执行: {cmd['user_id']} -> {cmd['target_id']}")
            del self.sync_registry[rid]

    async def settle_votes(self):
        """核心业务规则：平票序号小出局 + 胜负判定"""
        state = self.gsm.state
        votes = state["current_votes"]
        if not votes: return

        vote_counts = Counter(votes.values())
        max_v = max(vote_counts.values())
        # 规则：平票选序号最小者
        winners = sorted([t for t, c in vote_counts.items() if c == max_v], key=int)
        victim_id = winners[0]

        await self.gsm.commit_change({
            "players": {victim_id: {"alive": False}},
            "current_votes": {}
        })
        await self._check_victory(victim_id)

    async def _check_victory(self, last_victim: str):
        state = self.gsm.state
        p = state["players"]
        alive_wolves = [i for i, v in p.items() if v["alive"] and v["role"] == "WEREWOLF"]
        alive_villagers = [i for i, v in p.items() if v["alive"] and v["role"] == "VILLAGER"]
        alive_gods = [i for i, v in p.items() if v["alive"] and v["role"] == "SEER"]

        winner = None
        if not alive_wolves: winner = "GOOD_SIDE"
        elif not alive_villagers or not alive_gods: winner = "WOLF_SIDE"

        if winner:
            await self.gsm.commit_change({"game_over": True, "winner": winner})
            logger.info(f"游戏结束！获胜方: {winner}")

# ==========================================
# 7. 本地集成示例
# ==========================================
class LocalDataManager(IDataManager):
    """本地数据管理层实现类"""
    async def save_game_state(self, game_id, state):
        # 这里可以是写入本地文件、SQLite 或内存字典
        logger.info(f"[Local DM] 状态已保存至本地: {game_id}")
        return True

    async def load_game_state(self, game_id):
        return None

async def main():
    # 初始化
    dm = LocalDataManager()
    gsm = GameStateManager(dm)
    glc = GameLoopController(gsm)
    pf = PermissionFilter(gsm)
    gre = RuleEngine(gsm)

    # 订阅事件总线
    bus.subscribe("INBOUND_ACTION", pf.validate)
    bus.subscribe("INBOUND_ACTION", gre.validate)
    bus.subscribe("PF_DONE", lambda r: glc.on_validation_callback(r, "pf"))
    bus.subscribe("GRE_DONE", lambda r: glc.on_validation_callback(r, "gre"))

    # 模拟 AI Agent 本地调用
    print("--- 模拟 AI 玩家 1 杀 3 号 ---")
    await glc.handle_agent_action({"user_id": "1", "action": "KILL", "target_id": "3"})
    
    await asyncio.sleep(0.1)
    
    # 模拟系统结算
    print("--- 触发投票结算 ---")
    await glc.settle_votes()

if __name__ == "__main__":
    asyncio.run(main())

# ==========================================

