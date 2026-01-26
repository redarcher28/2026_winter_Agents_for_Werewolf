import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Any, List, Optional, Callable

# ==========================================
# 1. 基础配置
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

    # route 1-1: 订阅事件到handler
    # route 2-1: 订阅事件到handler
    # route 3-1: 订阅事件到lambda包装器
    # route 4-1: 订阅事件到lambda包装器
    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)

    # route 1-2: 发布事件到所有订阅者
    # route 2-2: 发布事件到所有订阅者
    # route 3-2: 发布事件到lambda包装器
    # route 4-2: 发布事件到lambda包装器
    # route 5-1: 从GLC.handle_agent_action发布
    # route 6-1: 从PF.validate发布
    # route 7-1: 从GRE.validate发布
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
                # 初始化12人局：1-4狼，5-8神(GOD)，9-12民(VILLAGER)
                **{str(i): {"role": "WEREWOLF", "alive": True} for i in range(1, 5)},
                **{str(i): {"role": "GOD", "alive": True} for i in range(5, 9)},
                **{str(i): {"role": "VILLAGER", "alive": True} for i in range(9, 13)},
            },
            "current_votes": {},
            "night_kill_target": None,  # 存储夜晚猎杀意图
            "game_over": False,
            "winner": None
        }

    # route 8-1: 从GLC.on_validation_callback调用
    # route 9-1: 从GLC.run_game_loop调用
    # route 10-1: 从GLC.settle_night调用
    # route 11-1: 从GLC.settle_votes调用
    # route 12-1: 从GLC._check_victory调用
    async def commit_change(self, delta: Dict[str, Any]):
        async with self.lock:
            for key, value in delta.items():
                if key == "players":
                    for p_id, p_data in value.items():
                        if p_id in self.state["players"]:
                            self.state["players"][p_id].update(p_data)
                elif key == "kill":  # 记录夜晚猎杀
                    self.state["night_kill_target"] = value
                elif key == "vote":  # 记录白天投票
                    voter, target = value
                    self.state["current_votes"][voter] = target
                else:
                    self.state[key] = value

            # route 8-2: 调用DM保存状态
            # route 9-2: 调用DM保存状态
            # route 10-2: 调用DM保存状态
            # route 11-2: 调用DM保存状态
            # route 12-2: 调用DM保存状态
            # todo: 只保存了游戏状态，没有保存玩家发言，玩家状态等
            #  与徐子灏对接，询问其实现了哪些存储功能，是否需要，
            #  然后相应像下面的函数一样实现引用
            await self.dm.save_game_state(self.game_id, self.state)
            return self.state


# ==========================================
# 5. 校验层 (PF & GRE)
# ==========================================
class PermissionFilter:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm

    # route 1-3: 事件总线调用
    async def validate(self, cmd_packet: Dict):
        state = self.gsm.state
        rid, data = cmd_packet["id"], cmd_packet["data"]
        uid, act = str(data.get("user_id")), data.get("action")
        user = state["players"].get(uid)

        ok, msg = False, "OK"
        if not user or not user["alive"]:
            msg = "玩家无效或已出局"
        elif act == "KILL" and (state["phase"] != "NIGHT" or user["role"] != "WEREWOLF"):
            msg = "非狼人或非夜晚禁止杀人"
        elif act == "VOTE" and state["phase"] != "DAY":
            msg = "非白天阶段禁止投票"
        else:
            ok = True
        # route 6-2: 发布PF_DONE事件
        await bus.publish("PF_DONE", {"id": rid, "ok": ok, "msg": msg})


class RuleEngine:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm

    # route 2-3: 事件总线调用
    async def validate(self, cmd_packet: Dict):
        state = self.gsm.state
        rid, tid = cmd_packet["id"], str(cmd_packet["data"].get("target_id"))
        target = state["players"].get(tid)
        ok, msg = True, "OK"
        if tid and (not target or not target["alive"]):
            ok, msg = False, "目标非法"
        # route 7-2: 发布GRE_DONE事件
        await bus.publish("GRE_DONE", {"id": rid, "ok": ok, "msg": msg})


# ==========================================
# 6. 核心流程控制器 (GLC)
# ==========================================
class GameLoopController:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm
        self.sync_registry = {}

    # route 5-2: 从run_game_loop调用
    async def handle_agent_action(self, action_data: Dict):
        req_id = f"local_act_{uuid.uuid4().hex[:6]}"
        self.sync_registry[req_id] = {"pf": False, "gre": False, "raw": action_data}
        # route 5-3: 发布INBOUND_ACTION事件
        await bus.publish("INBOUND_ACTION", {"id": req_id, "data": action_data})
        return req_id

    # route 3-3: lambda包装器调用（参数r来自事件总线）
    # route 4-3: lambda包装器调用（参数r来自事件总线）
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
            # route 8: 执行通过验证的动作
            if cmd["action"] == "KILL":
                # route 8-3: 提交杀动作到GSM
                await self.gsm.commit_change({"kill": str(cmd["target_id"])})
            elif cmd["action"] == "VOTE":
                # route 8-4: 提交投票动作到GSM
                await self.gsm.commit_change({"vote": (str(cmd["user_id"]), str(cmd["target_id"]))})

            logger.info(f"行动成功执行: {cmd['user_id']} {cmd['action']} -> {cmd['target_id']}")
            del self.sync_registry[rid]

    async def settle_night(self):
        """结算夜晚猎杀"""
        target = self.gsm.state.get("night_kill_target")
        if target:
            logger.info(f"📢 [公告] 昨晚遇害的是 {target} 号")
            # route 10: 夜晚结算
            await self.gsm.commit_change({
                "players": {target: {"alive": False}},
                "night_kill_target": None
            })
        else:
            logger.info("📢 [公告] 昨晚是个平安夜")
        await self._check_victory()

    async def settle_votes(self):
        """结算投票"""
        votes = self.gsm.state["current_votes"]
        if not votes: return
        vote_counts = Counter(votes.values())
        max_v = max(vote_counts.values())
        winners = sorted([t for t, c in vote_counts.items() if c == max_v], key=int)
        victim_id = winners[0]

        logger.info(f"📢 [公告] 玩家 {victim_id} 被公投出局")
        # route 11: 投票结算
        await self.gsm.commit_change({
            "players": {victim_id: {"alive": False}},
            "current_votes": {}
        })
        await self._check_victory()

    async def _check_victory(self):
        # route 12: 检查胜利条件
        p = self.gsm.state["players"]
        alive_wolves = [i for i, v in p.items() if v["alive"] and v["role"] == "WEREWOLF"]
        alive_goods = [i for i, v in p.items() if v["alive"] and v["role"] in ["GOD", "VILLAGER"]]

        winner = None
        if not alive_wolves:
            winner = "GOOD_SIDE"
        elif not alive_goods:
            winner = "WOLF_SIDE"

        if winner:
            # route 12-3: 提交游戏结束状态
            await self.gsm.commit_change({"game_over": True, "winner": winner})
            logger.info(f"🏆 游戏结束！获胜方: {winner}")

    # route 13: 主程序入口
    async def run_game_loop(self):
        logger.info("🚀 游戏主循环启动...")
        round_num = 1
        while not self.gsm.state["game_over"]:
            logger.info(f"\n--- 第 {round_num} 轮循环 ---")

            # 1. 夜晚
            # route 9: 阶段切换
            await self.gsm.commit_change({"phase": "NIGHT"})
            # 模拟狼人杀人行动
            alive_wolves = [i for i, p in self.gsm.state["players"].items() if p["alive"] and p["role"] == "WEREWOLF"]
            alive_targets = [i for i, p in self.gsm.state["players"].items() if p["alive"] and p["role"] != "WEREWOLF"]
            if alive_wolves and alive_targets:
                # route 5: 处理代理动作
                await self.handle_agent_action(
                    {"user_id": alive_wolves[0], "action": "KILL", "target_id": alive_targets[0]})
                await asyncio.sleep(0.1)  # 等待总线处理

            # 2. 黎明结算
            await self.gsm.commit_change({"phase": "DAY"})
            await self.settle_night()
            if self.gsm.state["game_over"]: break

            # 3. 白天投票
            alive_all = [i for i, p in self.gsm.state["players"].items() if p["alive"]]
            for p_id in alive_all:
                # 简单逻辑：所有人投1号（如果是活着的且不是自己），否则投活着的第一个
                target = "1" if "1" in alive_all and p_id != "1" else alive_all[0]
                await self.handle_agent_action({"user_id": p_id, "action": "VOTE", "target_id": target})

            await asyncio.sleep(0.1)
            await self.settle_votes()

            round_num += 1
            await asyncio.sleep(0.5)


# ==========================================
# 7. 本地集成
# ==========================================
class LocalDataManager(IDataManager):
    # route 8-5: 保存游戏状态
    # route 9-3: 保存游戏状态
    # route 10-3: 保存游戏状态
    # route 11-3: 保存游戏状态
    # route 12-4: 保存游戏状态
    async def save_game_state(self, game_id, state):
        logger.info(f"[DM] 状态快照已更新 (Game: {game_id})")
        return True

    async def load_game_state(self, game_id): return None


async def main():
    dm = LocalDataManager()
    gsm = GameStateManager(dm)
    glc = GameLoopController(gsm)
    pf = PermissionFilter(gsm)
    gre = RuleEngine(gsm)

    # route 1: 订阅INBOUND_ACTION到PF验证器
    bus.subscribe("INBOUND_ACTION", pf.validate)
    # route 2: 订阅INBOUND_ACTION到GRE验证器
    bus.subscribe("INBOUND_ACTION", gre.validate)
    # route 3: 订阅PF验证结果到GLC回调
    bus.subscribe("PF_DONE", lambda r: glc.on_validation_callback(r, "pf"))
    # route 4: 订阅GRE验证结果到GLC回调
    bus.subscribe("GRE_DONE", lambda r: glc.on_validation_callback(r, "gre"))

    # route 13-2: 启动游戏主循环
    await glc.run_game_loop()


if __name__ == "__main__":
    asyncio.run(main())