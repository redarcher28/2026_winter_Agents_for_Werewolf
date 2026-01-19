import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ==========================================
# 1. 基础配置与外部接口协议
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("WerewolfEngine")


class IDataManager(ABC):
    """
    外部 DM 团队必须实现的接口协议
    """

    @abstractmethod
    async def save_game_state(self, game_id: str, state: Dict[str, Any]) -> bool:
        pass


# ==========================================
# 2. 异步事件总线 (Event Bus)
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
            # 并行触发所有订阅异步任务
            await asyncio.gather(*(h(payload) for h in self._listeners[event_type]))


bus = EventBus()


# ==========================================
# 3. 状态管理器 (GSM)
# ==========================================
class GameStateManager:
    def __init__(self, dm: IDataManager):
        self.dm = dm
        self.game_id = "ROOM_888"
        self.lock = asyncio.Lock()
        # 初始内存状态
        self.state = {
            "phase": "NIGHT",
            "players": {
                "1": {"role": "WEREWOLF", "alive": True},
                "2": {"role": "SEER", "alive": True},
                "3": {"role": "VILLAGER", "alive": True},
                "4": {"role": "VILLAGER", "alive": True},
            },
            "current_votes": {},  # voter_id -> target_id
            "game_over": False,
            "winner": None
        }

    async def commit_change(self, delta: Dict[str, Any]):
        """核心：状态控制器指向数据控制器接口"""
        async with self.lock:
            # 处理玩家状态更新
            if "players" in delta:
                for p_id, p_data in delta["players"].items():
                    if p_id in self.state["players"]:
                        self.state["players"][p_id].update(p_data)
            # 处理投票记录
            elif "vote" in delta:
                voter, target = delta["vote"]
                self.state["current_votes"][voter] = target
            # 处理其他通用字段更新
            else:
                self.state.update(delta)

            # 强制触发外部团队提供的持久化接口
            await self.dm.save_game_state(self.game_id, self.state)
            return self.state


# ==========================================
# 4. 校验层 (PF & GRE)
# ==========================================
class PermissionFilter:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm

    async def validate(self, cmd_packet: Dict):
        state = self.gsm.state
        rid = cmd_packet["id"]
        data = cmd_packet["data"]
        uid = str(data.get("user_id"))
        act = data.get("action")

        user = state["players"].get(uid)
        ok, msg = False, "OK"

        # 权限矩阵逻辑
        if not user or not user["alive"]:
            msg = "玩家无效或已出局"
        elif act in ["KILL", "VERIFY"] and state["phase"] != "NIGHT":
            msg = "非夜晚阶段无法执行此行动"
        elif act == "KILL" and user["role"] != "WEREWOLF":
            msg = "只有狼人拥有杀人权限"
        elif act == "VERIFY" and user["role"] != "SEER":
            msg = "只有预言家拥有验人权限"
        else:
            ok = True

        await bus.publish("PF_DONE", {"id": rid, "ok": ok, "msg": msg})


class RuleEngine:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm

    async def validate(self, cmd_packet: Dict):
        state = self.gsm.state
        rid = cmd_packet["id"]
        tid = str(cmd_packet["data"].get("target_id"))

        target = state["players"].get(tid)
        ok, msg = True, "OK"

        # 规则逻辑：目标必须存活
        if tid and (not target or not target["alive"]):
            ok, msg = False, "目标玩家已死亡或不存在"

        await bus.publish("GRE_DONE", {"id": rid, "ok": ok, "msg": msg})


# ==========================================
# 5. 核心流程控制器 (GLC)
# ==========================================
class GameLoopController:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm
        self.sync_registry = {}  # 用于汇聚并行校验结果

    async def handle_action(self, action_data: Dict):
        """接受分发器传来的动作指令"""
        req_id = f"act_{uuid.uuid4().hex[:6]}"
        self.sync_registry[req_id] = {"pf": False, "gre": False, "raw": action_data}
        await bus.publish("INBOUND_ACTION", {"id": req_id, "data": action_data})

    async def on_validation_callback(self, res: Dict, source: str):
        rid = res["id"]
        if rid not in self.sync_registry: return

        entry = self.sync_registry[rid]
        if not res["ok"]:
            await bus.publish("WS_FEEDBACK", {"req_id": rid, "status": "ERROR", "msg": res["msg"]})
            del self.sync_registry[rid]
            return

        entry[source] = True
        if entry["pf"] and entry["gre"]:
            # 校验全通，提交至 GSM
            cmd = entry["raw"]
            await self.gsm.commit_change({"vote": (str(cmd["user_id"]), str(cmd["target_id"]))})
            await bus.publish("WS_FEEDBACK", {"req_id": rid, "status": "SUCCESS", "action": cmd["action"]})
            del self.sync_registry[rid]

    async def settle_votes(self):
        """核心判定：结算投票 + 平票序号小出局 + 胜负判定"""
        state = self.gsm.state
        votes = state["current_votes"]
        if not votes:
            await bus.publish("WS_FEEDBACK", {"type": "BROADCAST", "msg": "无人投票，平安夜/流局"})
            return

        # 1. 票数汇总
        vote_counts = Counter(votes.values())
        max_v = max(vote_counts.values())
        # 2. 找出平票者并按序号排序（取最小）
        winners = sorted([t for t, c in vote_counts.items() if c == max_v], key=int)
        victim_id = winners[0]

        # 3. 更新状态 (GSM -> DM)
        await self.gsm.commit_change({
            "players": {victim_id: {"alive": False}},
            "current_votes": {}  # 清空本轮投票
        })

        # 4. 触发胜负判定闭环
        await self._check_victory(victim_id)

    async def _check_victory(self, last_victim: str):
        state = self.gsm.state
        p = state["players"]

        alive_wolves = [i for i, v in p.items() if v["alive"] and v["role"] == "WEREWOLF"]
        alive_villagers = [i for i, v in p.items() if v["alive"] and v["role"] == "VILLAGER"]
        alive_gods = [i for i, v in p.items() if v["alive"] and v["role"] == "SEER"]

        winner = None
        if not alive_wolves:
            winner = "GOOD_SIDE (好人阵营获胜)"
        elif not alive_villagers or not alive_gods:
            winner = "WOLF_SIDE (狼人阵营获胜)"

        if winner:
            await self.gsm.commit_change({"game_over": True, "winner": winner})
            await bus.publish("WS_FEEDBACK", {"type": "GAME_OVER", "winner": winner, "last_out": last_victim})
        else:
            await bus.publish("WS_FEEDBACK", {"type": "SETTLEMENT", "out": last_victim, "msg": "游戏继续"})


# ==========================================
# 6. WebSocket 事件分发器 (Dispatcher)
# ==========================================
class WebSocketDispatcher:
    def __init__(self, glc: GameLoopController):
        self.glc = glc

    async def process_message(self, message: str):
        try:
            packet = json.loads(message)
            event = packet.get("event")
            data = packet.get("data", {})

            if event == "PLAYER_ACTION":
                await self.glc.handle_action(data)
            elif event == "CHAT":
                await bus.publish("WS_FEEDBACK",
                                  {"type": "CHAT", "from": data.get("user_id"), "content": data.get("content")})
            elif event == "PING":
                await bus.publish("WS_FEEDBACK", {"type": "PONG"})
            else:
                logger.warning(f"未知事件类型: {event}")
        except Exception as e:
            logger.error(f"消息解析失败: {e}")


# ==========================================
# 7. 系统组装与 FastAPI 路由接入
# ==========================================
app = FastAPI()


# 模拟外部 DM 团队的实现
class TeamDataManager(IDataManager):
    async def save_game_state(self, game_id, state):
        logger.info(
            f"[DM_SERVICE] 收到状态推送 - 房间: {game_id}, 存活人数: {sum(1 for p in state['players'].values() if p['alive'])}")
        return True


# 依赖注入与绑定
dm_service = TeamDataManager()
gsm = GameStateManager(dm_service)
glc = GameLoopController(gsm)
pf = PermissionFilter(gsm)
gre = RuleEngine(gsm)
dispatcher = WebSocketDispatcher(glc)

# 事件总线订阅
bus.subscribe("INBOUND_ACTION", pf.validate)
bus.subscribe("INBOUND_ACTION", gre.validate)
bus.subscribe("PF_DONE", lambda r: glc.on_validation_callback(r, "pf"))
bus.subscribe("GRE_DONE", lambda r: glc.on_validation_callback(r, "gre"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 注册闭环反馈：将总线结果实时推回 WebSocket
    async def fb_handler(payload):
        try:
            await websocket.send_json(payload)
        except:
            pass

    bus.subscribe("WS_FEEDBACK", fb_handler)

    try:
        while True:
            raw_msg = await websocket.receive_text()
            # 进入分发接口
            await dispatcher.process_message(raw_msg)
    except WebSocketDisconnect:
        logger.info("客户端连接已断开")


@app.post("/system/settle")
async def manual_settle():
    """手动触发结算接口"""
    await glc.settle_votes()
    return {"status": "Settle process executed"}

# ==========================================
# 启动说明
# 命令行运行: uvicorn filename:app --reload
# ==========================================