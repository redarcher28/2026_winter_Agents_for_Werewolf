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

# ==========================================
# 2. 接口契约：定义你对数据层（DM）的要求
# ==========================================
# question: 这个类是否由徐子灏负责？
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
    """
    异步事件总线，用于多个处理模块之间的异步数据传输
    para: _listeners: 监听器字典，结构为 { event_type: [监听器函数列表], ...}
    """
    def __init__(self):
        self._listeners = {}

    # 注册监听event_type事件类型的函数handler
    # route: 0-1
    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        if event_type not in self._listeners: # 如果该事件类型并没有注册在event_type中，则加入
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler) # 注册监听event_type事件类型的函数handler

    # route: 1-1-1
    async def publish(self, event_type: str, payload: Any):
        """
        将数据上传到总线
        :param event_type: 事件类型
        :param payload: ？
        """
        # 检查 event_type 是否在 self._listeners 字典中注册过监听器
        if event_type in self._listeners:
            # 异步事件监听的核心部分，用于触发特定类型的事件的所有监听器（handler）
            # h代表handler， 该代码将监听event_type的所有监听函数一起执行，参数均为payload
            # question: payload是用来干嘛的？为什么类型是any？
            await asyncio.gather(*(h(payload) for h in self._listeners[event_type]))

bus = EventBus()

# ==========================================
# 4. 状态管理器 (GSM)
# ==========================================
class GameStateManager:
    """
    游戏状态管理器
    para: dm: 数据管理器
    para: game_id: 本局游戏的ID号
    para: lock: 异步互斥锁
    para: state: 游戏状态

    # 多个协程尝试修改 state 时：
    # 协程1:
    async with self.lock:  # 获取锁成功
    # 修改 state...

    # 协程2:
    async with self.lock:  # 等待锁释放（协程1还在执行）
    # 等待...直到协程1释放锁
    # 然后获取锁并执行

    """
    def __init__(self, dm: IDataManager, game_id: str = "ROOM_888"):
        self.dm = dm #
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
    # route: 4-1-1
    # route: 2-1-1
    async def commit_change(self, delta: Dict[str, Any]):
        """
        核心：业务计算完成后，强制同步到本地 DM
        para: delta: 具体动作，例如{"vote": (str(cmd["user_id"]), str(cmd["target_id"]))}
        """
        # 在提交更改时必须互斥
        async with self.lock:
            if "players" in delta:
                for p_id, p_data in delta["players"].items():
                    if p_id in self.state["players"]:
                        self.state["players"][p_id].update(p_data)
                        # dict().update函数作用：
                        # 将另一个字典的键值对更新 / 合并到当前字典中：
                        # 如果键已存在：覆盖原有值
                        # 如果键不存在：添加新的键值对
            elif "vote" in delta:
                voter, target = delta["vote"]
                self.state["current_votes"][voter] = target
            else:
                self.state.update(delta)

            # 调用本地数据管理存储接口
            success = await self.dm.save_game_state(self.game_id, self.state)
            if not success:
                logger.error("本地数据保存失败")
            return self.state

# ==========================================
# 5. 校验逻辑 (PF & GRE)
# ==========================================
class PermissionFilter:
    """
    权限过滤器
    para: gsm: 游戏状态管理器
    """
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm


    # question:
    #  await asyncio.gather(*(h(payload) for h in self._listeners[event_type]))
    #  中, h(payload)实际引用的是validate(cmd_packet), payload和cmd_packet之间的联系是
    #  答： payload与cmd_packet等价，是由总线传递的事件数据

    # 1. 发布事件
    # await bus.publish("INBOUND_ACTION", {"id": "act_123", "data": {"user_id": "1", "action": "KILL"}})
    #
    # # 2. 事件数据成为 payload
    # # payload = {"id": "act_123", "data": {"user_id": "1", "action": "KILL"}}
    #
    # # 3. 调用所有监听器
    # for handler in self._listeners["INBOUND_ACTION"]:
    #     # handler 是 pf.validate
    #     # 调用: pf.validate(payload)
    #     await handler(payload)
    # route: 0-1所传目标函数
    async def validate(self, cmd_packet: Dict):
        state = self.gsm.state # 将当前游戏状态赋值到state中
        rid, data = cmd_packet["id"], cmd_packet["data"] # 将本次事件的id和data传入rid， data
        uid, act = str(data.get("user_id")), data.get("action") # 将本次事件的动作来源人，具体动作名称传入uid，act
        user = state["players"].get(uid)

        # question: ok, msg都是用来干嘛的？
        #  答：ok用于表示该动作是否成功过滤，msg是不同过滤结果的字符串表示
        ok, msg = False, "OK"
        if not user or not user["alive"]: # 如果没有这个玩家，或者玩家已死亡
            msg = "玩家无效或已出局"
        elif act in ["KILL", "VERIFY"] and state["phase"] != "NIGHT": # 如果agent在白天做出了杀人或验人的动作
            msg = "非夜晚阶段"
        elif act == "KILL" and user["role"] != "WEREWOLF": # 如果非狼人做出了杀人的动作
            msg = "无杀人权限"
        else:
            ok = True
        await bus.publish("PF_DONE", {"id": rid, "ok": ok, "msg": msg})

class RuleEngine:
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm
    # route: 3-1 规则验证，目前只实现了判断当前动作目标制定是否非法
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
    """
    控制游戏流程循环
    para:
        gsm: 游戏状态管理器
        sync_registry: dict(),
    """
    def __init__(self, gsm: GameStateManager):
        self.gsm = gsm  # gsm：游戏状态管理器
        # question: sync_registry是用来干嘛的？
        #  答：用于记录所有动作是否已经过fl和gre的处理
        self.sync_registry = {}

    # route: 1-1 处理agent行动, action_data应为agent返回的动作结果，需要调用agent类中生成这个结果的函数
    async def handle_agent_action(self, action_data: Dict):
        """
        供本地 AI Agent 调用的接口
        action_data: {"user_id": "1", "action": "KILL", "target_id": "3"}
        re: 返回该行动的id号
        """
        req_id = f"local_act_{uuid.uuid4().hex[:6]}" # 生成动作id
        # 可能的输出：
        # req_id = "local_act_a1b2c3"
        # req_id = "local_act_f456e7"
        # req_id = "local_act_9a8b7c"

        # question： pf, gre, raw代表什么意思？
        #  答：pf: permission filter, gre: gamerule, raw: pf和gre的目标动作数据？
        self.sync_registry[req_id] = {"pf": False, "gre": False, "raw": action_data}

        await bus.publish("INBOUND_ACTION", {"id": req_id, "data": action_data})
        return req_id
    # route 4-1 验证回调
    async def on_validation_callback(self, res: Dict, source: str):
        """
        功能：处理验证结果的回调函数
        para:
            res: 验证结果，包含 {"id": req_id, "ok": bool, "msg": str}
            source: 验证来源，值为 "pf"(permission filter) 或 "gre"(gamerule)
        """
        rid = res["id"]
        if rid not in self.sync_registry: return
        entry = self.sync_registry[rid]

        if not res["ok"]:
            logger.warning(f"行动拒绝 [{rid}]: {res['msg']}")
            del self.sync_registry[rid]
            return

        entry[source] = True
        if entry["pf"] and entry["gre"]:
            cmd = entry["raw"] # 将动作数据传入command（cmd）

            # todo: 将参数{"vote": (str(cmd["user_id"]), str(cmd["target_id"]))}改为一般性的数据
            await self.gsm.commit_change({"vote": (str(cmd["user_id"]), str(cmd["target_id"]))})
            logger.info(f"行动成功执行: {cmd['user_id']} -> {cmd['target_id']}")
            del self.sync_registry[rid]

    # route 2-1 确定投票结果
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

    # route 2-1-2 确定本局游戏哪方获胜
    async def _check_victory(self, last_victim: str):
        """
        确定本局游戏哪方获胜
        """
        state = self.gsm.state
        p = state["players"]
        alive_wolves = [i for i, v in p.items() if v["alive"] and v["role"] == "WEREWOLF"]
        alive_villagers = [i for i, v in p.items() if v["alive"] and v["role"] == "VILLAGER"]
        alive_gods = [i for i, v in p.items() if v["alive"] and v["role"] == "SEER"]

        winner = None
        if not alive_wolves: winner = "GOOD_SIDE"
        # fixme: 狼人胜利的条件应该是特殊好人职业和村民无一幸存
        # elif not alive_villagers or not alive_gods: winner = "WOLF_SIDE"
        elif not alive_villagers and not alive_gods: winner = "WOLF_SIDE"
        if winner:
            await self.gsm.commit_change({"game_over": True, "winner": winner})
            logger.info(f"游戏结束！获胜方: {winner}")
        # todo: 把游戏结果传送给数据管理器

    # todo： 实现游戏的流程主循环
    def _game_loop(self):
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

# todo: 完成一个完整的游戏循环
async def main():
    # 初始化
    dm = LocalDataManager() # 本地数据管理器
    gsm = GameStateManager(dm) # 游戏状态管理器 参数为本地数据管理器
    glc = GameLoopController(gsm) # 游戏循环控制器，参数为游戏状态管理器
    pf = PermissionFilter(gsm) # 权限过滤器， 参数为游戏状态管理器
    gre = RuleEngine(gsm) # 规则引擎， 参数为游戏状态管理器

    # 订阅事件总线
    # question: bus.subscribe(str, Callable)是干什么用的
    #  答: INBOUND_ACTION表示该数据类型是action型，需要规则引擎和权限过滤器进行监听
    # route: 0 发现新的动作时
    bus.subscribe("INBOUND_ACTION", pf.validate)
    # route: 3 发现新的动作时
    bus.subscribe("INBOUND_ACTION", gre.validate)
    # route: 4 权限过滤完成时
    bus.subscribe("PF_DONE", lambda r: glc.on_validation_callback(r, "pf"))
    # route: 5 规则验证完成时
    bus.subscribe("GRE_DONE", lambda r: glc.on_validation_callback(r, "gre"))

    # 模拟 AI Agent 本地调用
    print("--- 模拟 AI 玩家 1 杀 3 号 ---")
    # route: 1 处理agent的新动作
    # todo: 需要与第二组确定agent返回的实际动作格式， 将参数改为一般形式
    await glc.handle_agent_action({"user_id": "1", "action": "KILL", "target_id": "3"})

    # question: 为什么要睡0.1秒？
    await asyncio.sleep(0.1)
    
    # 模拟系统结算
    # route: 2
    print("--- 触发投票结算 ---")
    await glc.settle_votes()


if __name__ == "__main__":
    asyncio.run(main())

# ==========================================

