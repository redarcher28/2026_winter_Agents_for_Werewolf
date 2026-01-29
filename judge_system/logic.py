import asyncio
import logging
import uuid
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

# ==========================================
# 1. 基础配置
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("WerewolfJudge")


# ==========================================
# 2. 校验系统：PF (权限) & GRE (规则)
# ==========================================
class PermissionFilter:
    """第一道防线：校验身份与阶段权限"""

    @staticmethod
    def validate(state: Dict, cmd: Dict) -> Tuple[bool, str]:
        uid = str(cmd.get("user_id"))
        act = cmd.get("action")
        user = state["players"].get(uid)
        phase = state["phase"]

        if not user or not user["alive"]:
            return False, f"玩家 {uid} 无效或已出局"

        # 阶段权限校验
        if phase == "NIGHT":
            role_perms = {
                "WEREWOLF": ["KILL", "WOLF_CHAT"],
                "SEER": ["VERIFY"],
                "WITCH": ["HEAL", "POISON", "WAIT"]
            }
            if act not in role_perms.get(user["role"], ["WAIT"]):
                return False, f"角色 {user['role']} 在夜晚无权执行 {act}"
        elif phase == "DAY_DISCUSSION":
            if act != "SPEECH": return False, "讨论阶段只能执行 SPEECH"
        elif phase == "DAY_VOTING":
            if act != "VOTE": return False, "投票阶段只能执行 VOTE"

        return True, "PF_PASS"


class RuleEngine:
    """第二道防线：校验动作逻辑合法性"""

    @staticmethod
    def validate(state: Dict, cmd: Dict) -> Tuple[bool, str]:
        act = cmd.get("action")
        tid = str(cmd.get("target_id"))
        uid = str(cmd.get("user_id"))

        # 目标存活校验
        if act in ["KILL", "VERIFY", "POISON", "VOTE"]:
            target = state["players"].get(tid)
            if not target or not target["alive"]:
                return False, f"目标 {tid} 已死亡或不存在"

        # 女巫规则校验
        if act == "HEAL":
            if tid == uid: return False, "当前规则禁止女巫自救"
            if not state["players"][uid].get("has_heal"): return False, "解药已耗尽"

        if act == "POISON":
            if not state["players"][uid].get("has_poison"): return False, "毒药已耗尽"

        return True, "GRE_PASS"


# ==========================================
# 3. 状态与逻辑核心 (GSM & GLC)
# ==========================================
class GameEngine:
    def __init__(self):
        self.state = {
            "game_id": "STRICT_12_ROOM",
            "phase": "INIT",
            "day_count": 1,
            "game_over": False,
            "players": {
                **{str(i): {"role": "WEREWOLF", "alive": True} for i in range(1, 5)},
                "5": {"role": "SEER", "alive": True},
                "6": {"role": "WITCH", "alive": True, "has_heal": True, "has_poison": True},
                **{str(i): {"role": "GOD", "alive": True} for i in range(7, 9)},
                **{str(i): {"role": "VILLAGER", "alive": True} for i in range(9, 13)},
            },
            "discussion_history": [],
            "wolf_night_history": [],
            "night_kill_target": None,
            "witch_heal_target": None,
            "witch_poison_target": None,
            "current_votes": {}
        }

    async def commit_change(self, delta: Dict):
        """原子化更新状态"""
        for k, v in delta.items():
            if k == "players":
                for pid, pdata in v.items(): self.state["players"][pid].update(pdata)
            else:
                self.state[k] = v

    async def process_action(self, cmd: Dict) -> bool:
        """集成 PF 和 GRE 的统一动作入口"""
        if not cmd or cmd.get("action") == "WAIT": return True

        # 1. 权限过滤
        pf_ok, pf_msg = PermissionFilter.validate(self.state, cmd)
        if not pf_ok:
            logger.warning(f"🛡️ PF 拦截: {pf_msg}")
            return False

        # 2. 规则校验
        gre_ok, gre_msg = RuleEngine.validate(self.state, cmd)
        if not gre_ok:
            logger.warning(f"⚖️ GRE 拦截: {gre_msg}")
            return False

        # 3. 执行生效
        await self._apply_effect(cmd)
        return True

    async def _apply_effect(self, cmd: Dict):
        act, uid, tid = cmd["action"], str(cmd["user_id"]), str(cmd.get("target_id"))
        if act == "KILL":
            self.state["night_kill_target"] = tid
        elif act == "HEAL":
            self.state["witch_heal_target"] = tid
            self.state["players"][uid]["has_heal"] = False
        elif act == "POISON":
            self.state["witch_poison_target"] = tid
            self.state["players"][uid]["has_poison"] = False
        elif act == "SPEECH":
            self.state["discussion_history"].append({"user_id": uid, "content": cmd.get("content", "")})
        elif act == "WOLF_CHAT":
            self.state["wolf_night_history"].append({"user_id": uid, "content": cmd.get("content", "")})
        elif act == "VOTE":
            self.state["current_votes"][uid] = tid

    def settle_night(self):
        s = self.state
        kill, heal, poison = s["night_kill_target"], s["witch_heal_target"], s["witch_poison_target"]
        dead = set()
        if kill and kill != heal: dead.add(kill)
        if poison: dead.add(poison)

        for pid in dead: s["players"][pid]["alive"] = False
        logger.info(f"📢 [天亮了] 昨晚死亡玩家: {', '.join(dead) if dead else '无（平安夜）'}")
        s.update({"night_kill_target": None, "witch_heal_target": None, "witch_poison_target": None})

    def settle_votes(self):
        v = self.state["current_votes"]
        if not v:
            # 补救机制：没人投票则处决序号最小的活人，防止死循环
            target = sorted([i for i, p in self.state["players"].items() if p["alive"]], key=int)[0]
            logger.warning(f"⚠️ 无有效投票，法官强制处决玩家 {target}")
        else:
            target = Counter(v.values()).most_common(1)[0][0]

        self.state["players"][target]["alive"] = False
        logger.info(f"📢 [公投结果] 玩家 {target} 被投票出局")
        self.state["current_votes"] = {}

    def check_victory(self) -> bool:
        p = self.state["players"]
        wolves = [i for i, v in p.items() if v["alive"] and v["role"] == "WEREWOLF"]
        goods = [i for i, v in p.items() if v["alive"] and v["role"] != "WEREWOLF"]

        if not wolves:
            logger.info("🏆 好人胜利！"); self.state["game_over"] = True
        elif not goods:
            logger.info("🏆 狼人胜利！"); self.state["game_over"] = True
        return self.state["game_over"]


# ==========================================
# 4. StrategyContext 构造器
# ==========================================
class ContextBuilder:
    @staticmethod
    def build(gsm_state: Dict, user_id: str, current_speaker: str = None, turn_idx: int = 0) -> Dict[str, Any]:
        p = gsm_state["players"]
        me = p[user_id]
        ctx = {
            "meta": {
                "role": me["role"].lower(), "phase": gsm_state["phase"].lower(),
                "day_number": gsm_state["day_count"], "self_player_id": user_id,
                "current_speaker_id": current_speaker
            },
            "public_state": {
                "alive_players": [{"id": pid} for pid, info in p.items() if info["alive"]],
                "dead_players": [{"id": pid} for pid, info in p.items() if not info["alive"]],
            },
            "private_info": {},
            "constraints": {
                "allowed_actions": [],
                "forbid_targets": [pid for pid, info in p.items() if not info["alive"]],
                "current_turn_order": turn_idx
            },
            "memory": {"memory_summary": "\n".join(
                [f"{h['user_id']}: {h['content']}" for h in gsm_state["discussion_history"]])}
        }
        # 角色私密信息注入
        if me["role"] == "WEREWOLF":
            ctx["private_info"]["werewolf_partners"] = [i for i, info in p.items() if info["role"] == "WEREWOLF"]
        elif me["role"] == "WITCH":
            ctx["private_info"]["witch_potions"] = {"antidote_left": 1 if me["has_heal"] else 0,
                                                    "poison_left": 1 if me["has_poison"] else 0}
            ctx["private_info"]["tonight_victim_hint"] = gsm_state["night_kill_target"]

        return ctx


# ==========================================
# 5. 运行与模拟
# ==========================================
async def ask_agent_action(user_id: str, context: Dict) -> Dict:
    """模拟 Agent：确保投票和杀人时始终提供目标"""
    phase = context["meta"]["phase"]
    others = [p["id"] for p in context["public_state"]["alive_players"] if p["id"] != user_id]
    target = others[0] if others else user_id

    if "voting" in phase: return {"user_id": user_id, "action": "VOTE", "target_id": target}
    if "night" in phase:
        if context["meta"]["role"] == "werewolf": return {"user_id": user_id, "action": "KILL", "target_id": target}
        if context["meta"]["role"] == "witch" and context["private_info"]["witch_potions"]["antidote_left"]:
            victim = context["private_info"].get("tonight_victim_hint")
            if victim: return {"user_id": user_id, "action": "HEAL", "target_id": victim}
    if "discussion" in phase: return {"user_id": user_id, "action": "SPEECH", "content": "投1号！"}
    return {"user_id": user_id, "action": "WAIT"}


async def main():
    engine = GameEngine()

    while not engine.state["game_over"] and engine.state["day_count"] < 10:
        logger.info(f"\n{'=' * 10} 第 {engine.state['day_count']} 天 {'=' * 10}")

        # 1. 夜晚
        await engine.commit_change({"phase": "NIGHT"})
        for pid, p in engine.state["players"].items():
            if p["alive"] and p["role"] in ["WEREWOLF", "SEER", "WITCH"]:
                act = await ask_agent_action(pid, ContextBuilder.build(engine.state, pid))
                await engine.process_action(act)

        # 2. 天亮与结算
        engine.settle_night()
        if engine.check_victory(): break

        # 3. 讨论
        await engine.commit_change({"phase": "DAY_DISCUSSION"})
        alives = [i for i, p in engine.state["players"].items() if p["alive"]]
        for idx, pid in enumerate(alives):
            act = await ask_agent_action(pid,
                                         ContextBuilder.build(engine.state, pid, current_speaker=pid, turn_idx=idx))
            await engine.process_action(act)

        # 4. 投票
        await engine.commit_change({"phase": "DAY_VOTING", "current_votes": {}})
        for pid in alives:
            act = await ask_agent_action(pid, ContextBuilder.build(engine.state, pid))
            await engine.process_action(act)

        engine.settle_votes()
        if engine.check_victory(): break

        await engine.commit_change({"day_count": engine.state["day_count"] + 1})


if __name__ == "__main__":
    asyncio.run(main())



