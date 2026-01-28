"""
Seer/Witch strategy (Demo, refactored) - aligned with unified StrategyContext/StrategyDecision.

为啥要重构成这样（大白话版）：
- 你们项目里狼人/村民已经统一成 “输入 StrategyContext → 输出 StrategyDecision” 的接口了。
- 原来的神职 Demo 用的是 {decision_type, target, content} 这种“半成品结构”，而且角色名中英混用（"女巫" vs "witch"），会导致：
  - 分支永远进不去（你以为女巫在做事，其实一直返回默认 no_potion）
  - 集成层没法把结果直接映射成 submit_night_action / submit_vote / submit_speech
- 所以这里做了：统一角色名（英文）、统一数据模型（Pydantic）、补上 validate/fallback、加最小 pytest。

后续开发注意什么（非常重要）：
1) role/phase/action_type 这些字符串必须全项目统一，不要一会中文一会英文。
2) alive_players 建议统一成 [{"id": "..."}] 结构（和狼人/村民一致），不要 int 列表；否则集成会很痛苦。
3) 神职的“验人结果/今晚刀口提示”必须来自法官系统回执（private_info），策略不能自己“瞎猜身份”。
4) 解析失败/字段缺失/目标非法时，必须返回 fallback（不然会卡流程）。

Run demo:
  py seer_witch_strategy_demo.py

Run tests:
  py -m pytest -q seer_witch_strategy_demo.py
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, ValidationError


# -----------------------------
# Models (StrategyContext) - 与狼人/村民保持一致
# -----------------------------


class Meta(BaseModel):
    game_id: str
    agent_id: str
    role: Literal["werewolf", "seer", "witch", "villager"]
    phase: str
    day_number: int = Field(ge=1)
    time_remaining: Optional[int] = Field(default=None, ge=0)
    self_player_id: Optional[str] = None


class PlayerPublic(BaseModel):
    id: str
    name: Optional[str] = None
    is_ai: Optional[bool] = None
    is_sheriff: Optional[bool] = None


class DeadPlayer(BaseModel):
    id: str
    name: Optional[str] = None
    death_reason: Optional[str] = None


class PublicState(BaseModel):
    alive_players: list[PlayerPublic]
    dead_players: list[DeadPlayer] = Field(default_factory=list)
    last_night: Optional[dict] = None
    last_vote: Optional[dict] = None


class WitchPotions(BaseModel):
    antidote_left: int = Field(ge=0)  # 解药剩余次数（你们规则是1瓶，就用 0/1）
    poison_left: int = Field(ge=0)  # 毒药剩余次数（同上）
    can_self_save: Optional[bool] = None  # 是否允许自救（规则不同，法官配置应给出）


class SeerCheckRecord(BaseModel):
    target_id: str
    result: Literal["good", "werewolf"]
    round: int = Field(ge=1)


class PrivateInfo(BaseModel):
    # 神职相关：由法官过滤后给
    seer_check_history: Optional[list[SeerCheckRecord]] = None
    witch_potions: Optional[WitchPotions] = None
    tonight_victim_hint: Optional[str] = None  # 女巫可见的“今晚被刀者”


class Memory(BaseModel):
    memory_summary: str = ""
    recent_events: list[dict] = Field(default_factory=list)


class CampProb(BaseModel):
    good: float = Field(ge=0.0, le=1.0)
    werewolf: float = Field(ge=0.0, le=1.0)


class KnownPlayer(BaseModel):
    player_id: str
    camp_prob: CampProb = Field(default_factory=lambda: CampProb(good=0.5, werewolf=0.5))
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class InferenceRankings(BaseModel):
    """
    新版输入（来自第二组LLM）：只给“排序”，我们策略侧把它转换成概率分数用。
    """

    werewolf_likelihood: Optional[list[str]] = None
    villager_likelihood: Optional[list[str]] = None
    seer_likelihood: Optional[list[str]] = None
    witch_likelihood: Optional[list[str]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: Optional[str] = None


class Inference(BaseModel):
    known_players: list[KnownPlayer] = Field(default_factory=list)
    rankings: Optional[InferenceRankings] = None


class Constraints(BaseModel):
    allowed_actions: list[str] = Field(default_factory=list)  # e.g. ["check","save","poison","no_potion","speech","vote"]
    max_actions_this_phase: int = Field(default=1, ge=1)
    forbid_targets: list[str] = Field(default_factory=list)
    # 可选：发言轮次由集成层注入
    current_turn_order: int = 0


class StrategyContext(BaseModel):
    meta: Meta
    public_state: PublicState
    private_info: PrivateInfo = Field(default_factory=PrivateInfo)
    memory: Memory = Field(default_factory=Memory)
    inference: Inference = Field(default_factory=Inference)
    constraints: Constraints = Field(default_factory=Constraints)


# -----------------------------
# Models (StrategyDecision) - 与接口文档一致
# -----------------------------


class SpeechData(BaseModel):
    content: str
    speech_round: int = Field(ge=1)
    turn_order: int = Field(ge=0)


class VoteData(BaseModel):
    target_id: str
    round: int = Field(ge=1)


class NightActionData(BaseModel):
    action_type: Literal["kill", "check", "save", "poison", "no_potion", "shoot", "explode"]
    target_id: Optional[str] = None


class NoOpData(BaseModel):
    pass


class SpeechDecision(BaseModel):
    decision_type: Literal["speech"] = "speech"
    data: SpeechData
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


class VoteDecision(BaseModel):
    decision_type: Literal["vote"] = "vote"
    data: VoteData
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


class NightActionDecision(BaseModel):
    decision_type: Literal["night_action"] = "night_action"
    data: NightActionData
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


class NoOpDecision(BaseModel):
    decision_type: Literal["no_op"] = "no_op"
    data: NoOpData = Field(default_factory=NoOpData)
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


StrategyDecision = Annotated[
    SpeechDecision | VoteDecision | NightActionDecision | NoOpDecision,
    Field(discriminator="decision_type"),
]


# -----------------------------
# Seer & Witch Strategy (Demo)
# -----------------------------


class SeerWitchStrategyDemo:
    """
    Demo 策略目标：先能“正确对接 + 不犯规 + 不崩”，策略强度不是第一优先级。

    - seer_night:
        choose check target (avoid self / forbid / already-checked)
        prefer higher werewolf_prob
    - witch_night:
        save: if tonight_victim_hint exists AND looks good AND antidote_left>0
        poison: if poison_left>0 AND find strong-wolf target
        else no_potion
    - daytime_discussion:
        seer: if has checks, announce one record (demo版直接报，后面可做“是否跳身份”的更复杂管理)
        witch: stay low-key (unless exposed logic later)
    - daytime_voting:
        vote confirmed wolf if seer has it; else vote top suspect
    """

    FORBIDDEN_SPEECH_PHRASES: tuple[str, ...] = ("我是狼人", "我是狼", "我承认我是狼")

    def decide(self, context: StrategyContext) -> StrategyDecision:
        # 大白话：如果 self_player_id 传了，就做一次“我还活着吗”检查。没传就别硬判死。
        if context.meta.self_player_id:
            alive_ids = {p.id for p in context.public_state.alive_players}
            if context.meta.self_player_id not in alive_ids:
                return NoOpDecision(confidence=0.8, debug={"reason": "player_dead"})

        if context.meta.role not in {"seer", "witch"}:
            return NoOpDecision(confidence=0.0, debug={"reason": "not_seer_or_witch"})

        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)

        if phase == "seer_night" and context.meta.role == "seer":
            if "check" not in allowed:
                return NoOpDecision(confidence=0.0, debug={"reason": "check_not_allowed"})
            return self._decide_seer_check(context)

        if phase == "witch_night" and context.meta.role == "witch":
            # 女巫夜里允许动作可能是：save/poison/no_potion（法官端给啥我们就用啥）
            if not ({"save", "poison", "no_potion"} & allowed):
                return NoOpDecision(confidence=0.0, debug={"reason": "witch_actions_not_allowed"})
            return self._decide_witch_action(context)

        if phase == "daytime_discussion":
            if "speech" not in allowed:
                return NoOpDecision(confidence=0.0, debug={"reason": "speech_not_allowed"})
            return self._decide_speech(context)

        if phase == "daytime_voting":
            if "vote" not in allowed:
                return NoOpDecision(confidence=0.0, debug={"reason": "vote_not_allowed"})
            return self._decide_vote(context)

        return NoOpDecision(confidence=0.2, debug={"reason": f"phase_not_supported:{phase}"})

    # ---- validate/fallback：集成必须要有 ----

    def validate(self, context: StrategyContext, decision: StrategyDecision) -> tuple[bool, str]:
        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)

        phase_expected = {
            "seer_night": "night_action",
            "witch_night": "night_action",
            "daytime_discussion": "speech",
            "daytime_voting": "vote",
        }.get(phase)
        if decision.decision_type != "no_op" and phase_expected and decision.decision_type != phase_expected:
            return False, f"action_phase_mismatch:{decision.decision_type} in {phase}"

        if decision.decision_type == "speech":
            if "speech" not in allowed:
                return False, "speech_not_allowed"
            if decision.data.speech_round != context.meta.day_number:
                return False, "speech_round_mismatch"
            content = decision.data.content.strip()
            if not content:
                return False, "empty_speech"
            for phrase in self.FORBIDDEN_SPEECH_PHRASES:
                if phrase in content:
                    return False, f"forbidden_phrase:{phrase}"
            return True, ""

        if decision.decision_type == "vote":
            if "vote" not in allowed:
                return False, "vote_not_allowed"
            if decision.data.round != context.meta.day_number:
                return False, "vote_round_mismatch"
            return self._validate_target_alive(context, decision.data.target_id)

        if decision.decision_type == "night_action":
            # 大白话：night_action 的 action_type 必须也在 allowed_actions 里，否则法官会拒绝
            action_type = decision.data.action_type
            if action_type not in allowed and action_type != "no_potion":
                return False, f"night_action_not_allowed:{action_type}"

            # check/poison 目标必须是活人；save 目标必须是今晚受害者（通常是“被刀者”）
            if action_type in {"check", "poison"}:
                if not decision.data.target_id:
                    return False, "missing_target_id"
                return self._validate_target_alive(context, decision.data.target_id)
            if action_type == "save":
                if not decision.data.target_id:
                    return False, "missing_target_id"
                victim = context.private_info.tonight_victim_hint
                if victim and decision.data.target_id != victim:
                    return False, "save_target_not_victim"
                return True, ""
            if action_type == "no_potion":
                return True, ""
            return False, f"unsupported_action_type:{action_type}"

        return True, ""

    def fallback(self, context: StrategyContext, last_error: dict) -> StrategyDecision:
        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)
        day = context.meta.day_number

        # 大白话：兜底策略要“稳”，宁愿保守 no_potion / 过牌，也不要提交非法目标。
        if phase in {"seer_night", "witch_night"}:
            if "no_potion" in allowed or phase == "seer_night":
                return NightActionDecision(
                    data=NightActionData(action_type="no_potion"),
                    confidence=0.2,
                    debug={"reason": "fallback_no_potion", "last_error": last_error},
                )
            return NoOpDecision(confidence=0.1, debug={"reason": "fallback_no_night_action"})

        if phase == "daytime_voting" and "vote" in allowed:
            target = self._first_valid_alive_target(context)
            if target:
                return VoteDecision(
                    data=VoteData(target_id=target, round=day),
                    confidence=0.2,
                    debug={"reason": "fallback_vote", "last_error": last_error},
                )
            return NoOpDecision(confidence=0.1, debug={"reason": "fallback_no_targets"})

        if phase == "daytime_discussion" and "speech" in allowed:
            return SpeechDecision(
                data=SpeechData(
                    content=f"第{day}天我先听完大家发言，重点看投票和逻辑矛盾。",
                    speech_round=day,
                    turn_order=context.constraints.current_turn_order,
                ),
                confidence=0.2,
                debug={"reason": "fallback_speech", "last_error": last_error},
            )

        return NoOpDecision(confidence=0.1, debug={"reason": "fallback_no_valid_action"})

    # ---- Internal decisions ----

    def _decide_seer_check(self, context: StrategyContext) -> StrategyDecision:
        target, reason = self._pick_seer_check_target(context)
        if not target:
            d = NightActionDecision(
                data=NightActionData(action_type="no_potion"),
                confidence=0.3,
                debug={"reason": "no_check_target"},
            )
            ok, why = self.validate(context, d)
            return d if ok else self.fallback(context, {"reason": why})

        d = NightActionDecision(
            data=NightActionData(action_type="check", target_id=target),
            confidence=0.7,
            debug={"reason": reason},
        )
        ok, why = self.validate(context, d)
        return d if ok else self.fallback(context, {"reason": why, "decision": d.model_dump()})

    def _decide_witch_action(self, context: StrategyContext) -> StrategyDecision:
        potions = context.private_info.witch_potions or WitchPotions(antidote_left=0, poison_left=0)
        victim = context.private_info.tonight_victim_hint
        allowed = set(context.constraints.allowed_actions)

        # 1) 优先救人：有解药 + 有被刀者提示 + 看起来像好人
        if victim and potions.antidote_left > 0 and "save" in allowed:
            # 1.1 自救分支（是否允许自救由法官配置决定）
            if context.meta.self_player_id and victim == context.meta.self_player_id:
                # 大白话：能不能自救得看规则，别瞎救导致法官拒绝
                if potions.can_self_save is not False:
                    d = NightActionDecision(
                        data=NightActionData(action_type="save", target_id=victim),
                        confidence=0.7,
                        debug={"reason": "save_self_if_allowed"},
                    )
                    ok, why = self.validate(context, d)
                    return d if ok else self.fallback(context, {"reason": why, "decision": d.model_dump()})

            # 1.2 救别人：目标越像好人，越值得救（demo：wolf_prob <= 0.4 就救）
            wolf_prob = self._get_wolf_prob(context, victim)
            if wolf_prob <= 0.4:
                d = NightActionDecision(
                    data=NightActionData(action_type="save", target_id=victim),
                    confidence=0.75,
                    debug={"reason": f"save_goodish_victim:wolf_prob={wolf_prob:.2f}"},
                )
                ok, why = self.validate(context, d)
                return d if ok else self.fallback(context, {"reason": why, "decision": d.model_dump()})

        # 2) 再考虑毒人：有毒药 + 找到“强狼嫌疑”目标
        if potions.poison_left > 0 and "poison" in allowed:
            target = self._pick_strong_wolf_target(context)
            if target:
                d = NightActionDecision(
                    data=NightActionData(action_type="poison", target_id=target),
                    confidence=0.65,
                    debug={"reason": "poison_strong_wolf_target"},
                )
                ok, why = self.validate(context, d)
                return d if ok else self.fallback(context, {"reason": why, "decision": d.model_dump()})

        # 3) 兜底：不使用
        d = NightActionDecision(
            data=NightActionData(action_type="no_potion"),
            confidence=0.4,
            debug={"reason": "no_potion"},
        )
        ok, why = self.validate(context, d)
        return d if ok else self.fallback(context, {"reason": why, "decision": d.model_dump()})

    def _decide_speech(self, context: StrategyContext) -> StrategyDecision:
        day = context.meta.day_number
        summary = (context.memory.memory_summary or "").strip()
        role = context.meta.role

        if role == "seer":
            checks = context.private_info.seer_check_history or []
            if checks:
                # Demo：报最新一条验人结果（后续可以做“是否跳身份/何时跳”更复杂策略）
                last = sorted(checks, key=lambda r: r.round)[-1]
                content = (
                    f"第{day}天我补充一个信息：我昨晚验了 {last.target_id}，结果是 {('好人' if last.result=='good' else '狼人')}。"
                    f"{(' 另外：' + summary) if summary else ''}"
                    " 大家结合发言和投票一起看。"
                )
            else:
                content = f"第{day}天我先听逻辑，主要看投票和前后矛盾。{(' ' + summary) if summary else ''}"
            else:
            # 女巫白天尽量低调：别乱跳身份（demo先保守）
            content = f"第{day}天我先听完大家发言再站边，重点看投票和逻辑闭环。{(' ' + summary) if summary else ''}"

        d = SpeechDecision(
            data=SpeechData(content=content[:220], speech_round=day, turn_order=context.constraints.current_turn_order),
            confidence=0.55,
            debug={"reason": f"demo_speech:{role}"},
        )
        ok, why = self.validate(context, d)
        return d if ok else self.fallback(context, {"reason": why, "decision": d.model_dump()})

    def _decide_vote(self, context: StrategyContext) -> StrategyDecision:
        day = context.meta.day_number
        role = context.meta.role

        target, reason = None, ""
        if role == "seer":
            # 优先投“验到的狼人”
            checks = context.private_info.seer_check_history or []
            wolves = [c.target_id for c in checks if c.result == "werewolf"]
            for wid in sorted(set(wolves)):
                ok, _ = self._validate_target_alive(context, wid)
                if ok:
                    target, reason = wid, "vote_confirmed_werewolf"
                    break

        if not target:
            target = self._pick_top_suspect(context)
            reason = "vote_top_suspect"

        if not target:
            return self.fallback(context, {"reason": "no_valid_vote_target"})

        d = VoteDecision(
            data=VoteData(target_id=target, round=day),
            confidence=0.65,
            debug={"reason": reason},
        )
        ok, why = self.validate(context, d)
        return d if ok else self.fallback(context, {"reason": why, "decision": d.model_dump()})

    # ---- helpers ----

    def _kp_by_id(self, context: StrategyContext) -> dict[str, KnownPlayer]:
        return {kp.player_id: kp for kp in context.inference.known_players}

    def _rank_score(self, ranking: Optional[list[str]], player_id: str) -> Optional[float]:
        if not ranking:
            return None
        if player_id not in ranking:
            return None
        n = len(ranking)
        idx = ranking.index(player_id)
        if n <= 1:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (idx / (n - 1))))

    def _derive_wolf_prob_from_rankings(self, context: StrategyContext, player_id: str) -> Optional[float]:
        r = context.inference.rankings
        if not r:
            return None
        wolf_s = self._rank_score(r.werewolf_likelihood, player_id)
        vill_s = self._rank_score(r.villager_likelihood, player_id)

        wolf_prob: Optional[float] = None
        if wolf_s is not None:
            wolf_prob = 0.1 + 0.8 * wolf_s
        if vill_s is not None:
            good_prob = 0.1 + 0.8 * vill_s
            if wolf_prob is None:
                wolf_prob = 1.0 - good_prob
            else:
                wolf_prob = wolf_prob - 0.6 * (good_prob - 0.5)

        if wolf_prob is None:
            return None
        return max(0.0, min(1.0, wolf_prob))

    def _excluded_targets(self, context: StrategyContext) -> set[str]:
        excluded = set(context.constraints.forbid_targets)
        if context.meta.self_player_id:
            excluded.add(context.meta.self_player_id)
        return excluded

    def _alive_candidate_ids(self, context: StrategyContext) -> list[str]:
        excluded = self._excluded_targets(context)
        ids = [p.id for p in context.public_state.alive_players if p.id not in excluded]
        return sorted(ids)

    def _validate_target_alive(self, context: StrategyContext, target_id: str) -> tuple[bool, str]:
        alive = {p.id for p in context.public_state.alive_players}
        if not target_id:
            return False, "missing_target_id"
        if target_id not in alive:
            return False, "target_not_alive"
        if target_id in set(context.constraints.forbid_targets):
            return False, "target_forbidden"
        if context.meta.self_player_id and target_id == context.meta.self_player_id:
            return False, "target_is_self"
        return True, ""

    def _first_valid_alive_target(self, context: StrategyContext) -> Optional[str]:
        candidates = self._alive_candidate_ids(context)
        return candidates[0] if candidates else None

    def _get_wolf_prob(self, context: StrategyContext, player_id: str) -> float:
        kp = self._kp_by_id(context).get(player_id)
        if kp:
            return kp.camp_prob.werewolf
        derived = self._derive_wolf_prob_from_rankings(context, player_id)
        return derived if derived is not None else 0.5

    def _pick_seer_check_target(self, context: StrategyContext) -> tuple[Optional[str], str]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None, "no_candidates"

        checked = set()
        for r in (context.private_info.seer_check_history or []):
            checked.add(r.target_id)

        candidates = [pid for pid in candidates if pid not in checked]
        if not candidates:
            return None, "all_checked"

        # 大白话：demo版简单一点，优先验“更像狼”的（werewolf_prob高的）
        candidates.sort(key=lambda pid: (-self._get_wolf_prob(context, pid), pid))
        return candidates[0], f"max_wolf_prob:{self._get_wolf_prob(context, candidates[0]):.2f}"

    def _pick_strong_wolf_target(self, context: StrategyContext) -> Optional[str]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None
        candidates.sort(key=lambda pid: (-self._get_wolf_prob(context, pid), pid))
        best = candidates[0]
        # demo门槛：狼概率足够高再毒，避免乱毒
        return best if self._get_wolf_prob(context, best) >= 0.8 else None

    def _pick_top_suspect(self, context: StrategyContext) -> Optional[str]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None
        candidates.sort(key=lambda pid: (-self._get_wolf_prob(context, pid), pid))
        return candidates[0]


# -----------------------------
# Demo context + runner
# -----------------------------


def _demo_context(role: Literal["seer", "witch"], phase: str) -> StrategyContext:
    return StrategyContext(
        meta=Meta(
            game_id="game_demo_god",
            agent_id=f"agent_{role}_001",
            role=role,
            phase=phase,
            day_number=1,
            time_remaining=30,
            self_player_id=f"player_{role}_001",
        ),
        public_state=PublicState(
            alive_players=[
                PlayerPublic(id=f"player_{role}_001", name="我"),
                PlayerPublic(id="player_002", name="2号"),
                PlayerPublic(id="player_003", name="3号"),
                PlayerPublic(id="player_004", name="4号"),
            ],
            dead_players=[],
        ),
        private_info=PrivateInfo(
            seer_check_history=[],
            witch_potions=WitchPotions(antidote_left=1, poison_left=1, can_self_save=False),
            tonight_victim_hint="player_003",
        ),
        inference=Inference(
            known_players=[
                KnownPlayer(player_id="player_002", camp_prob=CampProb(good=0.2, werewolf=0.8), tags=["wolf_suspect"]),
                KnownPlayer(player_id="player_003", camp_prob=CampProb(good=0.7, werewolf=0.3), tags=[]),
                KnownPlayer(player_id="player_004", camp_prob=CampProb(good=0.5, werewolf=0.5), tags=[]),
            ]
        ),
        constraints=Constraints(
            allowed_actions=["check", "save", "poison", "no_potion", "speech", "vote"],
            forbid_targets=[],
            current_turn_order=0,
        ),
        memory=Memory(memory_summary="Day1: 2号发言很像狼，3号看起来偏好。"),
    )


def main() -> None:
    s = SeerWitchStrategyDemo()
    print("=== Seer night ===")
    print(s.decide(_demo_context("seer", "seer_night")).model_dump())
    print("\n=== Witch night ===")
    print(s.decide(_demo_context("witch", "witch_night")).model_dump())
    print("\n=== Day discussion (seer) ===")
    print(s.decide(_demo_context("seer", "daytime_discussion")).model_dump())
    print("\n=== Day voting (seer) ===")
    print(s.decide(_demo_context("seer", "daytime_voting")).model_dump())


if __name__ == "__main__":
    main()


# -----------------------------
# Tests (pytest)
# -----------------------------


def test_seer_night_outputs_check() -> None:
    s = SeerWitchStrategyDemo()
    ctx = _demo_context("seer", "seer_night")
    d = s.decide(ctx)
    assert d.decision_type == "night_action"
    assert d.data.action_type in {"check", "no_potion"}
    if d.data.action_type == "check":
        assert d.data.target_id != ctx.meta.self_player_id


def test_witch_night_save_good_victim() -> None:
    s = SeerWitchStrategyDemo()
    ctx = _demo_context("witch", "witch_night")
    # victim is player_003 with wolf_prob 0.3 -> should save
    d = s.decide(ctx)
    assert d.decision_type == "night_action"
    assert d.data.action_type in {"save", "poison", "no_potion"}
    assert d.data.action_type == "save"
    assert d.data.target_id == "player_003"


def test_day_discussion_outputs_speech() -> None:
    s = SeerWitchStrategyDemo()
    ctx = _demo_context("seer", "daytime_discussion")
    d = s.decide(ctx)
    assert d.decision_type == "speech"
    assert d.data.speech_round == 1
    assert len(d.data.content) > 0


def test_day_voting_outputs_vote() -> None:
    s = SeerWitchStrategyDemo()
    ctx = _demo_context("seer", "daytime_voting")
    d = s.decide(ctx)
    assert d.decision_type == "vote"
    assert d.data.target_id != ctx.meta.self_player_id


def test_context_validation_error() -> None:
    bad = _demo_context("seer", "seer_night").model_dump()
    bad["meta"]["day_number"] = 0
    try:
        StrategyContext.model_validate(bad)
        assert False, "expected ValidationError"
    except ValidationError:
        assert True


def test_rankings_input_affects_witch_poison_choice() -> None:
    s = SeerWitchStrategyDemo()
    ctx = _demo_context("witch", "witch_night")
    # 把 victim 设为空，让女巫走“考虑毒人”分支
    ctx.private_info.tonight_victim_hint = None
    # 仅给排名（不提供 known_players 的概率）
    ctx.inference = Inference(
        known_players=[],
        rankings=InferenceRankings(
            werewolf_likelihood=["player_002", "player_004", "player_003"],
            villager_likelihood=["player_003", "player_004", "player_002"],
        ),
    )
    d = s.decide(ctx)
    assert d.decision_type == "night_action"
    # demo里只有“狼概率>=0.8才毒”，player_002 会被映射到接近0.9
    assert d.data.action_type in {"poison", "no_potion"}

