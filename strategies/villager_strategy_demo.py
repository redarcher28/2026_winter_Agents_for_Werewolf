"""
villager strategy (demo) - single-file design + tests.

Run demo:
  py villager_strategy_demo.py

Run tests:
  py -m pytest -q villager_strategy_demo.py
"""
from __future__ import annotations

import random

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, ValidationError


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
    antidote_left: int = Field(ge=0)
    poison_left: int = Field(ge=0)
    can_self_save: Optional[bool] = None


class PrivateInfo(BaseModel):
    werewolf_partners: Optional[list[str]] = None
    seer_check_history: Optional[list[dict]] = None
    witch_potions: Optional[WitchPotions] = None
    tonight_victim_hint: Optional[str] = None


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


class Inference(BaseModel):
    known_players: list[KnownPlayer] = Field(default_factory=list)


class Constraints(BaseModel):
    allowed_actions: list[str] = Field(default_factory=list)
    max_actions_this_phase: int = Field(default=1, ge=1)
    forbid_targets: list[str] = Field(default_factory=list)


class StrategyContext(BaseModel):
    meta: Meta
    public_state: PublicState
    private_info: PrivateInfo = Field(default_factory=PrivateInfo)
    memory: Memory = Field(default_factory=Memory)
    inference: Inference = Field(default_factory=Inference)
    constraints: Constraints = Field(default_factory=Constraints)


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
    decision_type: Literal["speech"]
    data: SpeechData
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


class VoteDecision(BaseModel):
    decision_type: Literal["vote"]
    data: VoteData
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


class NightActionDecision(BaseModel):
    decision_type: Literal["night_action"]
    data: NightActionData
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


class NoOpDecision(BaseModel):
    decision_type: Literal["no_op"]
    data: NoOpData = Field(default_factory=NoOpData)
    confidence: float = Field(ge=0.0, le=1.0)
    debug: dict = Field(default_factory=dict)


StrategyDecision = Annotated[
    SpeechDecision | VoteDecision | NightActionDecision | NoOpDecision,
    Field(discriminator="decision_type"),
]


class VillagerStrategyEngine:
    SAFE_SPEECH_TEMPLATES = (
        "我是村民，希望找出狼人。",
        "大家要冷静分析投票情况，不要盲目投票。",
        "我会仔细观察每个人的发言和投票行为，找出异常点。",
        "好人要团结一致对抗狼人，不要内斗。",
        "这一轮的发言很重要，大家认真听，细节里能看出身份。",
        "暂时没找到明确的狼人线索，本轮我会谨慎投票。",
        "预言家请尽快报查验，好人需要你的信息带队。",
        "昨晚的死亡情况值得深思，大概率是狼人刀的，女巫有没有救人可以说说。",
        "我没有任何信息，只能跟着大家的分析走，希望别投错好人。"
    )

    def decide(self, context: StrategyContext) -> StrategyDecision:
        alive_player_ids = {p.id for p in context.public_state.alive_players}
        if context.meta.self_player_id not in alive_player_ids:
            return NoOpDecision(
                decision_type="no_op",
                confidence=0.8,
                debug={"reason": "player_dead"}
            )

        if context.meta.role != "villager":
            return NoOpDecision(
                decision_type="no_op",
                confidence=0.8,
                debug={"reason": "not_villager_role"}
            )

        allowed_actions = set(context.constraints.allowed_actions)
        if context.meta.phase == "daytime_discussion":
            if "speech" in allowed_actions:
                return self._make_speech_decision(context)
            else:
                return NoOpDecision(decision_type="no_op", confidence=0.6,
                                    debug={"reason": "speech_not_allowed", "allowed_actions": list(allowed_actions)})
        elif context.meta.phase == "daytime_voting":
            if "vote" in allowed_actions:
                return self._make_vote_decision(context)
            else:
                return NoOpDecision(decision_type="no_op", confidence=0.6,
                                    debug={"reason": "vote_not_allowed", "allowed_actions": list(allowed_actions)})
        else:
            return NoOpDecision(decision_type="no_op", confidence=0.8,
                                debug={"reason": f"phase_not_supported:{context.meta.phase}"})

    def validate(self, context: StrategyContext, decision: StrategyDecision) -> tuple[bool, str]:
        allowed_actions = set(context.constraints.allowed_actions)

        if decision.decision_type not in ["speech", "vote", "no_op"]:
            return False, f"unsupported_decision_type:{decision.decision_type}"

        decision_action_map = {
            "speech": "speech",
            "vote": "vote"
        }

        required_action = decision_action_map.get(decision.decision_type)
        if required_action and required_action not in allowed_actions:
            return False, f"action_not_allowed:{required_action}"

        phase_action_map = {
            "daytime_discussion": "speech",
            "daytime_voting": "vote"
        }
        current_phase = context.meta.phase
        expected_action = phase_action_map.get(current_phase)
        if decision.decision_type != "no_op" and decision.decision_type != expected_action:
            return False, f"action_phase_mismatch:{decision.decision_type} in {current_phase}"

        if decision.decision_type == "speech":
            return self._validate_speech_decision(context, decision)
        elif decision.decision_type == "vote":
            return self._validate_vote_decision(context, decision)

        return True, ""

    def _make_speech_decision(self, context: StrategyContext) -> StrategyDecision:
        speech_content = self._generate_speech_content(context)

        return SpeechDecision(
            decision_type="speech",
            data=SpeechData(
                content=speech_content,
                speech_round=context.meta.day_number,
                turn_order=0
            ),
            confidence=0.7,
            debug={
                "reason": "daytime_discussion_speech",
                "phase": context.meta.phase,
                "day": context.meta.day_number
            }
        )

    def _make_vote_decision(self, context: StrategyContext) -> StrategyDecision:
        target_id, reason = self._select_vote_target(context)

        if target_id is None:
            return NoOpDecision(
                decision_type="no_op",
                confidence=0.8,
                debug={"reason": "no_valid_vote_target"}
            )

        return VoteDecision(
            decision_type="vote",
            data=VoteData(
                target_id=target_id,
                round=context.meta.day_number
            ),
            confidence=0.6,
            debug={
                "reason": reason,
                "phase": context.meta.phase,
                "day": context.meta.day_number
            }
        )

    def _generate_speech_content(self, context: StrategyContext) -> str:
        if not context.memory.memory_summary:
            return random.choice(self.SAFE_SPEECH_TEMPLATES)

        summary = context.memory.memory_summary.lower()
        day_number = context.meta.day_number

        matching_rules = [
            (lambda s: day_number >= 2 and any(word in s for word in ["死亡", "倒牌", "被杀", "出局", "阵亡"]),
             [
                 "昨晚有人倒牌，我们要从刀法和发言里找狼人破绽。",
                 "出局的玩家信息很关键，狼人大概率会冲刀神职，大家要警惕。",
                 "从昨晚的死亡情况来看，狼人应该有明确的战术，我们别被带节奏。"
             ]),
            (lambda s: day_number > 1 and any(word in s for word in ["投票", "弃票", "跟投", "冲票", "投出"]),
             [
                 "上一轮的投票模式很有参考价值，跟风投票的人大概率有问题。",
                 "弃票的玩家需要给出合理理由，好人没必要在关键轮次弃票。",
                 "投票结果能看出阵营倾向，投错票的人本轮要好好表水。"
             ]),
            (lambda s: any(word in s for word in ["预言家", "查验", "查杀", "金水", "悍跳", "报验"]),
             [
                 "预言家的查验信息是好人的明灯，希望真预言家能好好带队。",
                 "查杀和金水都要结合发言验证，谨防狼人悍跳预言家带节奏。",
                 "预言家要尽快报出查验，好人需要你的信息来排狼坑。"
             ]),
            (lambda s: any(word in s for word in ["女巫", "救药", "毒药", "银水", "开药", "毒人"]),
             [
                 "女巫的用药选择关乎好人阵营的胜负，希望能精准救药、谨慎用毒。",
                 "银水大概率是好人，狼人自刀的情况不多，大家可以参考这个信息。",
                 "女巫要藏好身份，别被狼人抿出后刀掉，我们还需要你的保护。"
             ]),
            (lambda s: any(word in s for word in ["狼人", "刀人", "自爆", "倒钩", "聊爆", "狼坑"]),
             [
                 "狼人肯定混在我们中间，发言前后矛盾、逻辑混乱的人就是重点怀疑对象。",
                 "狼人会刻意带节奏冲票，大家别被牵着鼻子走，冷静分析再投票。",
                 "找出狼坑是好人的核心目标，我们要抱团把狼人投出去。"
             ]),
        ]

        for condition, speech_pool in matching_rules:
            if condition(summary):
                return random.choice(speech_pool)

        neutral_speech = [
            "目前场上信息比较少，大家都多说说看法，好人别乱投票。",
            "我暂时没找到明确的狼人线索，本轮会谨慎发言和投票。",
            "好人要团结起来，别被狼人挑拨离间，一步步缩小狼坑范围。"
        ]
        return random.choice(neutral_speech + list(self.SAFE_SPEECH_TEMPLATES))

    def _select_vote_target(self, context: StrategyContext) -> tuple[Optional[str], str]:
        alive_players = context.public_state.alive_players
        self_id = context.meta.self_player_id

        alive_ids = {p.id for p in alive_players}
        candidate_ids = alive_ids - {self_id} if self_id else alive_ids

        if not candidate_ids:
            return None, "no_candidates"

        if context.inference.known_players:
            werewolf_candidates = []
            for kp in context.inference.known_players:
                if kp.player_id in candidate_ids:
                    werewolf_prob = kp.camp_prob.werewolf
                    werewolf_candidates.append((werewolf_prob, kp.player_id))

            if werewolf_candidates:
                werewolf_candidates.sort(key=lambda x: -x[0])
                best_target = werewolf_candidates[0][1]
                return best_target, f"max_werewolf_prob:{werewolf_candidates[0][0]:.2f}"

        target = random.choice(list(candidate_ids))
        return target, "random_selection"

    def _validate_speech_decision(self, context: StrategyContext, decision: SpeechDecision) -> tuple[bool, str]:
        content = decision.data.content

        forbidden_phrases = ["我是狼人", "我是狼", "我承认我是狼"]
        for phrase in forbidden_phrases:
            if phrase in content:
                return False, f"forbidden_phrase:{phrase}"

        if not content.strip():
            return False, "empty_speech"

        if decision.data.speech_round != context.meta.day_number:
            return False, f"speech_round_mismatch:{decision.data.speech_round}"

        return True, ""

    def _validate_vote_decision(self, context: StrategyContext, decision: VoteDecision) -> tuple[bool, str]:
        target_id = decision.data.target_id
        alive_ids = {p.id for p in context.public_state.alive_players}
        self_id = context.meta.self_player_id

        if target_id not in alive_ids:
            return False, "target_not_alive"

        if target_id == self_id:
            return False, "self_vote"

        if target_id in set(context.constraints.forbid_targets):
            return False, "target_forbidden"

        if decision.data.round != context.meta.day_number:
            return False, f"vote_round_mismatch:{decision.data.round}"

        return True, ""

    def get_fallback_decision(self, context: StrategyContext, last_error: dict) -> StrategyDecision:
        if context.meta.phase == "daytime_discussion":
            return SpeechDecision(
                decision_type="speech",
                data=SpeechData(
                    content="我是村民，会认真分析局势。",
                    speech_round=context.meta.day_number,
                    turn_order=0
                ),
                confidence=0.3,
                debug={"reason": "fallback_speech", "last_error": last_error}
            )
        elif context.meta.phase == "daytime_voting":
            alive_players = context.public_state.alive_players
            self_id = context.meta.self_player_id
            for player in alive_players:
                if player.id != self_id:
                    return VoteDecision(
                        decision_type="vote",
                        data=VoteData(
                            target_id=player.id,
                            round=context.meta.day_number
                        ),
                        confidence=0.3,
                        debug={"reason": "fallback_vote", "last_error": last_error}
                    )

        return NoOpDecision(
            decision_type="no_op",
            confidence=0.1,
            debug={"reason": "fallback_no_valid_action", "last_error": last_error}
        )


# -----------------------------
# Demo runner (optional)
# -----------------------------
def _demo_context(phase: str = "daytime_discussion") -> StrategyContext:
    """标准村民演示上下文，可切换发言/投票阶段"""
    return StrategyContext(
        meta=Meta(
            game_id="game_demo",
            agent_id="agent_villager_001",
            role="villager",
            phase=phase,
            day_number=1,
            time_remaining=30,
            self_player_id="player_001",
        ),
        public_state=PublicState(
            alive_players=[
                PlayerPublic(id="player_001", name="我(村民)"),
                PlayerPublic(id="player_002", name="玩家2"),
                PlayerPublic(id="player_003", name="玩家3"),
                PlayerPublic(id="player_004", name="玩家4"),
                PlayerPublic(id="player_005", name="玩家5"),
            ],
            dead_players=[],
        ),
        private_info=PrivateInfo(),
        inference=Inference(
            known_players=[
                KnownPlayer(player_id="player_002", camp_prob=CampProb(good=0.2, werewolf=0.8), tags=["wolf_suspect"]),
                KnownPlayer(player_id="player_003", camp_prob=CampProb(good=0.7, werewolf=0.3), tags=["seer_claim"]),
                KnownPlayer(player_id="player_004", camp_prob=CampProb(good=0.6, werewolf=0.4), tags=[]),
                KnownPlayer(player_id="player_005", camp_prob=CampProb(good=0.1, werewolf=0.9),
                            tags=["talk_inconsistent"]),
            ]
        ),
        constraints=Constraints(
            allowed_actions=["speech", "vote"],
            forbid_targets=[],
        ),
        memory=Memory(memory_summary="Day1: 3号跳预言家，报验4号金水，2号和5号发言逻辑混乱，疑似狼人。"),
    )


def main() -> None:
    """运行村民策略演示：依次展示【白天发言】+【白天投票】决策"""
    # 演示1：白天讨论阶段 - 发言决策
    print("===== 村民策略演示 - 白天讨论阶段 (发言) =====")
    ctx_speech = _demo_context(phase="daytime_discussion")
    strat = VillagerStrategyEngine()
    speech_decision = strat.decide(ctx_speech)
    print(speech_decision.model_dump())

    # 演示2：白天投票阶段 - 投票决策
    print("\n===== 村民策略演示 - 白天投票阶段 (投票) =====")
    ctx_vote = _demo_context(phase="daytime_voting")
    vote_decision = strat.decide(ctx_vote)
    print(vote_decision.model_dump())


if __name__ == "__main__":
    main()


# -----------------------------
# Tests (pytest)
# -----------------------------
def _ctx_base(**overrides) -> StrategyContext:
    """基础测试上下文，支持参数覆盖 """
    import copy
    base_dict = _demo_context().model_dump()
    new_dict = copy.deepcopy(base_dict)

    def deep_update(target: dict, source: dict):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                deep_update(target[k], v)
            else:
                target[k] = v

    deep_update(new_dict, overrides)
    return StrategyContext.model_validate(new_dict)


def test_decide_speech_in_day_discussion() -> None:
    """测试：白天讨论阶段，返回合法发言决策"""
    s = VillagerStrategyEngine()
    d = s.decide(_ctx_base(meta={"phase": "daytime_discussion"}))
    assert d.decision_type == "speech"
    assert len(d.data.content) > 0
    assert d.confidence == 0.7


def test_decide_vote_in_day_voting() -> None:
    """测试：白天投票阶段，返回合法投票决策"""
    s = VillagerStrategyEngine()
    d = s.decide(_ctx_base(meta={"phase": "daytime_voting"}))
    assert d.decision_type == "vote"
    assert d.data.target_id in ["player_002", "player_005"]
    assert d.confidence == 0.6


def test_priority_vote_max_werewolf_prob() -> None:
    """测试：投票优先级 - 优先选择狼人概率最高的玩家"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"phase": "daytime_voting"})
    d = s.decide(ctx)
    assert d.data.target_id == "player_005"
    assert "max_werewolf_prob:0.90" in d.debug.get("reason", "")


def test_no_op_when_not_villager_role() -> None:
    """测试：非村民角色，返回NoOp"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"role": "werewolf"})
    d = s.decide(ctx)
    assert d.decision_type == "no_op"
    assert d.debug["reason"] == "not_villager_role"


def test_no_op_when_phase_not_supported() -> None:
    """测试：非白天阶段(狼人夜)，返回NoOp"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"phase": "werewolf_night"})
    d = s.decide(ctx)
    assert d.decision_type == "no_op"
    assert "phase_not_supported:werewolf_night" in d.debug["reason"]


def test_no_op_when_action_not_allowed() -> None:
    """测试：无对应行动权限，返回NoOp"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"phase": "daytime_discussion"}, constraints={"allowed_actions": ["vote"]})
    d = s.decide(ctx)
    assert d.decision_type == "no_op"
    assert d.debug["reason"] == "speech_not_allowed"


def test_validate_reject_self_vote() -> None:
    """测试：校验规则 - 禁止投自己"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"phase": "daytime_voting"})
    decision = VoteDecision(
        decision_type="vote",
        data=VoteData(target_id="player_001", round=1),
        confidence=0.6
    )
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert reason == "self_vote"


def test_validate_reject_dead_target() -> None:
    """测试：校验规则 - 禁止投死亡玩家"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(
        meta={"phase": "daytime_voting"},  # 修复：指定投票阶段，避免阶段不匹配优先级校验
        public_state={
            "alive_players": [{"id": "player_001"}, {"id": "player_002"}],
            "dead_players": [{"id": "player_005"}]
        }
    )
    decision = VoteDecision(
        decision_type="vote",
        data=VoteData(target_id="player_005", round=1),
        confidence=0.6
    )
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert reason == "target_not_alive"


def test_validate_reject_empty_speech() -> None:
    """测试：校验规则 - 禁止空发言"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"phase": "daytime_discussion"})
    decision = SpeechDecision(
        decision_type="speech",
        data=SpeechData(content="   ", speech_round=1, turn_order=0),
        confidence=0.7
    )
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert reason == "empty_speech"


def test_validate_reject_forbidden_phrase() -> None:
    """测试：校验规则 - 发言含违禁词"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"phase": "daytime_discussion"})
    decision = SpeechDecision(
        decision_type="speech",
        data=SpeechData(content="我是狼人，我摊牌了", speech_round=1, turn_order=0),
        confidence=0.7
    )
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert reason == "forbidden_phrase:我是狼人"


def test_fallback_decision_work() -> None:
    """测试：兜底策略生效"""
    s = VillagerStrategyEngine()
    ctx = _ctx_base(meta={"phase": "daytime_voting"}, inference={"known_players": []})
    fallback = s.get_fallback_decision(ctx, {"reason": "no_target_found"})
    assert fallback.decision_type == "vote"
    assert fallback.confidence == 0.3