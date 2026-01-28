"""
Werewolf strategy (Alpha 1.0) - single-file design + tests.

Goals vs demo:
- Cover key phases: werewolf_night (kill), daytime_discussion (speech), daytime_voting (vote)
- Strong validation + deterministic fallback
- Avoid partners / forbidden targets / self

Run demo:
  py werewolf_strategy_alpha1.py

Run tests:
  py -m pytest -q werewolf_strategy_alpha1.py
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, ValidationError


# -----------------------------
# Models (StrategyContext)
# -----------------------------


class Meta(BaseModel):
    game_id: str
    agent_id: str
    role: Literal["werewolf", "seer", "witch", "villager"]
    phase: str
    day_number: int = Field(ge=1)
    time_remaining: Optional[int] = Field(default=None, ge=0)
    self_player_id: Optional[str] = None  # strongly recommended to set


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


class InferenceRankings(BaseModel):
    """
    新版输入（来自第二组LLM）：不再给每个玩家的数值概率，而是给“排序”。
    约定：列表从左到右 = 越可能（更像）该身份/阵营。
    - werewolf_likelihood：越靠前越像狼人（wolf_prob 越高）
    - villager_likelihood：越靠前越像好人（wolf_prob 越低 / good_prob 越高）
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
    allowed_actions: list[str] = Field(default_factory=list)  # e.g. ["speech","vote","kill"]
    max_actions_this_phase: int = Field(default=1, ge=1)
    forbid_targets: list[str] = Field(default_factory=list)
    # 可选：发言轮次/顺序由集成层（通常来自法官系统）注入
    current_turn_order: int = 0


class StrategyContext(BaseModel):
    meta: Meta
    public_state: PublicState
    private_info: PrivateInfo = Field(default_factory=PrivateInfo)
    memory: Memory = Field(default_factory=Memory)
    inference: Inference = Field(default_factory=Inference)
    constraints: Constraints = Field(default_factory=Constraints)


# -----------------------------
# Models (StrategyDecision)
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


# -----------------------------
# Werewolf strategy (Alpha 1.0)
# -----------------------------


class WerewolfStrategyAlpha1:
    """
    Alpha 1.0 werewolf strategy:

    - werewolf_night:
        prioritize seer/witch signals (tags) > kill most-likely-good alive
    - daytime_discussion:
        produce safe speech that subtly pushes a chosen scapegoat (high werewolf_prob),
        without claiming wolf / explicit self-exposure phrases.
    - daytime_voting:
        vote consistent with scapegoat selection (high werewolf_prob), avoid partners.

    Notes:
    - This is intentionally deterministic (for stable tests / integration).
    - 'allowed_actions' uses: "speech", "vote", "kill".
    """

    # tags interpreted from inference.known_players[*].tags
    NIGHT_PRIORITY_TAGS: tuple[str, ...] = ("seer_confirmed", "witch_confirmed", "seer_claim", "witch_claim")
    DAY_SUSPECT_TAGS: tuple[str, ...] = ("wolf_suspect", "talk_inconsistent", "vote_inconsistent")

    FORBIDDEN_SPEECH_PHRASES: tuple[str, ...] = ("我是狼人", "我是狼", "我承认我是狼")

    def decide(self, context: StrategyContext) -> StrategyDecision:
        # Basic liveness check (if self_player_id is known)
        if context.meta.self_player_id:
            alive = {p.id for p in context.public_state.alive_players}
            if context.meta.self_player_id not in alive:
                return NoOpDecision(decision_type="no_op", confidence=0.8, debug={"reason": "player_dead"})

        if context.meta.role != "werewolf":
            return NoOpDecision(decision_type="no_op", confidence=0.0, debug={"reason": "not_werewolf"})

        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)

        if phase == "werewolf_night":
            if "kill" not in allowed:
                return NoOpDecision(
                    decision_type="no_op",
                    confidence=0.0,
                    debug={"reason": "kill_not_allowed", "allowed_actions": sorted(allowed)},
                )
            return self._decide_kill(context)

        if phase == "daytime_discussion":
            if "speech" not in allowed:
                return NoOpDecision(
                    decision_type="no_op",
                    confidence=0.0,
                    debug={"reason": "speech_not_allowed", "allowed_actions": sorted(allowed)},
                )
            return self._decide_speech(context)

        if phase == "daytime_voting":
            if "vote" not in allowed:
                return NoOpDecision(
                    decision_type="no_op",
                    confidence=0.0,
                    debug={"reason": "vote_not_allowed", "allowed_actions": sorted(allowed)},
                )
            return self._decide_vote(context)

        return NoOpDecision(decision_type="no_op", confidence=0.2, debug={"reason": f"phase_not_supported:{phase}"})

    def validate(self, context: StrategyContext, decision: StrategyDecision) -> tuple[bool, str]:
        allowed = set(context.constraints.allowed_actions)
        phase = context.meta.phase

        # Phase-action compatibility
        phase_expected = {
            "werewolf_night": "night_action",
            "daytime_discussion": "speech",
            "daytime_voting": "vote",
        }.get(phase)
        if decision.decision_type != "no_op" and phase_expected and decision.decision_type != phase_expected:
            return False, f"action_phase_mismatch:{decision.decision_type} in {phase}"

        if decision.decision_type == "speech":
            if "speech" not in allowed:
                return False, "speech_not_allowed"
            if decision.data.speech_round != context.meta.day_number:
                return False, f"speech_round_mismatch:{decision.data.speech_round}"
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
                return False, f"vote_round_mismatch:{decision.data.round}"
            return self._validate_target(context, decision.data.target_id)

        if decision.decision_type == "night_action":
            if decision.data.action_type != "kill":
                return False, f"unsupported_night_action:{decision.data.action_type}"
            if "kill" not in allowed:
                return False, "kill_not_allowed"
            if not decision.data.target_id:
                return False, "missing_target_id"
            return self._validate_target(context, decision.data.target_id)

        return True, ""

    def fallback(self, context: StrategyContext, last_error: dict) -> StrategyDecision:
        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)

        if phase == "werewolf_night" and "kill" in allowed:
            target = self._first_valid_target(context)
            if target:
                return NightActionDecision(
                    decision_type="night_action",
                    data=NightActionData(action_type="kill", target_id=target),
                    confidence=0.3,
                    debug={"reason": "fallback_kill", "last_error": last_error},
                )
            return NoOpDecision(decision_type="no_op", confidence=0.1, debug={"reason": "fallback_no_targets"})

        if phase == "daytime_voting" and "vote" in allowed:
            target = self._first_valid_target(context)
            if target:
                return VoteDecision(
                    decision_type="vote",
                    data=VoteData(target_id=target, round=context.meta.day_number),
                    confidence=0.3,
                    debug={"reason": "fallback_vote", "last_error": last_error},
                )
            return NoOpDecision(decision_type="no_op", confidence=0.1, debug={"reason": "fallback_no_targets"})

        if phase == "daytime_discussion" and "speech" in allowed:
            return SpeechDecision(
                decision_type="speech",
                data=SpeechData(
                    content="我先听完大家发言，再结合投票和逻辑给出我的怀疑点。",
                    speech_round=context.meta.day_number,
                    turn_order=0,
                ),
                confidence=0.3,
                debug={"reason": "fallback_speech", "last_error": last_error},
            )

        return NoOpDecision(decision_type="no_op", confidence=0.1, debug={"reason": "fallback_no_valid_action"})

    # ---- internal decisions ----

    def _decide_kill(self, context: StrategyContext) -> StrategyDecision:
        target_id, reason, confidence = self._pick_kill_target(context)
        if not target_id:
            return self.fallback(context, {"reason": "no_valid_kill_target"})
        decision = NightActionDecision(
            decision_type="night_action",
            data=NightActionData(action_type="kill", target_id=target_id),
            confidence=confidence,
            debug={"reason": reason},
        )
        ok, why = self.validate(context, decision)
        return decision if ok else self.fallback(context, {"reason": why, "decision": decision.model_dump()})

    def _decide_vote(self, context: StrategyContext) -> StrategyDecision:
        target_id, reason, confidence = self._pick_day_scapegoat(context)
        if not target_id:
            return self.fallback(context, {"reason": "no_valid_vote_target"})
        decision = VoteDecision(
            decision_type="vote",
            data=VoteData(target_id=target_id, round=context.meta.day_number),
            confidence=confidence,
            debug={"reason": reason},
        )
        ok, why = self.validate(context, decision)
        return decision if ok else self.fallback(context, {"reason": why, "decision": decision.model_dump()})

    def _decide_speech(self, context: StrategyContext) -> StrategyDecision:
        target_id, reason, _ = self._pick_day_scapegoat(context)
        summary = (context.memory.memory_summary or "").strip()
        day = context.meta.day_number

        if target_id:
            content = (
                f"第{day}天我觉得场上信息要围绕投票和发言逻辑来验。"
                f"{(' 结合已有信息：' + summary) if summary else ''}"
                f" 我目前更关注 {target_id} 的发言/投票一致性，建议大家听他/她的逻辑闭环。"
            )
        else:
            content = (
                f"第{day}天信息还不算充分，大家先把关键逻辑讲清楚。"
                f"{(' 结合已有信息：' + summary) if summary else ''}"
                " 我会重点看投票流向和前后矛盾点。"
            )

        decision = SpeechDecision(
            decision_type="speech",
            data=SpeechData(content=content, speech_round=context.meta.day_number, turn_order=0),
            confidence=0.55,
            debug={"reason": f"push_scapegoat:{reason}" if target_id else "neutral_speech"},
        )
        ok, why = self.validate(context, decision)
        return decision if ok else self.fallback(context, {"reason": why, "decision": decision.model_dump()})

    # ---- target selection ----

    def _excluded_targets(self, context: StrategyContext) -> set[str]:
        excluded = set(context.constraints.forbid_targets)
        excluded |= set(context.private_info.werewolf_partners or [])
        if context.meta.self_player_id:
            excluded.add(context.meta.self_player_id)
        return excluded

    def _alive_candidate_ids(self, context: StrategyContext) -> list[str]:
        alive_ids = [p.id for p in context.public_state.alive_players]
        excluded = self._excluded_targets(context)
        candidates = [pid for pid in alive_ids if pid not in excluded]
        return sorted(candidates)  # deterministic

    def _validate_target(self, context: StrategyContext, target_id: str) -> tuple[bool, str]:
        alive = {p.id for p in context.public_state.alive_players}
        if target_id not in alive:
            return False, "target_not_alive"
        if target_id in set(context.constraints.forbid_targets):
            return False, "target_forbidden"
        if target_id in set(context.private_info.werewolf_partners or []):
            return False, "target_is_partner"
        if context.meta.self_player_id and target_id == context.meta.self_player_id:
            return False, "target_is_self"
        return True, ""

    def _kp_by_id(self, context: StrategyContext) -> dict[str, KnownPlayer]:
        return {kp.player_id: kp for kp in context.inference.known_players}

    def _rank_score(self, ranking: Optional[list[str]], player_id: str) -> Optional[float]:
        """
        把“排名列表”映射成 [0,1] 的分数：
        - 排名越靠前，分数越接近 1
        - 排名越靠后，分数越接近 0
        """
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
        """
        核心改动：第二组只给“排序”，我们在这里统一把排序变成可用的 wolf_prob。
        规则（先简单可控，后续可以调参）：
        - 若有 werewolf_likelihood：越靠前 wolf_prob 越高（映射到 [0.1, 0.9]）
        - 若有 villager_likelihood：越靠前越像好人，会拉低 wolf_prob
        """
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
                # 没给“像狼排序”，那就从“像好人排序”反推
                wolf_prob = 1.0 - good_prob
            else:
                # 同时给了两种排序：稍微融合一下（好人排序高 -> wolf_prob 往下拉）
                wolf_prob = wolf_prob - 0.6 * (good_prob - 0.5)

        if wolf_prob is None:
            return None

        return max(0.0, min(1.0, wolf_prob))

    def _get_wolf_prob(self, context: StrategyContext, player_id: str) -> float:
        kp = self._kp_by_id(context).get(player_id)
        if kp:
            return kp.camp_prob.werewolf
        derived = self._derive_wolf_prob_from_rankings(context, player_id)
        return derived if derived is not None else 0.5

    def _get_good_prob(self, context: StrategyContext, player_id: str) -> float:
        kp = self._kp_by_id(context).get(player_id)
        if kp:
            return kp.camp_prob.good
        wolf_prob = self._derive_wolf_prob_from_rankings(context, player_id)
        if wolf_prob is None:
            return 0.5
        return max(0.0, min(1.0, 1.0 - wolf_prob))

    def _pick_kill_target(self, context: StrategyContext) -> tuple[Optional[str], str, float]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None, "no_candidates", 0.0

        kp_by_id = self._kp_by_id(context)

        # 1) Priority tags for night: likely seer/witch
        for tag in self.NIGHT_PRIORITY_TAGS:
            tagged = [pid for pid in candidates if (kp_by_id.get(pid) and tag in kp_by_id[pid].tags)]
            if tagged:
                return tagged[0], f"priority_tag:{tag}", 0.8

        # 2) Otherwise: kill most-likely-good (wolves want to remove strong good)
        scored: list[tuple[float, str]] = []
        for pid in candidates:
            good_prob = self._get_good_prob(context, pid)
            scored.append((good_prob, pid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        best_good_prob, best_pid = scored[0]
        return best_pid, f"max_good_prob:{best_good_prob:.2f}", 0.65

    def _pick_day_scapegoat(self, context: StrategyContext) -> tuple[Optional[str], str, float]:
        """
        For daytime speech/vote: pick a non-partner scapegoat to push.
        Prefer high werewolf_prob targets (easy to justify), with suspicious tags as tie-breaker.
        """
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None, "no_candidates", 0.0

        kp_by_id = self._kp_by_id(context)

        scored: list[tuple[float, int, str]] = []
        for pid in candidates:
            kp = kp_by_id.get(pid)
            wolf_prob = self._get_wolf_prob(context, pid)
            tag_bonus = 0
            if kp:
                tag_bonus = sum(1 for t in kp.tags if t in self.DAY_SUSPECT_TAGS)
            scored.append((wolf_prob, tag_bonus, pid))

        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        wolf_prob, tag_bonus, best = scored[0]
        return best, f"max_werewolf_prob:{wolf_prob:.2f};tag_bonus:{tag_bonus}", 0.6

    def _first_valid_target(self, context: StrategyContext) -> Optional[str]:
        candidates = self._alive_candidate_ids(context)
        return candidates[0] if candidates else None


# -----------------------------
# Demo runner
# -----------------------------


def _demo_context(phase: str) -> StrategyContext:
    return StrategyContext(
        meta=Meta(
            game_id="game_demo",
            agent_id="agent_wolf_001",
            role="werewolf",
            phase=phase,
            day_number=1,
            time_remaining=30,
            self_player_id="player_001",
        ),
        public_state=PublicState(
            alive_players=[
                PlayerPublic(id="player_001", name="我(狼)"),
                PlayerPublic(id="player_002", name="玩家2"),
                PlayerPublic(id="player_003", name="玩家3"),
                PlayerPublic(id="player_004", name="玩家4(狼队友)"),
                PlayerPublic(id="player_005", name="玩家5"),
            ],
            dead_players=[],
        ),
        private_info=PrivateInfo(werewolf_partners=["player_004"]),
        inference=Inference(
            known_players=[
                KnownPlayer(player_id="player_002", camp_prob=CampProb(good=0.7, werewolf=0.3), tags=["seer_claim"]),
                KnownPlayer(player_id="player_003", camp_prob=CampProb(good=0.6, werewolf=0.4), tags=["wolf_suspect"]),
                KnownPlayer(player_id="player_005", camp_prob=CampProb(good=0.4, werewolf=0.6), tags=["talk_inconsistent"]),
            ]
        ),
        constraints=Constraints(allowed_actions=["speech", "vote", "kill"], forbid_targets=[]),
        memory=Memory(memory_summary="Day1: 2号跳预言家，5号发言前后矛盾。"),
    )


def main() -> None:
    s = WerewolfStrategyAlpha1()

    print("===== Werewolf Alpha1 - daytime_discussion =====")
    print(s.decide(_demo_context("daytime_discussion")).model_dump())

    print("\n===== Werewolf Alpha1 - daytime_voting =====")
    print(s.decide(_demo_context("daytime_voting")).model_dump())

    print("\n===== Werewolf Alpha1 - werewolf_night =====")
    print(s.decide(_demo_context("werewolf_night")).model_dump())


if __name__ == "__main__":
    main()


# -----------------------------
# Tests (pytest)
# -----------------------------


def _ctx_base(**overrides) -> StrategyContext:
    base = _demo_context("daytime_discussion").model_dump()

    # simple deep merge for overrides
    import copy

    d = copy.deepcopy(base)

    def deep_update(t: dict, s: dict) -> None:
        for k, v in s.items():
            if isinstance(v, dict) and isinstance(t.get(k), dict):
                deep_update(t[k], v)
            else:
                t[k] = v

    deep_update(d, overrides)
    return StrategyContext.model_validate(d)


def test_discussion_returns_speech() -> None:
    s = WerewolfStrategyAlpha1()
    ctx = _ctx_base(meta={"phase": "daytime_discussion"})
    d = s.decide(ctx)
    assert d.decision_type == "speech"
    assert d.data.speech_round == 1
    assert len(d.data.content) > 0


def test_voting_returns_vote_and_avoids_partner() -> None:
    s = WerewolfStrategyAlpha1()
    ctx = _ctx_base(meta={"phase": "daytime_voting"})
    d = s.decide(ctx)
    assert d.decision_type == "vote"
    assert d.data.target_id != "player_004"  # partner
    assert d.data.target_id != "player_001"  # self


def test_night_returns_kill_and_prioritizes_seer_claim() -> None:
    s = WerewolfStrategyAlpha1()
    ctx = _ctx_base(meta={"phase": "werewolf_night"})
    d = s.decide(ctx)
    assert d.decision_type == "night_action"
    assert d.data.action_type == "kill"
    assert d.data.target_id == "player_002"  # seer_claim priority


def test_no_op_when_action_not_allowed() -> None:
    s = WerewolfStrategyAlpha1()
    ctx = _ctx_base(meta={"phase": "daytime_discussion"}, constraints={"allowed_actions": ["vote", "kill"]})
    d = s.decide(ctx)
    assert d.decision_type == "no_op"
    assert d.debug["reason"] == "speech_not_allowed"


def test_validate_reject_self_target_vote() -> None:
    s = WerewolfStrategyAlpha1()
    ctx = _ctx_base(meta={"phase": "daytime_voting"})
    decision = VoteDecision(decision_type="vote", data=VoteData(target_id="player_001", round=1), confidence=0.5)
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert reason == "target_is_self"


def test_fallback_kill_when_only_one_valid_candidate() -> None:
    s = WerewolfStrategyAlpha1()
    ctx = _ctx_base(
        meta={"phase": "werewolf_night"},
        public_state={
            "alive_players": [{"id": "player_001"}, {"id": "player_002"}, {"id": "player_004"}],
            "dead_players": [],
        },
        private_info={"werewolf_partners": ["player_004"]},
        inference={"known_players": []},
    )
    # Force main selection to fail by forbidding player_002, then fallback should no_op
    ctx2 = _ctx_base(
        meta={"phase": "werewolf_night"},
        public_state={
            "alive_players": [{"id": "player_001"}, {"id": "player_002"}, {"id": "player_004"}],
            "dead_players": [],
        },
        private_info={"werewolf_partners": ["player_004"]},
        inference={"known_players": []},
        constraints={"allowed_actions": ["kill"], "forbid_targets": ["player_002"]},
    )
    d = s.decide(ctx2)
    assert d.decision_type in {"no_op", "night_action"}  # if any candidate exists, ok
    if d.decision_type == "night_action":
        assert d.data.target_id == "player_002" is False


def test_context_validation_error_example() -> None:
    bad = _demo_context("daytime_discussion").model_dump()
    bad["meta"]["day_number"] = 0
    try:
        StrategyContext.model_validate(bad)
        assert False, "expected ValidationError"
    except ValidationError:
        assert True


def test_rankings_input_works_without_probabilities() -> None:
    s = WerewolfStrategyAlpha1()
    ctx = _ctx_base(
        meta={"phase": "werewolf_night"},
        inference={
            "known_players": [],
            "rankings": {
                # 约定：越靠前越像狼人 / 越像村民
                "werewolf_likelihood": ["player_005", "player_003", "player_002"],
                "villager_likelihood": ["player_002", "player_003", "player_005"],
            },
        },
    )
    d = s.decide(ctx)
    assert d.decision_type == "night_action"
    assert d.data.action_type == "kill"
    # 夜刀优先 kill “更像好人”的（good_prob更高）：villager_likelihood 里更靠前的 player_002
    assert d.data.target_id == "player_002"

