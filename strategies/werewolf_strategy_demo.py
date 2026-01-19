"""
Werewolf strategy (demo) - single-file design + tests.

Run demo:
  py werewolf_strategy_demo.py

Run tests:
  py -m pytest -q werewolf_strategy_demo.py
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
    self_player_id: Optional[str] = None  # if agent_id != player_id, pass it here


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
# Werewolf strategy (demo)
# -----------------------------


class WerewolfStrategyDemo:
    """
    Demo-level werewolf strategy:
    - Only acts in phase == "werewolf_night"
    - Only emits night_action: kill
    - Prefers killing players tagged as seer/witch claims (if present)
    - Otherwise kills the most-likely-good alive player (to hit strong good roles)
    - Always avoids killing partners / forbidden targets
    """

    SEER_WITCH_TAG_PRIORITY: tuple[str, ...] = (
        "seer_confirmed",
        "witch_confirmed",
        "seer_claim",
        "witch_claim",
    )

    def decide(self, context: StrategyContext) -> StrategyDecision:
        if context.meta.role != "werewolf":
            return NoOpDecision(decision_type="no_op", confidence=0.0, debug={"reason": "not_werewolf"})

        if context.meta.phase != "werewolf_night":
            return NoOpDecision(decision_type="no_op", confidence=0.0, debug={"reason": "not_my_phase"})

        if "kill" not in context.constraints.allowed_actions:
            return NoOpDecision(decision_type="no_op", confidence=0.0, debug={"reason": "kill_not_allowed"})

        target_id, reason, confidence = self._pick_kill_target(context)
        if target_id is None:
            return NoOpDecision(decision_type="no_op", confidence=0.1, debug={"reason": "no_valid_targets"})

        decision = NightActionDecision(
            decision_type="night_action",
            data=NightActionData(action_type="kill", target_id=target_id),
            confidence=confidence,
            debug={"reason": reason},
        )

        ok, why = self.validate(context, decision)
        if ok:
            return decision
        return self.fallback(context, {"reason": why})

    def validate(self, context: StrategyContext, decision: StrategyDecision) -> tuple[bool, str]:
        if decision.decision_type != "night_action":
            return False, "werewolf_strategy_only_supports_night_action"

        if decision.data.action_type != "kill":
            return False, "werewolf_strategy_demo_only_supports_kill"

        if "kill" not in context.constraints.allowed_actions:
            return False, "kill_not_allowed"

        if not decision.data.target_id:
            return False, "missing_target_id"

        alive_ids = {p.id for p in context.public_state.alive_players}
        if decision.data.target_id not in alive_ids:
            return False, "target_not_alive"

        if decision.data.target_id in set(context.constraints.forbid_targets):
            return False, "target_forbidden"

        partners = set(context.private_info.werewolf_partners or [])
        if decision.data.target_id in partners:
            return False, "target_is_partner"

        if context.meta.self_player_id and decision.data.target_id == context.meta.self_player_id:
            return False, "cannot_kill_self"

        return True, ""

    def fallback(self, context: StrategyContext, last_error: dict) -> StrategyDecision:
        target_id, reason, confidence = self._pick_first_valid_target(context)
        if target_id is None:
            return NoOpDecision(
                decision_type="no_op",
                confidence=0.1,
                debug={"reason": "fallback_no_targets", "last_error": last_error},
            )
        return NightActionDecision(
            decision_type="night_action",
            data=NightActionData(action_type="kill", target_id=target_id),
            confidence=confidence,
            debug={"reason": reason, "last_error": last_error},
        )

    # ---- internal helpers ----

    def _excluded_targets(self, context: StrategyContext) -> set[str]:
        excluded = set(context.constraints.forbid_targets)
        excluded |= set(context.private_info.werewolf_partners or [])
        if context.meta.self_player_id:
            excluded.add(context.meta.self_player_id)
        return excluded

    def _alive_candidate_ids(self, context: StrategyContext) -> list[str]:
        excluded = self._excluded_targets(context)
        ids = [p.id for p in context.public_state.alive_players if p.id not in excluded]
        return sorted(ids)  # deterministic for demo/tests

    def _pick_kill_target(self, context: StrategyContext) -> tuple[Optional[str], str, float]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None, "no_candidates", 0.0

        kp_by_id = {kp.player_id: kp for kp in context.inference.known_players}

        # 1) Prioritize claimed/confirmed seer/witch if alive & not excluded
        for tag in self.SEER_WITCH_TAG_PRIORITY:
            tagged = [pid for pid in candidates if tag in (kp_by_id.get(pid).tags if kp_by_id.get(pid) else [])]
            if tagged:
                return tagged[0], f"priority_tag:{tag}", 0.85

        # 2) Otherwise: kill the most-likely-good candidate (wolf wants to hit good roles)
        scored: list[tuple[float, str]] = []
        for pid in candidates:
            kp = kp_by_id.get(pid)
            good_prob = kp.camp_prob.good if kp else 0.5
            scored.append((good_prob, pid))

        scored.sort(key=lambda t: (-t[0], t[1]))
        best_good_prob, best_pid = scored[0]
        return best_pid, f"max_good_prob:{best_good_prob:.2f}", 0.65

    def _pick_first_valid_target(self, context: StrategyContext) -> tuple[Optional[str], str, float]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None, "fallback_no_candidates", 0.0
        return candidates[0], "fallback_first_candidate", 0.5


# -----------------------------
# Demo runner (optional)
# -----------------------------


def _demo_context() -> StrategyContext:
    return StrategyContext(
        meta=Meta(
            game_id="game_demo",
            agent_id="agent_wolf_001",
            role="werewolf",
            phase="werewolf_night",
            day_number=1,
            time_remaining=20,
            self_player_id="player_001",
        ),
        public_state=PublicState(
            alive_players=[
                PlayerPublic(id="player_001", name="我"),
                PlayerPublic(id="player_002", name="玩家2"),
                PlayerPublic(id="player_003", name="玩家3"),
                PlayerPublic(id="player_004", name="玩家4"),
            ],
            dead_players=[],
        ),
        private_info=PrivateInfo(werewolf_partners=["player_004"]),
        inference=Inference(
            known_players=[
                KnownPlayer(player_id="player_002", camp_prob=CampProb(good=0.9, werewolf=0.1), tags=["seer_claim"]),
                KnownPlayer(player_id="player_003", camp_prob=CampProb(good=0.6, werewolf=0.4), tags=[]),
                KnownPlayer(player_id="player_004", camp_prob=CampProb(good=0.5, werewolf=0.5), tags=[]),
            ]
        ),
        constraints=Constraints(
            allowed_actions=["kill"],
            forbid_targets=[],
        ),
        memory=Memory(memory_summary="Day1: 2号跳预言家，报3号金水。"),
    )


def main() -> None:
    ctx = _demo_context()
    strat = WerewolfStrategyDemo()
    decision = strat.decide(ctx)
    print(decision.model_dump())


if __name__ == "__main__":
    main()


# -----------------------------
# Tests (pytest)
# -----------------------------


def _ctx_base(**overrides) -> StrategyContext:
    base = _demo_context()
    return StrategyContext.model_validate({**base.model_dump(), **overrides})


def test_decide_kill_in_werewolf_night() -> None:
    s = WerewolfStrategyDemo()
    d = s.decide(_demo_context())
    assert d.decision_type == "night_action"
    assert d.data.action_type == "kill"
    assert d.data.target_id in {"player_002", "player_003"}  # excludes self + partner


def test_prioritize_seer_claim_tag() -> None:
    s = WerewolfStrategyDemo()
    ctx = _demo_context()
    d = s.decide(ctx)
    assert d.decision_type == "night_action"
    assert d.data.target_id == "player_002"
    assert "priority_tag:seer_claim" in d.debug.get("reason", "")


def test_avoid_partner_target() -> None:
    s = WerewolfStrategyDemo()
    ctx = _ctx_base(
        private_info={"werewolf_partners": ["player_002", "player_004"]},
        inference={
            "known_players": [
                {"player_id": "player_002", "camp_prob": {"good": 0.9, "werewolf": 0.1}, "tags": ["seer_claim"]},
                {"player_id": "player_003", "camp_prob": {"good": 0.6, "werewolf": 0.4}, "tags": []},
            ]
        },
    )
    d = s.decide(ctx)
    assert d.decision_type == "night_action"
    assert d.data.target_id == "player_003"


def test_no_op_when_phase_not_match() -> None:
    s = WerewolfStrategyDemo()
    ctx = _ctx_base(meta={**_demo_context().meta.model_dump(), "phase": "daytime_discussion"})
    d = s.decide(ctx)
    assert d.decision_type == "no_op"


def test_no_op_when_kill_not_allowed() -> None:
    s = WerewolfStrategyDemo()
    ctx = _ctx_base(constraints={"allowed_actions": ["vote"]})
    d = s.decide(ctx)
    assert d.decision_type == "no_op"


def test_validate_reject_dead_target() -> None:
    s = WerewolfStrategyDemo()
    ctx = _ctx_base(
        public_state={
            "alive_players": [{"id": "player_001"}, {"id": "player_002"}],
            "dead_players": [{"id": "player_003"}],
        }
    )
    decision = NightActionDecision(
        decision_type="night_action",
        data=NightActionData(action_type="kill", target_id="player_003"),
        confidence=0.5,
    )
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert reason == "target_not_alive"


def test_context_validation_error_example() -> None:
    # day_number must be >= 1
    bad = _demo_context().model_dump()
    bad["meta"]["day_number"] = 0
    try:
        StrategyContext.model_validate(bad)
        assert False, "expected ValidationError"
    except ValidationError:
        assert True

