"""
Villager strategy (Alpha 1.0) - single-file design + tests.

Goals vs demo:
- Cover key phases: daytime_discussion (speech), daytime_voting (vote), sheriff_election (vote) (no night actions for villagers)
- Event-driven light inference engine to dynamically update player camp probabilities (good/werewolf)
- Strong validation + deterministic fallback logic (avoid self/forbidden targets)
- Sheriff follow logic + tie-break mechanism for top suspects
- Four-dimension seer siding rules (consistency/speech quality/vote support/check closure)

Run demo:
  py villager_strategy_alpha1.py

Run tests:
  py -m pytest -q villager_strategy_alpha1.py
"""

from __future__ import annotations
import re
from typing import Annotated, Literal, Optional, List, Dict, Tuple, Set
from pydantic import BaseModel, Field, ValidationError
from collections import defaultdict


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
    新版输入（来自第二组LLM）：只给“排序”而不是概率。
    约定：列表从左到右 = 越可能该阵营/身份。
    - werewolf_likelihood：越靠前 wolf_prob 越高
    - villager_likelihood：越靠前 wolf_prob 越低（更像好人）
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
    allowed_actions: list[str] = Field(default_factory=list)
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


class VillagerStrategyAlpha1:
    SHERIFF_TRUST_THRESHOLD = 0.3
    TOP_K_SUSPECTS = 5
    TIE_BREAK_THRESHOLD = 0.05
    PROPHET_TAG_BONUS_WEIGHT = 0.2

    CONTRADICTION_PATTERNS = re.compile(r"之前说.*?但现在|投.*?却|验.*?结果|信任.*?却投")
    PROPHET_PATTERNS = re.compile(r"我是预言家|我验了|查杀|金水|我是先知")
    FORBIDDEN_SPEECH_PHRASES = ("我是狼人", "我是狼", "我承认我是狼")
    PROPHET_CLAIM_TAGS = ("seer_claim", "prophet_claim")
    SUSPICION_TAGS = ("wolf_suspect", "talk_inconsistent", "vote_inconsistent")

    def decide(self, context: StrategyContext) -> StrategyDecision:
        if context.meta.self_player_id:
            alive = {p.id for p in context.public_state.alive_players}
            if context.meta.self_player_id not in alive:
                return NoOpDecision(confidence=0.8, debug={"reason": "player_dead"})

        if context.meta.role != "villager":
            return NoOpDecision(confidence=0.0, debug={"reason": "not_villager"})

        self._update_camp_prob(context)

        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)

        if phase == "daytime_discussion":
            if "speech" not in allowed:
                return NoOpDecision(
                    confidence=0.8,
                    debug={"reason": "speech_not_allowed", "allowed_actions": sorted(allowed)}
                )
            return self._decide_speech(context)

        if phase == "daytime_voting":
            if "vote" not in allowed:
                return NoOpDecision(
                    confidence=0.8,
                    debug={"reason": "vote_not_allowed", "allowed_actions": sorted(allowed)}
                )
            return self._decide_vote(context)

        if phase == "sheriff_election":
            if "vote" not in allowed:
                return NoOpDecision(
                    confidence=0.8,
                    debug={"reason": "vote_not_allowed", "allowed_actions": sorted(allowed)}
                )
            return self._decide_sheriff_vote(context)

        return NoOpDecision(
            confidence=0.2,
            debug={"reason": f"phase_not_supported:{phase}", "allowed_actions": sorted(allowed)}
        )

    def validate(self, context: StrategyContext, decision: StrategyDecision) -> tuple[bool, str]:
        required_errors = self._validate_required_fields(context)
        if required_errors:
            return False, f"missing_required_fields:{','.join(required_errors)}"

        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)
        phase_expected = {
            "daytime_discussion": "speech",
            "daytime_voting": "vote",
            "sheriff_election": "vote",
        }.get(phase)

        if decision.decision_type != "no_op" and phase_expected and decision.decision_type != phase_expected:
            return False, f"action_phase_mismatch:{decision.decision_type} in {phase}"

        if decision.decision_type == "speech":
            return self._validate_speech(context, decision.data)
        if decision.decision_type == "vote":
            return self._validate_vote(context, decision.data)

        return True, ""

    def fallback(self, context: StrategyContext, last_error: dict) -> StrategyDecision:
        phase = context.meta.phase
        allowed = set(context.constraints.allowed_actions)
        day = context.meta.day_number

        if phase == "daytime_voting" and "vote" in allowed:
            target = self._first_valid_target(context)
            if target:
                return VoteDecision(
                    data=VoteData(target_id=target, round=day),
                    confidence=0.3,
                    debug={"reason": "fallback_vote", "last_error": last_error}
                )
            return NoOpDecision(confidence=0.1, debug={"reason": "fallback_no_targets"})

        if phase == "sheriff_election" and "vote" in allowed:
            target = self._first_valid_target(context)
            if target:
                return VoteDecision(
                    data=VoteData(target_id=target, round=1),
                    confidence=0.3,
                    debug={"reason": "fallback_sheriff_vote", "last_error": last_error}
                )
            return NoOpDecision(confidence=0.1, debug={"reason": "fallback_no_targets"})

        if phase == "daytime_discussion" and "speech" in allowed:
            return SpeechDecision(
                data=SpeechData(
                    content=f"第{day}天信息还不充分，建议重点看投票和发言逻辑，尤其是矛盾点。",
                    speech_round=day,
                    turn_order=getattr(context.constraints, 'current_turn_order', 0)
                ),
                confidence=0.3,
                debug={"reason": "fallback_speech", "last_error": last_error}
            )

        return NoOpDecision(confidence=0.1, debug={"reason": "fallback_no_valid_action"})

    def _decide_speech(self, context: StrategyContext) -> StrategyDecision:
        top_suspects = self._rank_suspects(context)
        focus_info = self._extract_focus_events(context)
        day = context.meta.day_number
        summary = context.memory.memory_summary.strip()

        if top_suspects:
            target_id, target_prob, target_reason = top_suspects[0]
            content = (
                f"第{day}天核心信息：{focus_info}。"
                f"{f' 补充信息：{summary}' if summary else ''}"
                f" 我重点怀疑{target_id}（{target_reason}，狼人概率{target_prob:.2f}），建议听其逻辑闭环。"
            )
        else:
            content = f"第{day}天核心信息：{focus_info}。{f' 补充信息：{summary}' if summary else ''} 大家先讲清关键逻辑。"

        decision = SpeechDecision(
            data=SpeechData(content=content[:200], speech_round=day, turn_order=getattr(context.constraints, 'current_turn_order', 0)),
            confidence=0.55,
            debug={"reason": "dynamic_focus_speech" if top_suspects else "neutral_speech"}
        )
        ok, why = self.validate(context, decision)
        return decision if ok else self.fallback(context, {"reason": why, "decision": decision.model_dump()})

    def _decide_vote(self, context: StrategyContext) -> StrategyDecision:
        target_id, reason, confidence = self._pick_vote_target(context)
        if not target_id:
            return self.fallback(context, {"reason": "no_valid_vote_target"})

        decision = VoteDecision(
            data=VoteData(target_id=target_id, round=context.meta.day_number),
            confidence=confidence,
            debug={"reason": reason}
        )
        ok, why = self.validate(context, decision)
        return decision if ok else self.fallback(context, {"reason": why, "decision": decision.model_dump()})

    def _decide_sheriff_vote(self, context: StrategyContext) -> StrategyDecision:
        target_id, reason, confidence = self._pick_sheriff_target(context)
        if not target_id:
            return self.fallback(context, {"reason": "no_valid_sheriff_target"})

        decision = VoteDecision(
            data=VoteData(target_id=target_id, round=1),
            confidence=confidence,
            debug={"reason": reason}
        )
        ok, why = self.validate(context, decision)
        return decision if ok else self.fallback(context, {"reason": why, "decision": decision.model_dump()})

    def _pick_vote_target(self, context: StrategyContext) -> tuple[Optional[str], str, float]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None, "no_candidates", 0.0

        sheriff_id = self._sheriff_id_for_context(context)
        if sheriff_id:
            sheriff_wolf_prob = self._get_wolf_prob(context, sheriff_id)
            is_sheriff_trustworthy = sheriff_wolf_prob <= self.SHERIFF_TRUST_THRESHOLD
            if is_sheriff_trustworthy:
                sheriff_vote = context.public_state.last_vote.get("votes", {}).get(sheriff_id)
                if sheriff_vote and self._validate_target(context, sheriff_vote)[0]:
                    return sheriff_vote, f"follow_trusted_sheriff:{sheriff_id} (wolf_prob:{sheriff_wolf_prob:.2f})", 0.8

        top_suspects = self._rank_suspects(context)
        if not top_suspects:
            return self._first_valid_target(context), "fallback_first_valid", 0.3

        top_id, top_prob, top_reason = top_suspects[0]
        is_tie = False
        last_vote_result = context.public_state.last_vote.get("result", {}) if context.public_state.last_vote else {}
        if last_vote_result:
            vote_counts = list(last_vote_result.values())
            max_count = max(vote_counts)
            is_tie = vote_counts.count(max_count) >= 2

        if is_tie and len(top_suspects) >= 2:
            sec_id, sec_prob, _ = top_suspects[1]
            if abs(top_prob - sec_prob) < self.TIE_BREAK_THRESHOLD:
                return sec_id, f"tie_break_top2:{sec_id}", 0.7

        return top_id, f"top_suspect:{top_reason}", 0.75

    def _pick_sheriff_target(self, context: StrategyContext) -> tuple[Optional[str], str, float]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return None, "no_candidates", 0.0

        kp_by_id = self._kp_by_id(context)
        prophet_cred = self._prophet_credibility_for_context(context)

        scored = []
        for pid in candidates:
            kp = kp_by_id.get(pid)
            wolf_prob = kp.camp_prob.werewolf if kp else 0.5
            cred = prophet_cred.get(pid, 0.0)
            scored.append((wolf_prob, cred, pid))

        scored.sort(key=lambda t: (t[0], -t[1], t[2]))
        wolf_prob, cred, best_id = scored[0]

        return best_id, f"low_wolf_prob:{wolf_prob:.2f};prophet_cred:{cred:.2f}", 0.7

    def _rank_suspects(self, context: StrategyContext) -> List[Tuple[str, float, str]]:
        candidates = self._alive_candidate_ids(context)
        if not candidates:
            return []

        scored = [
            (
                self._get_wolf_prob(context, pid),
                self._get_suspicion_reason(context, pid),
                pid
            )
            for pid in candidates
        ]
        scored.sort(key=lambda t: (-t[0], t[2]))
        return [(s[2], s[0], s[1]) for s in scored[:self.TOP_K_SUSPECTS]]

    def _update_camp_prob(self, context: StrategyContext) -> None:
        self._init_prob_cache(context)
        self._update_prob_from_vote(context)
        self._update_prob_from_speech(context)
        self._update_prob_from_night(context)
        self._sync_prob_cache(context)

    def _update_prob_from_vote(self, context: StrategyContext) -> None:
        if not context.public_state.last_vote:
            return

        votes = context.public_state.last_vote.get("votes", {})
        vote_result = context.public_state.last_vote.get("result", {})
        if not vote_result:
            return

        max_votes = max(vote_result.values(), default=0)
        rush_voters = [v for v, t in votes.items() if vote_result.get(t, 0) == max_votes and max_votes > 0]
        candidates = self._alive_candidate_ids(context)

        for pid in candidates:
            if pid not in votes:
                self._adjust_wolf_prob(pid, +0.1)
            elif pid in rush_voters:
                target_prob = self._get_wolf_prob(context, votes[pid])
                if target_prob < 0.4:
                    self._adjust_wolf_prob(pid, +0.15)
            else:
                target_prob = self._get_wolf_prob(context, votes[pid])
                if target_prob > 0.7:
                    self._adjust_wolf_prob(pid, -0.1)

    def _update_prob_from_speech(self, context: StrategyContext) -> None:
        candidates = self._alive_candidate_ids(context)
        for pid in candidates:
            if self._has_speech_contradiction(context, pid):
                self._adjust_wolf_prob(pid, +0.15)

    def _update_prob_from_night(self, context: StrategyContext) -> None:
        if not context.public_state.last_night:
            return

        is_peace = context.public_state.last_night.get("is_peaceful", False)
        if is_peace:
            return

        dead_players_data = context.public_state.last_night.get("dead_players", [])

        if dead_players_data is None:
            dead_players_data = []

        for dead_info in dead_players_data:
            dead_id = None
            death_reason = ""

            if isinstance(dead_info, DeadPlayer):
                dead_id = dead_info.id
                death_reason = dead_info.death_reason or ""
            elif isinstance(dead_info, dict):
                dead_id = dead_info.get("id")
                death_reason = dead_info.get("death_reason", "")
            elif hasattr(dead_info, "id") and hasattr(dead_info, "death_reason"):
                dead_id = getattr(dead_info, "id", None)
                death_reason = getattr(dead_info, "death_reason", "") or ""

            if not dead_id or not isinstance(dead_id, str):
                continue

            if dead_id in self._camp_prob_cache:
                death_reason_str = str(death_reason)
                if "狼人刀杀" in death_reason_str:
                    self._adjust_wolf_prob(dead_id, -0.2)
                else:
                    self._adjust_wolf_prob(dead_id, -0.1)

    def _prophet_credibility_for_context(self, context: StrategyContext) -> Dict[str, float]:
        kp_by_id = self._kp_by_id(context)
        prophet_ids = []
        for pid in self._alive_candidate_ids(context):
            kp = kp_by_id.get(pid, KnownPlayer(player_id=pid))
            has_claim_tag = any(t in self.PROPHET_CLAIM_TAGS for t in kp.tags)
            has_claim_speech = any(self.PROPHET_PATTERNS.search(e.get("content", ""))
                                   for e in context.memory.recent_events if e.get("player_id") == pid)
            if has_claim_tag or has_claim_speech:
                prophet_ids.append(pid)

        if len(prophet_ids) == 0:
            return {}

        cred_dict = {}
        initial_score = 0.8 if len(prophet_ids) == 1 else 0.5
        for pid in prophet_ids:
            cred_dict[pid] = initial_score

        INITIAL_WEIGHT = 0.3
        EVIDENCE_WEIGHT = 0.7

        for pid in prophet_ids:
            kp = kp_by_id.get(pid, KnownPlayer(player_id=pid))

            consistency = self._is_prophet_consistent(context, pid)
            speech_quality = 1.0 if any(t in ("查杀", "金水") for t in kp.tags) else 0.5
            support_rate = self._get_prophet_support_rate(context, pid)
            check_closed = self._is_prophet_check_closed(context, pid)

            four_dim_score = (consistency * 0.3) + (speech_quality * 0.2) + (support_rate * 0.2) + (check_closed * 0.3)

            final_score = (cred_dict[pid] * INITIAL_WEIGHT) + (four_dim_score * EVIDENCE_WEIGHT)
            cred_dict[pid] = min(1.0, max(0.0, final_score))

        return cred_dict

    def _validate_required_fields(self, context: StrategyContext) -> List[str]:
        errors = []
        if not context.meta.self_player_id:
            errors.append("self_player_id")
        if not context.public_state.alive_players:
            errors.append("alive_players")
        if not context.constraints.allowed_actions:
            errors.append("allowed_actions")
        return errors

    def _validate_speech(self, context: StrategyContext, speech_data: SpeechData) -> tuple[bool, str]:
        if speech_data.speech_round != context.meta.day_number:
            return False, f"speech_round_mismatch:{speech_data.speech_round}"
        content = speech_data.content.strip()
        if not content:
            return False, "empty_speech"
        for phrase in self.FORBIDDEN_SPEECH_PHRASES:
            if phrase in content:
                return False, f"forbidden_phrase:{phrase}"
        return True, ""

    def _validate_vote(self, context: StrategyContext, vote_data: VoteData) -> tuple[bool, str]:
        if vote_data.round not in (1, context.meta.day_number):
            return False, f"vote_round_mismatch:{vote_data.round}"
        return self._validate_target(context, vote_data.target_id)

    def _validate_target(self, context: StrategyContext, target_id: str) -> tuple[bool, str]:
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

    def _alive_candidate_ids(self, context: StrategyContext) -> list[str]:
        alive_ids = [p.id for p in context.public_state.alive_players]
        excluded = set(context.constraints.forbid_targets)
        if context.meta.self_player_id:
            excluded.add(context.meta.self_player_id)
        candidates = [pid for pid in alive_ids if pid not in excluded]
        return sorted(candidates)

    def _first_valid_target(self, context: StrategyContext) -> Optional[str]:
        candidates = self._alive_candidate_ids(context)
        return candidates[0] if candidates else None

    def _kp_by_id(self, context: StrategyContext) -> dict[str, KnownPlayer]:
        return {kp.player_id: kp for kp in context.inference.known_players}

    # -----------------------------
    # 排名输入 → 概率（兼容第二组LLM只给排名）
    # -----------------------------

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

    def _init_prob_cache(self, context: StrategyContext) -> None:
        current_game_id = context.meta.game_id

        if not hasattr(self, '_camp_prob_cache') or not hasattr(self, '_current_game_id'):
            self._camp_prob_cache = {}
            self._current_game_id = current_game_id
        elif self._current_game_id != current_game_id:
            self._camp_prob_cache = {}
            self._current_game_id = current_game_id

        alive_ids = {p.id for p in context.public_state.alive_players}
        dead_ids = {d.id for d in context.public_state.dead_players}
        all_player_ids = alive_ids | dead_ids
        kp_by_id = self._kp_by_id(context)

        for pid, kp in kp_by_id.items():
            if pid in all_player_ids:
                self._camp_prob_cache[pid] = kp.camp_prob

        for pid in all_player_ids:
            if pid not in self._camp_prob_cache:
                derived = self._derive_wolf_prob_from_rankings(context, pid)
                if derived is None:
                    self._camp_prob_cache[pid] = CampProb(good=0.5, werewolf=0.5)
                else:
                    self._camp_prob_cache[pid] = CampProb(good=1.0 - derived, werewolf=derived)

    def _get_wolf_prob(self, context: StrategyContext, player_id: str) -> float:
        if not hasattr(self, '_camp_prob_cache'):
            self._init_prob_cache(context)
        return self._camp_prob_cache.get(player_id, CampProb(good=0.5, werewolf=0.5)).werewolf

    def _adjust_wolf_prob(self, player_id: str, delta: float) -> None:
        if player_id not in self._camp_prob_cache:
            return
        new_wolf = self._camp_prob_cache[player_id].werewolf + delta
        self._camp_prob_cache[player_id].werewolf = max(0.0, min(1.0, new_wolf))
        self._camp_prob_cache[player_id].good = 1.0 - self._camp_prob_cache[player_id].werewolf

    def _sync_prob_cache(self, context: StrategyContext) -> None:
        kp_by_id = self._kp_by_id(context)

        for pid, prob in self._camp_prob_cache.items():
            if pid in kp_by_id:
                kp_by_id[pid].camp_prob = prob
            else:
                existing_ids = {kp.player_id for kp in context.inference.known_players}
                if pid not in existing_ids:
                    context.inference.known_players.append(KnownPlayer(
                        player_id=pid,
                        camp_prob=prob
                    ))

        current_players = {p.id for p in context.public_state.alive_players} | \
                          {d.id for d in context.public_state.dead_players}

        for pid in list(self._camp_prob_cache.keys()):
            if pid not in current_players:
                del self._camp_prob_cache[pid]

    def _sheriff_id_for_context(self, context: StrategyContext) -> Optional[str]:
        for player in context.public_state.alive_players:
            if player.is_sheriff:
                return player.id
        return None

    def _has_speech_contradiction(self, context: StrategyContext, player_id: str) -> bool:
        statements = [e.get("content", "") for e in context.memory.recent_events if e.get("player_id") == player_id]
        return len(statements) >= 2 and bool(self.CONTRADICTION_PATTERNS.search(" ".join(statements)))

    def _get_suspicion_reason(self, context: StrategyContext, player_id: str) -> str:
        reasons = []
        wolf_prob = self._get_wolf_prob(context, player_id)
        if wolf_prob > 0.7:
            reasons.append("狼人概率高")
        if self._has_speech_contradiction(context, player_id):
            reasons.append("发言矛盾")

        if context.public_state.last_vote and player_id not in context.public_state.last_vote.get("votes", {}):
            reasons.append("弃票")

        return "；".join(reasons) if reasons else "无明确嫌疑"

    def _extract_focus_events(self, context: StrategyContext) -> str:
        focus = []
        if context.public_state.last_vote:
            max_votes = max(context.public_state.last_vote.get("result", {}).values(), default=0)
            top_voted = next((k for k, v in context.public_state.last_vote.get("result", {}).items() if v == max_votes), None)
            if top_voted:
                focus.append(f"上轮{max_votes}票投出{top_voted}")
        if context.public_state.last_night:
            focus.append("平安夜" if context.public_state.last_night.get("is_peaceful") else "昨晚有人死亡")
        conflict_pids = [pid for pid in self._alive_candidate_ids(context) if self._has_speech_contradiction(context, pid)]
        if conflict_pids:
            focus.append(f"{','.join(conflict_pids[:2])}发言矛盾")
        return "；".join(focus) if focus else "无关键信息"

    def _is_prophet_consistent(self, context: StrategyContext, prophet_id: str) -> float:
        prophet_speeches = [e.get("content", "") for e in context.memory.recent_events
                            if e.get("player_id") == prophet_id and self.PROPHET_PATTERNS.search(e.get("content", ""))]

        if not prophet_speeches:
            return 0.5

        checks = []
        for speech in prophet_speeches:
            if "查杀" in speech or "金水" in speech:
                target = self._extract_check_target(speech, context)
                if target:
                    check_type = "kill" if "查杀" in speech else "gold"
                    checks.append((check_type, target))

        if not checks:
            return 0.5

        player_checks = {}
        contradiction_count = 0

        for check_type, target in checks:
            if target not in player_checks:
                player_checks[target] = [check_type]
            else:
                if check_type not in player_checks[target]:
                    player_checks[target].append(check_type)
                    if len(player_checks[target]) > 1:
                        contradiction_count += 1

        total_checks = len(checks)
        if total_checks == 0:
            return 0.5

        consistency_score = 1.0 - (contradiction_count / total_checks)
        consistency_score = 0.2 + 0.6 * consistency_score

        return consistency_score

    def _get_prophet_support_rate(self, context: StrategyContext, prophet_id: str) -> float:
        if not context.public_state.last_vote or "sheriff" not in context.meta.phase:
            return 0.5
        votes = context.public_state.last_vote.get("votes", {})
        support = sum(1 for t in votes.values() if t == prophet_id)
        total = len(votes) if len(votes) > 0 else 1
        return support / total

    def _is_prophet_check_closed(self, context: StrategyContext, prophet_id: str) -> float:
        prophet_speeches = [e.get("content", "") for e in context.memory.recent_events
                            if e.get("player_id") == prophet_id and self.PROPHET_PATTERNS.search(e.get("content", ""))]
        if not prophet_speeches:
            return 0.5

        dead_ids = {d.id for d in context.public_state.dead_players}
        alive_ids = {p.id for p in context.public_state.alive_players}
        closed_count = 0
        total_checks = 0

        for speech in prophet_speeches:
            target = self._extract_check_target(speech, context)
            if not target:
                continue
            total_checks += 1
            if "查杀" in speech and target in dead_ids:
                closed_count += 1
            elif "金水" in speech and target in alive_ids:
                closed_count += 1

        return closed_count / total_checks if total_checks > 0 else 0.5

    def _extract_check_target(self, speech: str, context: StrategyContext) -> Optional[str]:
        if not speech or not isinstance(speech, str):
            return None

        all_players = context.public_state.alive_players + context.public_state.dead_players

        for player in all_players:
            if player.id and isinstance(player.id, str) and player.id in speech:
                return player.id

        for player in all_players:
            if player.name and isinstance(player.name, str) and player.name in speech:
                return player.id

        return None


def _demo_context_villager(phase: str) -> StrategyContext:
    return StrategyContext(
        meta=Meta(
            game_id="game_demo_villager",
            agent_id="agent_villager_001",
            role="villager",
            phase=phase,
            day_number=1,
            time_remaining=30,
            self_player_id="player_v001",
        ),
        public_state=PublicState(
            alive_players=[
                PlayerPublic(id="player_v001", name="我(村民)"),
                PlayerPublic(id="player_v002", name="玩家2", is_sheriff=True),
                PlayerPublic(id="player_v003", name="玩家3"),
                PlayerPublic(id="player_v004", name="玩家4"),
                PlayerPublic(id="player_v005", name="玩家5"),
            ],
            dead_players=[],
            last_vote={"votes": {"player_v002": "player_v004"}, "result": {"player_v004": 1}},
        ),
        private_info=PrivateInfo(),
        inference=Inference(
            known_players=[
                KnownPlayer(player_id="player_v002", camp_prob=CampProb(good=0.8, werewolf=0.2), tags=["seer_claim"]),
                KnownPlayer(player_id="player_v003", camp_prob=CampProb(good=0.6, werewolf=0.4), tags=[]),
                KnownPlayer(player_id="player_v004", camp_prob=CampProb(good=0.3, werewolf=0.7), tags=["wolf_suspect"]),
                KnownPlayer(player_id="player_v005", camp_prob=CampProb(good=0.5, werewolf=0.5),
                            tags=["talk_inconsistent"]),
            ]
        ),
        constraints=Constraints(allowed_actions=["speech", "vote"], forbid_targets=[]),
        memory=Memory(
            memory_summary="Day1: 2号警长跳预言家查杀4号，5号发言前后矛盾。",
            recent_events=[
                {"player_id": "player_v002", "content": "我是预言家，昨晚验了4号是狼人"},
                {"player_id": "player_v005", "content": "我信任2号但我觉得4号可能是好人"},
            ]
        ),
    )


def main_villager() -> None:
    s = VillagerStrategyAlpha1()

    print("===== Villager Alpha1 - daytime_discussion =====")
    print(s.decide(_demo_context_villager("daytime_discussion")).model_dump())

    print("\n===== Villager Alpha1 - daytime_voting =====")
    print(s.decide(_demo_context_villager("daytime_voting")).model_dump())

    print("\n===== Villager Alpha1 - sheriff_election =====")
    print(s.decide(_demo_context_villager("sheriff_election")).model_dump())


def _ctx_villager_base(**overrides) -> StrategyContext:
    base = _demo_context_villager("daytime_discussion").model_dump()

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


def test_villager_discussion_returns_speech() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(meta={"phase": "daytime_discussion"})
    d = s.decide(ctx)
    assert d.decision_type == "speech"
    assert d.data.speech_round == 1
    assert len(d.data.content) > 0
    for phrase in s.FORBIDDEN_SPEECH_PHRASES:
        assert phrase not in d.data.content


def test_villager_voting_returns_vote() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(meta={"phase": "daytime_voting"})
    d = s.decide(ctx)
    assert d.decision_type == "vote"
    assert d.data.target_id != "player_v001"
    assert d.data.round == 1


def test_villager_sheriff_election_returns_vote() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(meta={"phase": "sheriff_election"})
    d = s.decide(ctx)
    assert d.decision_type == "vote"
    assert d.data.target_id != "player_v001"
    assert d.data.round == 1


def test_villager_no_op_when_action_not_allowed() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(
        meta={"phase": "daytime_discussion"},
        constraints={"allowed_actions": ["vote"]}
    )
    d = s.decide(ctx)
    assert d.decision_type == "no_op"
    assert "not_allowed" in d.debug["reason"] or "speech" in d.debug["reason"]


def test_villager_validate_reject_self_target_vote() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(meta={"phase": "daytime_voting"})
    decision = VoteDecision(
        decision_type="vote",
        data=VoteData(target_id="player_v001", round=1),
        confidence=0.5
    )
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert "self" in reason


def test_villager_validate_reject_empty_speech() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(meta={"phase": "daytime_discussion"})
    decision = SpeechDecision(
        decision_type="speech",
        data=SpeechData(content="", speech_round=1, turn_order=0),
        confidence=0.5
    )
    ok, reason = s.validate(ctx, decision)
    assert ok is False
    assert "empty" in reason


def test_villager_fallback_vote_when_main_logic_fails() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(
        meta={"phase": "daytime_voting"},
        public_state={
            "alive_players": [
                {"id": "player_v001"},
                {"id": "player_v002"}
            ],
            "dead_players": []
        },
        constraints={"allowed_actions": ["vote"], "forbid_targets": []}
    )
    d = s.decide(ctx)
    assert d.decision_type == "vote"
    assert d.data.target_id == "player_v002"


def test_villager_probability_update_mechanism() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(meta={"phase": "daytime_discussion"})

    initial_prob = s._get_wolf_prob(ctx, "player_v004")

    ctx.public_state.last_vote = {
        "votes": {"player_v004": "player_v002"},
        "result": {"player_v002": 1}
    }

    s._update_camp_prob(ctx)

    updated_prob = s._get_wolf_prob(ctx, "player_v004")
    assert initial_prob != updated_prob


def test_villager_sheriff_follow_logic() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(
        meta={"phase": "daytime_voting"},
        public_state={
            "alive_players": [
                {"id": "player_v001", "name": "我", "is_sheriff": False},
                {"id": "player_v002", "name": "警长", "is_sheriff": True},
                {"id": "player_v003", "name": "玩家3"},
            ],
            "dead_players": [],
            "last_vote": {"votes": {"player_v002": "player_v003"}, "result": {"player_v003": 1}}
        },
        inference={
            "known_players": [
                {"player_id": "player_v002", "camp_prob": {"good": 0.9, "werewolf": 0.1}},
                {"player_id": "player_v003", "camp_prob": {"good": 0.3, "werewolf": 0.7}},
            ]
        }
    )

    d = s.decide(ctx)
    assert d.decision_type == "vote"
    assert d.data.target_id == "player_v003"
    assert "sheriff" in d.debug["reason"]


def test_villager_tie_break_mechanism() -> None:
    s = VillagerStrategyAlpha1()

    ctx = _ctx_villager_base(
        meta={"phase": "daytime_voting", "day_number": 2},
        public_state={
            "last_vote": {
                "result": {"player_v003": 2, "player_v004": 2}
            }
        },
        inference={
            "known_players": [
                {"player_id": "player_v003", "camp_prob": {"good": 0.4, "werewolf": 0.6}},
                {"player_id": "player_v004", "camp_prob": {"good": 0.41, "werewolf": 0.59}},
            ]
        }
    )

    d = s.decide(ctx)
    assert d.decision_type == "vote"
    assert d.data.target_id in ["player_v003", "player_v004"]


def test_villager_rankings_seed_prob_cache() -> None:
    s = VillagerStrategyAlpha1()
    ctx = _ctx_villager_base(
        inference={
            "known_players": [],
            "rankings": {
                "werewolf_likelihood": ["player_v005", "player_v004", "player_v003", "player_v002"],
                "villager_likelihood": ["player_v002", "player_v003", "player_v004", "player_v005"],
            },
        }
    )
    # 这里不要求精确数值，只要“能初始化成非默认且可用”
    p2 = s._get_wolf_prob(ctx, "player_v002")
    p5 = s._get_wolf_prob(ctx, "player_v005")
    assert p2 < 0.5  # 更像村民，wolf_prob 应该偏低
    assert p5 > 0.5  # 更像狼人，wolf_prob 应该偏高


if __name__ == "__main__":
    main_villager()

    print("\n" + "=" * 50)
    print("Running basic tests...")

    test_villager_discussion_returns_speech()
    test_villager_voting_returns_vote()
    test_villager_validate_reject_self_target_vote()

    print("All basic tests passed!")