### 法官系统 →（Agent框架）→ 策略层：StrategyContext 输入规范（v1）

这份文档回答一个问题：**第一组法官系统需要“提供/回传”哪些信息，才能让策略层正常决策？**

结论先说清：
- **法官系统不需要直接调用策略层**（默认架构：策略由第二组 Agent 框架本地调用/或HTTP调用）。
- 但为了让 Agent 能构造 `StrategyContext`，**法官系统必须通过事件/查询接口提供公共状态与权限过滤后的私有信息**。
- **第二组（LLM/Agent）负责**：记忆汇总 `memory`、LLM 推断 `inference.rankings`（排名），以及把这些拼装成 `StrategyContext` 交给策略模块。

---

## 1) StrategyContext 总览：哪些字段来自法官？哪些来自 Agent？

- **meta（主要来自法官 + Agent自身信息）**
  - **法官给/可推导**：`phase`, `day_number`, `time_remaining`(若有)
  - **Agent给**：`game_id`, `agent_id`, `role`, `self_player_id`（通常等于自己的 player_id）

- **public_state（来自法官，所有人可见）**
  - `alive_players`: `[{id,name?,is_ai?,is_sheriff?}]`
  - `dead_players`: `[{id,name?,death_reason?}]`
  - `last_night`: `dict | null`（可选：昨夜结算摘要）
  - `last_vote`: `dict | null`（可选：上轮投票摘要）

- **private_info（来自法官，必须权限过滤）**
  - 狼人：`werewolf_partners`
  - 预言家：`seer_check_history`（含历史查验结果）
  - 女巫：`witch_potions`、`tonight_victim_hint`（今晚刀口提示，若规则/实现支持）

- **constraints（建议来自法官，或由 Agent 用法官规则配置推导）**
  - `allowed_actions`: 当前阶段允许动作集合（非常关键，避免非法提交）
  - `forbid_targets`: 当前阶段不可选目标集合（死者、自身、规则禁止自救等）
  - `max_actions_this_phase`: 通常 1
  - `current_turn_order`: 发言顺序（可选；若法官细化到“轮到谁发言”）

- **memory（来自 Agent，基于法官事件流构建）**
  - `memory_summary`: 近期关键事件摘要（建议 <= 1k tokens）
  - `recent_events`: 原始事件列表（可选）

- **inference（来自 Agent / 第二组 LLM）**
  - `rankings`: **第二组输出的“排名版推断”对象**（详见 `LLM_Rankings_Input_Spec.md`）
  - `known_players`: 旧版概率（兼容保留；第二组不必输出）

---

## 2) 法官系统需要提供的“最小信息集”（必须项）

### 2.1 所有角色/所有阶段（公共）
法官至少要让 Agent 获得：
- `phase`（例如：`werewolf_night / seer_night / witch_night / daytime_discussion / daytime_voting / sheriff_election`）
- `day_number`
- `alive_players`（必须含 `id`）
- `dead_players`（可为空数组）
- `time_remaining`（建议给，单位秒；可选）

### 2.2 角色私有（权限过滤后）
- **狼人**：`werewolf_partners: list[str]`
- **预言家**：`seer_check_history: list[{target_id,result,round}]`
- **女巫**：
  - `witch_potions: {antidote_left:int, poison_left:int, can_self_save:bool|null}`
  - `tonight_victim_hint: str|null`（若实现支持；用于“救人”目标校验）

### 2.3 约束（强烈建议）
法官系统最好在每个“需要决策”的阶段，给出：
- `allowed_actions: list[str]`
  - 例：狼人夜 `["kill"]`；预言家夜 `["check"]`；女巫夜 `["save","poison","no_potion"]`；白天讨论 `["speech"]`；白天投票 `["vote"]`
- `forbid_targets: list[str]`
  - 至少包含：已死亡玩家
  - 可选包含：当前规则下不允许的目标（例如女巫不可自救时，把自己的 id 放进 forbid_targets）
- `max_actions_this_phase: int`（通常为 1）
- `current_turn_order: int`（可选：若法官细化到“轮到谁发言”，建议给 turn_order，便于 speech 校验/复盘）

---

## 3) 建议的法官事件/查询 → StrategyContext 字段映射

你们现有《法官系统接口文档.txt》里已经有这些接口/事件，下面只说明“怎么映射进 StrategyContext”：

- **phase_change 事件**
  - `data.new_phase` → `meta.phase`
  - `day_number`（若事件里带）→ `meta.day_number`（否则由 query_public_state 提供）

- **query_public_state 响应**
  - `game_id` → `meta.game_id`（或由 Agent 保留初始 game_id）
  - `phase` → `meta.phase`
  - `day_number` → `meta.day_number`
  - `time_remaining` → `meta.time_remaining`
  - `alive_players / dead_players` → `public_state.alive_players / public_state.dead_players`

- **query_role_info 响应（权限过滤后）**
  - `werewolf_partners` → `private_info.werewolf_partners`
  - `seer_check_history` → `private_info.seer_check_history`
  - `witch_potions` → `private_info.witch_potions`
  - （若有）`tonight_victim_hint` → `private_info.tonight_victim_hint`

- **vote_result / night_reveal 等事件**
  - 事件摘要可写入 `public_state.last_vote / public_state.last_night`（可选）
  - 原始事件建议写入 `memory.recent_events`，并由 Agent 汇总成 `memory.memory_summary`

---

## 4) 最小可用 StrategyContext（示例）

说明：这是一段“Agent 调策略”时的完整输入示例；其中 `inference.rankings` 由第二组 LLM 提供，`public_state/private_info` 来自法官。

```json
{
  "meta": {
    "game_id": "g1",
    "agent_id": "a1",
    "role": "werewolf",
    "phase": "werewolf_night",
    "day_number": 2,
    "time_remaining": 30,
    "self_player_id": "player_001"
  },
  "public_state": {
    "alive_players": [{"id":"player_001"},{"id":"player_002"},{"id":"player_003"}],
    "dead_players": [],
    "last_night": null,
    "last_vote": null
  },
  "private_info": { "werewolf_partners": ["player_003"] },
  "memory": { "memory_summary": "...", "recent_events": [] },
  "inference": {
    "rankings": { "werewolf_likelihood": ["player_002","player_003"], "villager_likelihood": ["player_001"] }
  },
  "constraints": { "allowed_actions": ["kill"], "forbid_targets": ["player_003"], "max_actions_this_phase": 1 }
}
```

