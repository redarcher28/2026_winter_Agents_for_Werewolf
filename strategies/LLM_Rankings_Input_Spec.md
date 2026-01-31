### LLM 输出“排名版推断”JSON规范（供第二组对接）

你们（第二组）**只需要输出“排序列表（rankings）”**，不要输出每个玩家的 `camp_prob / wolf_prob / good_prob`（数值概率）。
我们（策略组）会在策略侧把排序映射成可用的 `wolf_prob / good_prob`，再沿用现有决策逻辑（投票/夜刀/用药基本不变）。

---

## 1) 基本约定（非常关键）

- **玩家ID必须和法官系统一致**：例如 `"player_001"`、`"p1"`，但必须全局统一。
- **排序方向**：列表从左到右，**越靠前越“像/可能是”该阵营/身份**。
  - `werewolf_likelihood`：越靠前越像狼人（我们会得到更高 `wolf_prob`）
  - `villager_likelihood`：越靠前越像好人/村民（我们会得到更低 `wolf_prob`）
- 允许只给其中一类排序；两类都给时我们会融合（好人排序会拉低 wolf 概率）。
- **严禁输出概率字段（第二组对接要求）**：
  - 不要传 `camp_prob`（例如 `{"good":0.7,"werewolf":0.3}` 这种）
  - 不要传 `wolf_prob / good_prob`（这些是策略侧派生出来的中间量）

---

## 2) 排名格式

**第二组需要输出的就是这个对象**（rankings 本体），我们会把它放到 `StrategyContext.inference.rankings` 里：

```json
{
  "werewolf_likelihood": ["player_005", "player_002", "player_003"],
  "villager_likelihood": ["player_001", "player_004", "player_006"],
  "seer_likelihood": ["player_002", "player_007"],
  "witch_likelihood": ["player_009"],
  "confidence": 0.72,
  "source": "LLMClient"
}
```

说明：
- `seer_likelihood/witch_likelihood` 当前主要用于未来扩展（神职识别），策略侧会先存着不用也不会报错。
- `confidence/source` 可选，但建议给，便于复盘。

---

## 3) 我们这边的“传入接口”（你们对接时怎么塞）

我们所有策略（狼人/村民/女巫等）统一吃一个结构化上下文 `StrategyContext`，其中：

- `context.inference.rankings`：**新版（推荐）**，排名形式（**你们第二组只需要提供这个**）
- `context.inference.known_players`：**旧版（兼容保留）**，数值概率形式（**第二组不要提供/不要填；可省略或置空**）

下面是**集成层把 rankings 塞进完整 StrategyContext** 的最小可用示例（注意：第二组不需要输出整段，只需要输出上一节的 rankings 本体）：

```json
{
  "meta": { "game_id":"g1", "agent_id":"a1", "role":"werewolf", "phase":"werewolf_night", "day_number": 2, "self_player_id":"player_001" },
  "public_state": { "alive_players":[{"id":"player_001"},{"id":"player_002"},{"id":"player_003"}], "dead_players":[] },
  "private_info": { "werewolf_partners":["player_003"] },
  "memory": { "memory_summary":"...", "recent_events":[] },
  "inference": {
    "rankings": {
      "werewolf_likelihood": ["player_002","player_001","player_003"],
      "villager_likelihood": ["player_001","player_003","player_002"]
    }
  },
  "constraints": { "allowed_actions":["kill"], "forbid_targets":[] }
}
```

---

## 4) 我们如何把排名转换成概率（便于你们理解效果）

- 先把排名映射成 [0,1] 分数（越靠前越接近1）
- 再把 `werewolf_likelihood` 映射到 `wolf_prob ∈ [0.1, 0.9]`
- 若同时给了 `villager_likelihood`，则会“拉低” wolf_prob（更像好人）

这保证：
- **狼人白天推票**：优先推“更像狼”的人（wolf_prob高）
- **狼人夜刀**：优先刀“更像好人/神职位”的人（good_prob高）
- **村民推理**：可以在没有数值概率的情况下正常排序与投票

