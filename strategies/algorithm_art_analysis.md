# Algorithm Art — Anthropic 「算法艺术」技术分析

## 1. 什么是 Algorithm Art（算法艺术）?

**Algorithm Art（算法艺术）** 是 Anthropic 在 Claude 3.7 Sonnet 及后续模型中引入的
**Extended Thinking（扩展思考）** 能力的非正式别称。

正式名称: **Extended Thinking / Extended Reasoning**

官方文档: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

### 核心思想

传统 LLM 直接从 prompt 生成 completion，而 Claude 的 Extended Thinking
允许模型在正式回复前进行**不受约束的内部推理（scratchpad thinking）**：

```
[用户问题]
    ↓
[隐藏思考块 (thinking block)]   ← 模型自由推理，不受输出格式约束
    ↓
[最终回复 (text block)]         ← 结合思考得出结论，输出结构化 JSON
```

这个"思考 → 回复"的过程被 Anthropic 形容为一种**艺术**——
模型像艺术家一样，将复杂算法（概率推理、博弈论、贝叶斯更新）
凝聚成优雅的最终决策，故名「**算法艺术**」。

---

## 2. Extended Thinking 的 API 调用方式

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")

response = client.messages.create(
    model="claude-opus-4-5",          # 需要支持 extended thinking 的模型
    max_tokens=16000,
    temperature=1,                     # extended thinking 强制要求 temperature=1
    thinking={
        "type": "enabled",
        "budget_tokens": 8000,         # 思考 token 预算（越大推理越深）
    },
    messages=[{
        "role": "user",
        "content": "你是狼人，请决定今晚刀人目标..."
    }],
)

# 遍历返回的内容块
for block in response.content:
    if block.type == "thinking":
        print("思考过程:", block.thinking)   # 内部推理（Algorithm Art 过程）
    elif block.type == "text":
        print("最终决策:", block.text)        # 结构化 JSON 输出
```

### 关键参数说明

| 参数 | 说明 | 约束 |
|------|------|------|
| `thinking.type` | `"enabled"` 开启 / `"disabled"` 关闭 | — |
| `thinking.budget_tokens` | 思考可使用的最大 token 数 | 最小 1024；不得超过 `max_tokens` |
| `temperature` | 温度 | **必须为 1**（extended thinking 时） |
| `max_tokens` | 总 token 上限（思考 + 回复） | 建议 ≥ 10000 |

---

## 3. 支持 Extended Thinking 的 Claude 模型

| 模型 | 是否支持 |
|------|---------|
| claude-3-7-sonnet-20250219 | ✅ |
| claude-opus-4-5-20251101 | ✅ |
| claude-sonnet-4-5-20251101 | ✅ |
| claude-3-5-sonnet-20241022 | ❌ |
| claude-3-5-haiku-20241022 | ❌ |

---

## 4. 在狼人杀场景中的应用价值

### 4.1 为什么 Extended Thinking 适合狼人杀?

狼人杀是一个**不完全信息博弈**，需要：
- **推断隐藏信息**（谁是狼人？谁是神职？）
- **多轮贝叶斯更新**（基于每次发言、投票不断修正判断）
- **欺骗与反欺骗**（狼人需要伪装，好人需要识破）
- **博弈论最优解**（投票时的纳什均衡）

这些任务都需要**深度推理**，而非简单模式匹配。
Extended Thinking 让 Claude 有时间在内部完成这些复杂计算。

### 4.2 标准模式 vs Algorithm Art 模式对比

```
标准模式（无 Extended Thinking）：
  prompt → [一步生成] → JSON 输出
  优点：速度快、成本低
  缺点：复杂推理容易出错，尤其在游戏后期信息复杂时

Algorithm Art 模式（Extended Thinking）：
  prompt → [内部思考: 分析局面 → 推断概率 → 模拟博弈] → JSON 输出
  优点：决策质量更高，推理链可追溯（存入 debug.thinking 字段）
  缺点：延迟增加（约 5-15 秒），token 消耗增加
```

### 4.3 在本项目中的具体收益

1. **狼人夜晚刀人（`decide_wolf_kill`）**
   - 思考：「玩家3是预言家的可能性最高（第1天他的发言暗示了信息来源），
     刀掉他比刀村民更有价值；但玩家7已经开始怀疑我，也需要考虑...」
   - 决策质量显著高于直接生成

2. **预言家查验（`decide_seer_check`）**
   - 思考：「已验玩家1为好人，玩家5发言风格与狼人行为模式吻合，
     查验玩家5的信息价值最大，但也需要保护自身安全...」

3. **白天辩论（`decide_*_speech`）**
   - 思考内部推理过程，生成更有逻辑、更具说服力的发言
   - 狼人可以生成更自然的"好人"发言，不会露出破绽

---

## 5. 本项目的 Anthropic 集成实现

见 `werewolf_demo_agent_strategy_driven/anthropic_llm_client.py`。

### 快速替换（从 OpenAI 切换到 Claude）

```python
from config import LLMConfig, AgentConfig
from anthropic_llm_client import AnthropicLLMClient

# 原来的 OpenAI 配置
openai_config = LLMConfig(
    provider="openai",
    api_key="sk-...",
    model="gpt-4",
)

# 替换为 Claude（Algorithm Art 模式）
claude_config = LLMConfig(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-opus-4-5",   # 支持 extended thinking
    temperature=1.0,            # extended thinking 要求
    max_tokens=16000,
)

# 创建客户端（接口与 LLMClient 完全相同）
client = AnthropicLLMClient(claude_config)

# 使用方式完全相同
decision = await client.decide_wolf_kill(context)
print(decision.data)          # {"action_type": "kill", "target_id": "player_3"}
print(decision.debug)         # {"reason": "...", "thinking": "<内部推理过程>"}
```

---

## 6. 成本与性能权衡

| 维度 | OpenAI GPT-4 | Claude 标准模式 | Claude Algorithm Art |
|------|-------------|----------------|----------------------|
| 响应延迟 | ~3s | ~3s | ~10-20s |
| 决策质量（复杂局面） | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Token 消耗 | 中 | 中 | 高（思考 budget + 回复）|
| 推理可追溯性 | ❌ | ❌ | ✅（thinking 块）|

**建议**：
- 游戏早期（信息少）：标准模式即可
- 游戏中后期（信息复杂）：启用 Algorithm Art 模式
- 实时约束较强时：可降低 `budget_tokens`（如 2000）以缩短延迟

---

## 7. 总结

「Algorithm Art」（算法艺术）本质上是 Anthropic Extended Thinking API 的形象化描述：

> Claude 将复杂的算法推理过程（概率论、博弈论、贝叶斯推断）
> 化为优雅的决策输出，如同将数学算法升华为艺术。

在狼人杀多 Agent 系统中，这个特性让 Claude Agent 拥有接近人类顶级玩家的
推理深度，能够处理信息博弈的复杂性，是目前最适合此场景的 LLM 能力之一。
