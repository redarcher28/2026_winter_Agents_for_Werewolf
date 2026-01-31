# config_template.py
"""
配置模板 - 复制到 game_config.py 使用
"""

# ============================================================
# 方法 1：使用 6 个独立 API Key（推荐，速度最快）
# ============================================================
API_KEYS = {
    "player_1": "sk-rkrdcksxrxrnzkewfkthhdjcgekayhyuvwpkmbcqrizdwiyr",
    "player_2": "sk-dmdpqbedwynywhnmddhshxhpyieuvevgsezneqooptnqapfw",
    "player_3": "sk-ljclftxkubvhtwzoftzbtuyqgkrklyxqrdudiqmgenvknlrz",
    "player_4": "sk-xyadrjmeaxxhsmdbswdixsblwnjurxhmnlknhtdbrqycekgw",
    "player_5": "sk-efgvsshxyztfjinyswggfaxlorwhvkrcfzeycilcfubjjsmo",
    "player_6": "sk-uaixgfacyzyvvebnuaahvukzyoruiurcrmroabdyodkasfxy",
}

# ============================================================
# 方法 2：使用单个 API Key（备用，速度较慢）
# ============================================================
# SILICONFLOW_API_KEY = "sk-your-single-api-key-here"

# ============================================================
# 其他配置
# ============================================================
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V3"

TOTAL_PLAYERS = 6
GAME_ID = "multi_agent_game_001"

ROLES = {
    "player_1": "werewolf",
    "player_2": "werewolf",
    "player_3": "seer",
    "player_4": "witch",
    "player_5": "villager",
    "player_6": "villager"
}

WEREWOLF_DISCUSSION_ROUNDS = 3
MEMORY_DB_BASE_PATH = "./multi_agent_memory"
SHARED_MEMORY_PATH = "./shared_memory"

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1500
LLM_TIMEOUT = 60.0
