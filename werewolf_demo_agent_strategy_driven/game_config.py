# game_config.py
"""
游戏配置文件
"""

# 硅基流动 API 配置
# 如果你有多个 API Key，可以为每个玩家分配一个
API_KEYS = {
    "player_1": "sk-rkrdcksxrxrnzkewfkthhdjcgekayhyuvwpkmbcqrizdwiyr",
    "player_2": "sk-dmdpqbedwynywhnmddhshxhpyieuvevgsezneqooptnqapfw",
    "player_3": "sk-ljclftxkubvhtwzoftzbtuyqgkrklyxqrdudiqmgenvknlrz",
    "player_4": "sk-xyadrjmeaxxhsmdbswdixsblwnjurxhmnlknhtdbrqycekgw",
    "player_5": "sk-efgvsshxyztfjinyswggfaxlorwhvkrcfzeycilcfubjjsmo",
    "player_6": "sk-uaixgfacyzyvvebnuaahvukzyoruiurcrmroabdyodkasfxy",
}

# 如果只有一个 API Key，可以设置这个（会被所有玩家共享）
SILICONFLOW_API_KEY = "your_api_key_here"  # 备用单个 API Key

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V3"

# 游戏配置
TOTAL_PLAYERS = 6
GAME_ID = "multi_agent_game_001"

# 角色配置（数量）
ROLE_CONFIG = {
    "werewolf": 2,   # 狼人数量
    "seer": 1,       # 预言家数量
    "witch": 1,      # 女巫数量
    "villager": 2    # 村民数量
}

# 注意：角色会在每局游戏开始时随机分配
# 不再使用固定的 ROLES 字典

# 狼人配置
WEREWOLF_DISCUSSION_ROUNDS = 3  # 狼人讨论轮数

# 记忆数据库配置
MEMORY_DB_BASE_PATH = "./multi_agent_memory"
SHARED_MEMORY_PATH = "./shared_memory"  # 公共记忆

# LLM 配置
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1500
LLM_TIMEOUT = 60.0
