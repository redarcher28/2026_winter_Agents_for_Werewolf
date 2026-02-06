# main.py
"""
主程序入口
"""
import asyncio
import os
import shutil
from game_config import *
from game_manager import GameManager
from logger import init_logger, get_logger


def cleanup_memory():
    """清理旧的记忆数据库"""
    print("正在清理旧的记忆数据库...")
    
    paths_to_clean = [MEMORY_DB_BASE_PATH, SHARED_MEMORY_PATH]
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"  ✓ 已清理 {path}")
            except Exception as e:
                print(f"  ⚠ 清理 {path} 失败: {e}")
    
    print("✓ 记忆数据库清理完成\n")


async def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("狼人杀多 Agent 系统".center(70))
    print(f"{'='*70}\n")
    
    # 检查 API Keys
    api_keys = {}

    # 检查是否启用人类玩家
    from game_config import HUMAN_PLAYER, HUMAN_PLAYER_ID, HUMAN_PLAYER_ROLE

    if HUMAN_PLAYER:
        print("🎮 游戏模式: 混合模式（人类玩家 + AI玩家）")
        if HUMAN_PLAYER_ID:
            print(f"👤 人类玩家ID: {HUMAN_PLAYER_ID}")
        else:
            print("👤 人类玩家ID: 随机分配")
        print()
    else:
        print("🤖 游戏模式: 纯AI模式")
        print()
    
    # 优先使用 API_KEYS 字典
    if all(key != "your_api_key_here" and key.startswith("your_api_key_") == False 
           for key in API_KEYS.values()):
        api_keys = API_KEYS
        print(f"✓ 检测到 {len(api_keys)} 个独立 API Key")
    else:
        # 检查是否有环境变量或单个 API Key
        single_key = os.environ.get("SILICONFLOW_API_KEY", SILICONFLOW_API_KEY)
        if single_key != "your_api_key_here":
            # 所有玩家共享一个 API Key
            player_ids = [f"player_{i+1}" for i in range(TOTAL_PLAYERS)]
            api_keys = {player_id: single_key for player_id in player_ids}
            print(f"✓ 使用单个 API Key（所有玩家共享）")
        else:
            print("❌ 错误: 请先设置 API Key")
            print("\n方法1: 在 game_config.py 中设置 API_KEYS 字典（推荐）")
            print("方法2: 在 game_config.py 中设置 SILICONFLOW_API_KEY")
            print("方法3: 设置环境变量 SILICONFLOW_API_KEY")
            print("\n获取 API Key: https://siliconflow.cn/")
            return
    
    print(f"✓ 模型: {SILICONFLOW_MODEL}")
    print(f"✓ 玩家数: {TOTAL_PLAYERS}")
    print(f"✓ 角色配置: {ROLE_CONFIG}\n")
    
    # 清理旧记忆
    cleanup_memory()
    
    # 初始化日志系统
    if ENABLE_LOGGING:
        logger = init_logger(log_dir="./game_logs", game_id=GAME_ID)
        logger.log_section("游戏初始化")
        logger.log_event(f"模型: {SILICONFLOW_MODEL}")
        logger.log_event(f"玩家数: {TOTAL_PLAYERS}")
        logger.log_event(f"角色配置: {ROLE_CONFIG}")
    
    # 创建游戏管理器
    print("正在初始化游戏...")
    # game_manager = GameManager(api_keys)
    # 如果有指定的人类玩家ID，使用它；否则随机分配
    human_player_id = None

    if HUMAN_PLAYER:
        if HUMAN_PLAYER_ID and HUMAN_PLAYER_ID in [f"player_{i + 1}" for i in range(TOTAL_PLAYERS)]:
            human_player_id = HUMAN_PLAYER_ID
        else:
            # 随机选择一个玩家作为人类玩家
            import random
            human_player_id = f"player_{random.randint(1, TOTAL_PLAYERS)}"

        print(f"✓ 人类玩家设置为: {human_player_id}")
        if HUMAN_PLAYER_ROLE:
            print(f"✓ 人类玩家角色指定为: {HUMAN_PLAYER_ROLE}")
        else:
            print(f"✓ 人类玩家角色: 随机分配")

    game_manager = GameManager(api_keys, HUMAN_PLAYER, human_player_id, HUMAN_PLAYER_ROLE)
    print("✓ 游戏初始化完成\n")

    if HUMAN_PLAYER:
        print("=" * 70)
        print("重要提示:".center(70))
        print("=" * 70)
        print("1. 人类玩家将在一个专用界面中操作")
        print("2. 其他AI玩家的决策过程不会显示在人类界面中")
        print("3. 请按照界面提示进行操作")
        print("=" * 70)
        print()
    
    # 记录初始游戏状态
    if ENABLE_LOGGING:
        logger = get_logger()
        logger.log_game_state(0, game_manager.alive_players, game_manager.dead_players, game_manager.roles)
    
    input("按 Enter 键开始游戏...")
    
    # 运行游戏
    await game_manager.run_game(max_days=5, have_human=HUMAN_PLAYER)
    
    # 记录游戏结束
    if ENABLE_LOGGING:
        logger = get_logger()
        winner = game_manager.check_game_end()
        winner_name = "好人阵营" if winner == "good" else "狼人阵营" if winner == "werewolf" else "平局"
        logger.log_game_end(winner_name, {
            'alive_players': game_manager.alive_players,
            'dead_players': game_manager.dead_players,
            'roles': game_manager.roles
        })
        logger.close()
        print(f"\n✓ 完整游戏日志已保存到: {logger.log_file}")


if __name__ == "__main__":
    asyncio.run(main())
