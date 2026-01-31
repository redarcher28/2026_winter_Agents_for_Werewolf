# main.py
"""
主程序入口
"""
import asyncio
import os
import shutil
from game_config import *
from game_manager import GameManager


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
    
    # 创建游戏管理器
    print("正在初始化游戏...")
    game_manager = GameManager(api_keys)
    print("✓ 游戏初始化完成\n")
    
    input("按 Enter 键开始游戏...")
    
    # 运行游戏
    await game_manager.run_game(max_days=3)


if __name__ == "__main__":
    asyncio.run(main())
