#!/usr/bin/env python3
"""
纯AI模式启动脚本
"""
import asyncio
import os

# 设置配置
os.environ['HUMAN_PLAYER_MODE'] = '0'

# 导入并运行主程序
from main import main

if __name__ == "__main__":
    # 设置纯AI模式
    import game_config

    game_config.HUMAN_PLAYER = False
    game_config.VERBOSE_MODE = True  # 可以开启详细日志

    # 运行游戏
    asyncio.run(main())