#!/usr/bin/env python3
"""
人类玩家模式启动脚本
"""
import asyncio
import sys
import os

# 设置配置
os.environ['HUMAN_PLAYER_MODE'] = '1'

# 导入并运行主程序
from main import main

if __name__ == "__main__":
    # 设置人类玩家模式
    import game_config

    game_config.HUMAN_PLAYER = True
    game_config.VERBOSE_MODE = False  # 关闭详细日志，避免信息泄露

    # 运行游戏
    asyncio.run(main())