# 创建WebSocket管理器实例

import json
from judge_system.web_socket import WebSocketManager

ws_manager = WebSocketManager()


# 注册回调函数
async def handle_action_response(game_id, player_id, action_data):
    print(f"收到行动响应: {player_id} -> {action_data}")
    return {"success": True, "processed": True}


ws_manager.register_callback("on_action_response", handle_action_response)


# 处理WebSocket连接
async def handle_websocket(websocket, game_id, player_id):
    # 注册连接
    player_info = {"role": "werewolf", "name": f"Player_{player_id}"}
    await ws_manager.register_connection(game_id, player_id, websocket, player_info)

    try:
        # 处理消息循环
        async for message in websocket:
            # 解析消息
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message

            # 处理消息
            result = await ws_manager.handle_message(game_id, player_id, data)

            # 发送响应
            await websocket.send_json(result)

    except Exception as e:
        print(f"WebSocket处理异常: {e}")
    finally:
        # 注销连接
        await ws_manager.unregister_connection(game_id, player_id)


# 启动夜晚阶段
async def start_night_phase(game_id):
    # 获取玩家角色信息
    werewolf_players = ["player1", "player3"]  # 示例狼人玩家
    seer_players = ["player2"]  # 示例预言家
    witch_players = ["player4"]  # 示例女巫

    # 处理夜晚阶段
    night_results = await ws_manager.process_night_phase(
        game_id,
        werewolf_players,
        seer_players,
        witch_players
    )

    print(f"夜晚阶段结果: {night_results}")
    return night_results


# 获取统计信息
stats = ws_manager.get_stats()
print(f"WebSocket管理器统计: {stats}")
