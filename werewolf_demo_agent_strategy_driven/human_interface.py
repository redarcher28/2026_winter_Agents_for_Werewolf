"""
人类玩家接口 - 提供安全的交互界面
"""
import os
import time
from typing import Dict, List, Optional
from enum import Enum
import threading
from dataclasses import dataclass


class MessageType(Enum):
    """消息类型"""
    PUBLIC = "public"  # 公开信息（所有人可见）
    PRIVATE = "private"  # 私有信息（仅人类玩家可见）
    SYSTEM = "system"  # 系统信息
    ACTION_REQUIRED = "action"  # 需要行动
    WARNING = "warning"  # 警告信息


@dataclass
class GameMessage:
    """游戏消息"""
    type: MessageType
    content: str
    timestamp: str
    phase: str  # 游戏阶段
    sender: Optional[str] = None  # 发送者（如果适用）


class HumanInterface:
    """
    人类玩家接口类
    提供安全的交互界面，防止信息泄露
    """

    def __init__(self, player_id: str, role: str, character: str):
        self.player_id = player_id
        self.role = role
        self.character = character
        self.messages: List[GameMessage] = []
        self.actions_pending: bool = False
        self.current_prompt: Optional[str] = None
        self.user_input: Optional[str] = None
        self.input_lock = threading.Lock()
        self.input_ready = threading.Event()

        print(f"\n{'=' * 70}")
        print(f"人类玩家初始化".center(70))
        print(f"{'=' * 70}")
        print(f"玩家ID: {player_id}")
        print(f"角色: {role}")
        print(f"性格: {character}")
        print(f"{'=' * 70}\n")

    def clear_screen(self):
        """清屏（跨平台）"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def format_message(self, msg: GameMessage) -> str:
        """格式化消息显示"""
        colors = {
            MessageType.PUBLIC: "\033[94m",  # 蓝色
            MessageType.PRIVATE: "\033[92m",  # 绿色
            MessageType.SYSTEM: "\033[93m",  # 黄色
            MessageType.ACTION_REQUIRED: "\033[91m",  # 红色
            MessageType.WARNING: "\033[95m",  # 紫色
        }
        reset = "\033[0m"

        prefix = {
            MessageType.PUBLIC: "[公开]",
            MessageType.PRIVATE: "[私有]",
            MessageType.SYSTEM: "[系统]",
            MessageType.ACTION_REQUIRED: "[行动]",
            MessageType.WARNING: "[警告]",
        }

        color = colors.get(msg.type, "")
        return f"{color}{prefix[msg.type]} {msg.content}{reset}"

    def display_header(self, game_state: Dict):
        """显示头部信息"""
        print(f"\n{'=' * 70}")
        print(f"狼人杀 - 人类玩家界面".center(70))
        print(f"{'=' * 70}")
        print(
            f"玩家: {self.player_id} | 角色: {self.role} | 状态: {'存活' if game_state.get('is_alive', True) else '死亡'}")
        print(f"当前阶段: {game_state.get('phase', 'unknown')} | 第 {game_state.get('day', 0)} 天")
        print(f"存活玩家: {', '.join(game_state.get('alive_players', []))}")
        print(f"{'=' * 70}\n")

    def add_message(self, msg_type: MessageType, content: str, phase: str = "", sender: str = None):
        """添加消息"""
        from datetime import datetime
        msg = GameMessage(
            type=msg_type,
            content=content,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            phase=phase,
            sender=sender
        )
        self.messages.append(msg)

        # 如果是行动提示，立即显示
        if msg_type == MessageType.ACTION_REQUIRED:
            self.clear_screen()
            self.display_recent_messages(game_state={"phase": phase})
            print(f"\n\033[91m⚠ 需要你的行动！\033[0m")
            print(f"\033[91m{content}\033[0m\n")

    def get_recent_messages(self, limit: int = 20) -> List[GameMessage]:
        """获取最近消息"""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def display_recent_messages(self, game_state: Dict):
        """显示最近消息"""
        self.clear_screen()
        self.display_header(game_state)

        print("\n📝 游戏记录:")
        print("-" * 70)

        recent_msgs = self.get_recent_messages(20)
        for msg in recent_msgs:
            print(self.format_message(msg))

        print("-" * 70)

    def prompt_action(self, prompt: str, options: List[str] = None) -> str:
        """
        提示人类玩家行动
        """
        with self.input_lock:
            self.actions_pending = True
            self.current_prompt = prompt
            self.user_input = None
            self.input_ready.clear()

            # 显示提示
            self.clear_screen()
            self.display_recent_messages(game_state={"phase": "action"})

            print(f"\n\033[91m{'=' * 70}\033[0m")
            print(f"\033[91m⚠ 需要你的行动\033[0m")
            print(f"\033[91m{'=' * 70}\033[0m\033[0m")
            print(f"\n{prompt}")

            if options:
                print("\n可选选项:")
                for i, option in enumerate(options, 1):
                    print(f"  {i}. {option}")
                print(f"  0. 取消")

            # 获取输入
            while True:
                try:
                    if options:
                        choice = input("\n请输入选项编号: ").strip()
                        if choice == "0":
                            return "cancel"
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(options):
                                return options[idx]
                        except ValueError:
                            print("无效输入，请输入数字")
                    else:
                        user_input = input("\n请输入: ").strip()
                        if user_input:
                            return user_input
                except KeyboardInterrupt:
                    print("\n\n操作被中断")
                    return "cancel"

    def prompt_speech(self, game_state: Dict) -> str:
        """提示发言"""
        prompt = f"""
【发言阶段】第 {game_state.get('day', 0)} 天
你是 {self.player_id} ({self.role})，请发表你的观点。

发言要点：
1. 分析当前局势
2. 怀疑对象及理由
3. 你的建议

请输入你的发言内容（50-150字）：
"""
        return self.prompt_action(prompt)

    def prompt_vote(self, game_state: Dict) -> str:
        """提示投票"""
        alive_players = [p for p in game_state.get('alive_players', []) if p != self.player_id]

        if not alive_players:
            return self.player_id  # 理论上不会发生

        prompt = f"""
【投票阶段】第 {game_state.get('day', 0)} 天
请选择要投票放逐的玩家（不能投自己）。

当前存活玩家（除你之外）：
{alive_players}
"""

        options = alive_players + ["弃权"]
        choice = self.prompt_action(prompt, options)

        if choice == "弃权":
            # 弃权时随机选择一个（实际游戏不允许弃权，这里返回第一个玩家）
            return alive_players[0] if alive_players else self.player_id

        return choice

    def prompt_night_action(self, game_state: Dict, action_type: str) -> Dict:
        """提示夜晚行动"""
        if self.role == "werewolf":
            return self._prompt_werewolf_night(game_state)
        elif self.role == "seer":
            return self._prompt_seer_night(game_state)
        elif self.role == "witch":
            return self._prompt_witch_night(game_state)
        else:
            return {}

    def _prompt_werewolf_night(self, game_state: Dict) -> Dict:
        """狼人夜晚行动"""
        # 显示队友信息
        teammates = game_state.get('teammates', [])
        non_werewolf_players = [p for p in game_state.get('alive_players', [])
                                if p not in teammates and p != self.player_id]

        prompt = f"""
【狼人夜晚】第 {game_state.get('day', 0)} 天
你和你的狼人队友需要选择今晚击杀的目标。

你的队友：{teammates}
可击杀目标：{non_werewolf_players}

请选择要击杀的玩家：
"""

        choice = self.prompt_action(prompt, non_werewolf_players)
        return {"target": choice}

    def _prompt_seer_night(self, game_state: Dict) -> Dict:
        """预言家夜晚行动"""
        alive_players = [p for p in game_state.get('alive_players', []) if p != self.player_id]
        checked_players = game_state.get('checked_players', [])
        all_roles = game_state.get('all_roles', {})

        prompt = f"""
    【预言家夜晚】第 {game_state.get('day', 0)} 天
    你可以查验一名玩家的身份。

    已查验过的玩家：{checked_players}
    可查验目标：{alive_players}

    请选择要查验的玩家：
    """

        choice = self.prompt_action(prompt, alive_players)

        if choice and choice in alive_players:
            # 根据角色信息确定查验结果
            target_role = all_roles.get(choice, "unknown")
            is_werewolf = target_role == 'werewolf'

            return {
                "action": "check",
                "target": choice,
                "result": "狼人" if is_werewolf else "好人"
            }

        return {"action": "none"}

    def _prompt_witch_night(self, game_state: Dict) -> Dict:
        """女巫夜晚行动"""
        killed_player = game_state.get('killed_tonight')
        has_antidote = game_state.get('has_antidote', True)
        has_poison = game_state.get('has_poison', True)
        alive_players = [p for p in game_state.get('alive_players', []) if p != self.player_id]

        prompt = f"""
【女巫夜晚】第 {game_state.get('day', 0)} 天
今晚被狼人击杀的玩家：{killed_player if killed_player else '无'}

药水状态：
- 解药：{'可用' if has_antidote else '已用'}
- 毒药：{'可用' if has_poison else '已用'}

请选择行动：
"""

        options = []
        actions = []

        if killed_player and has_antidote:
            options.append(f"使用解药救 {killed_player}")
            actions.append(("save", killed_player))

        if has_poison:
            for player in alive_players:
                options.append(f"使用毒药毒 {player}")
                actions.append(("poison", player))

        options.append("不使用药水")
        actions.append(("none", None))

        choice_idx = self.prompt_action(prompt, options)

        if choice_idx == "不使用药水":
            return {"action": "none"}
        elif "cancel" in choice_idx:
            return {"action": "none"}
        else:
            # 查找对应的行动
            for i, option in enumerate(options):
                if option == choice_idx:
                    action_type, target = actions[i]
                    return {"action": action_type, "target": target}

            return {"action": "none"}

    def add_night_discussion(self, round_num: int, discussion: str):
        """添加狼人讨论消息"""
        self.add_message(
            MessageType.PRIVATE,
            f"狼人讨论第{round_num}轮: {discussion}",
            phase="werewolf_night"
        )

    def update_game_state(self, game_state: Dict):
        """更新游戏状态显示"""
        self.display_recent_messages(game_state)

    def wait_for_input(self, timeout: float = 30.0) -> bool:
        """等待输入（用于异步操作）"""
        return self.input_ready.wait(timeout=timeout)

    def get_input(self) -> Optional[str]:
        """获取用户输入"""
        with self.input_lock:
            return self.user_input

    def set_input(self, value: str):
        """设置用户输入"""
        with self.input_lock:
            self.user_input = value
            self.input_ready.set()
            self.actions_pending = False