import uuid
import time
from datetime import datetime
from MemoryDB import AgentMemory, AgentConfig


# ================= 辅助工具 =================
def create_speech(day, agent_id, content):
    """生成发言事件"""
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "player_speech",
        "timestamp": datetime.now().isoformat(),
        "phase": "day_discussion",
        "data": {
            "day": day,
            "player_id": str(agent_id),
            "content": content
        }
    }


def create_vote(day, voter, target):
    """生成单人投票事件（为了更细致地模拟，我们拆分成单条记录或汇总记录）"""
    # 这里我们模拟系统发来的汇总结果
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "vote_result",
        "timestamp": datetime.now().isoformat(),
        "phase": "day_vote",
        "data": {
            "day": day,
            "votes": {str(voter): str(target)},  # 这里简化，实际系统可能是全员字典
            "result": f"{voter}号 投给了 {target}号"
        }
    }


def create_death(day, player_id, reason):
    """生成死亡事件"""
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "player_death",
        "timestamp": datetime.now().isoformat(),
        "phase": "night_reveal",
        "data": {
            "day": day,
            "player_id": str(player_id),
            "reason": reason
        }
    }


# ================= 主程序 =================

def run_full_simulation():
    # 1. 初始化 Agent 4 的记忆模块
    print("🎬 === 正在初始化 8人局 完整对局模拟 (视角: Agent 4) ===\n")
    config = AgentConfig(agent_id="agent_4", db_path="./memory_db")
    memory = AgentMemory(config)

    # 清理旧数据
    try:
        memory.client.delete_collection("events")
        memory.collection = memory.client.get_or_create_collection("events")
    except:
        pass

    # ==========================================
    # 📅 第一天：真假预言家对跳
    # ==========================================
    print("🌅 [Day 1] 游戏开始，7号玩家昨晚死亡...")

    events_day1 = [
        # 夜晚结果
        create_death(1, 7, "killed_by_werewolf"),

        # --- 警上/白天发言 ---
        # Agent 1 (悍跳狼) 起跳，查杀 3 (真预)
        create_speech(1, 1,
                      "我是全场唯一的预言家。昨晚验了3号，3号是查杀，没有任何容错。警徽流先验5号再验8号。3号不用拍身份了，直接出。"),

        # Agent 2 (平民) 还没分清局势
        create_speech(1, 2, "7号倒牌了？我觉得1号起跳的状态很饱满啊，3号如果是预言家得好好聊聊，不然我站1。"),

        # Agent 3 (真预言家) 原地干拔
        create_speech(1, 3,
                      "笑死我了，1号给我发查杀？我才是真预言家！昨晚验了1号是狼，本来想报金水的，结果抓到个悍跳的。1号是铁狼，全票出1！"),

        # Agent 5 (女巫) 保持沉默，暗中观察
        create_speech(1, 5,
                      "现在的局势是1、3对跳。7号昨晚走的，具体身份我不能说。我建议大家听完再投，不要盲目站边。目前我觉得3号逻辑更顺。"),

        # Agent 6 (冲锋狼) 疯狂攻击3号
        create_speech(1, 6, "3号聊的什么东西？被查杀才起跳，典型狼人视角。1号预言家面很大，我铁站边1号，今天必须出3！"),

        # Agent 8 (倒钩狼) 伪装好人，反向操作
        create_speech(1, 8,
                      "我不认同6号的观点。虽然1号起跳早，但3号的发言更诚恳。且1号的警徽流打得太随意了。我暂时保留意见，建议听听4号怎么说。"),

        # Agent 4 (你自己) 总结
        create_speech(1, 4,
                      "我是好人。目前看1号攻击性太强，3号防守逻辑清晰。8号的发言有点奇怪，像是在刻意做好身份。我建议今天先出1号，正视角。"),
    ]

    # 模拟存入 Day 1 发言
    for evt in events_day1:
        memory.add_event(evt, text_description="")  # 让系统自动生成文本

    # --- Day 1 投票阶段 (关键点) ---
    print("🗳️ [Day 1] 投票阶段...")
    votes_day1 = [
        # 狼队冲票
        create_vote(1, 1, 3),
        create_vote(1, 6, 3),
        # 倒钩狼卖队友
        create_vote(1, 8, 1),
        # 好人阵营 + 真预言家
        create_vote(1, 2, 1),  # 2号回头了
        create_vote(1, 3, 1),
        create_vote(1, 4, 1),
        create_vote(1, 5, 1)
    ]
    # 模拟存入 Day 1 投票
    vote_summary = {
        "votes": {"1":"3", "6":"3", "8":"1", "2":"1", "3":"1", "4":"1", "5":"1"},
        "result": {"exiled_player": "1", "description": "1号被放逐"} # 改成字典
    }
    memory.add_event({
        "event_id": str(uuid.uuid4()), "timestamp": datetime.now().isoformat(),
        "event_type": "vote_result", "data": {"day": 1, **vote_summary}
    })

    # Day 1 总结 (模拟夜晚 LLM 的复盘)
    memory.save_summary(1, "1号和3号对跳预言家，1号查杀3号。最终1号被放逐。6号铁站边1号，8号作为8号位玩家反水投了1号。")

    # ==========================================
    # 📅 第二天：双死局面
    # ==========================================
    print("\n🌅 [Day 2] 昨晚双死，3号和6号死亡...")

    events_day2 = [
        create_death(2, 3, "killed_by_werewolf"),  # 狼刀预言家
        create_death(2, 6, "poisoned"),  # 女巫毒冲锋狼

        # Agent 5 (女巫) 拍身份
        create_speech(2, 5,
                      "我摊牌了，我是女巫。昨晚7号是银水我没救（或者没药了）。昨晚3号倒牌，我毒了6号，因为6号昨天铁站边悍跳狼。现在的局势很清楚，场上还剩一狼。"),

        # Agent 2 (暴民) 懵逼
        create_speech(2, 2, "女巫厉害啊！那现在6号是狼走了，1号是狼走了。还剩谁？8号昨天投了1号，8号应该是好人吧？"),

        # Agent 8 (倒钩狼) 开始表演，嫁祸 Agent 2
        create_speech(2, 8,
                      "我也觉得我是好人。但2号你现在的发言很划水啊。昨天你也是摇摆不定，会不会你是那只深水狼？我建议女巫归票2号。"),

        # Agent 4 (你) 逻辑推理
        create_speech(2, 4,
                      "不对。8号昨天的票型太干净了，干净得像是在做身份。狼人看到1号大概率出局，卖队友是常规操作。2号虽然愚民，但不像狼。我怀疑8号是倒钩。"),
    ]

    for evt in events_day2:
        memory.add_event(evt, text_description="")

    # Day 2 投票：8号 PK 2号
    vote_summary_2 = {
        "votes": {"2": "8", "4": "8", "5": "8", "8": "2"},
        "result": {"exiled_player": "8", "description": "8号被放逐"}  # 改成字典
    }
    memory.add_event({
        "event_id": str(uuid.uuid4()), "timestamp": datetime.now().isoformat(),
        "event_type": "vote_result", "data": {"day": 2, **vote_summary_2}
    })

    memory.save_summary(2, "3号预言家倒牌，5号女巫毒死6号狼人。8号倒钩狼试图抗推2号，但被4号识破。最终8号被放逐，游戏结束。")

    # ==========================================
    # 🧠 深度与广度测试
    # ==========================================
    print("\n" + "=" * 50)
    print("🚀 记忆模块验收测试 (Perspective: Agent 4)")
    print("=" * 50)

    # 1. [广度测试] 检索特定玩家的所有行为
    print("\n🔍 1. 查底牌：分析 Agent 8 (倒钩狼) 的所有行为轨迹")
    print(memory.get_relevant_context("8号玩家", top_k=10))
    # 预期：Day1 他的发言很做好，投票投了1号；Day2 他开始踩2号。

    # 2. [深度测试] 跨天逻辑检索
    print("\n🔍 2. 查逻辑：谁是真预言家？为什么？")
    print(memory.get_relevant_context("真预言家", top_k=5))
    # 预期：搜出3号的发言，以及5号女巫认证3号的发言。

    # 3. [过滤测试] 查票型
    print("\n🔍 3. 查铁证：列出所有人的投票结果")
    print(memory.get_relevant_context("投票", type_filter="vote_result"))

    # 4. [标签测试] 谁悍跳了？
    print("\n🔍 4. 查标签：谁提到过'查杀'这个词？")
    tags = memory.search_by_tag("mentions_check")
    print(f"找到 {len(tags)} 条记录包含'查杀'。")
    # 预期：应该找到1号和3号的发言。

    # 5. [窗口限制测试] 模拟 LLM 提问
    print("\n🔍 5. 模拟 LLM：回顾整局游戏 (测试窗口限制)")
    # 我们故意设置一个很小的字符限制，看它是否会优先返回 Summary 或被截断
    print(memory.get_relevant_context("总结一下这局游戏", max_chars=2000))


if __name__ == "__main__":
    run_full_simulation()