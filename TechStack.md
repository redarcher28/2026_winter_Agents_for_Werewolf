# 狼人杀多Agent系统 - 个人技术栈学习指南

## 一、学习总则（10天快速学习策略）

### 学习优先级原则：

1. **够用即可**：学习深度以完成当前任务为准
2. **官方文档优先**：只看"Getting Started"和核心API部分
3. **代码优先**：通过阅读示例代码快速上手
4. **互相教学**：每日分享关键知识点
5. **问题驱动**：遇到具体问题再深入学习

### 每日学习时间分配：

- 上午1小时：学习新技术
- 下午30分钟：总结和分享
- 晚上30分钟：预习次日内容

---

## 二、第一小组：法官系统组

### 大组长（兼任第一小组长）

**核心任务**：法官整体架构、状态管理、团队协调

**技术栈学习清单**：

1. **Python核心**（复习，Day 1）
    - 重点：类与对象、数据结构（dict/list）、JSON序列化
    - 资源：[Python官方教程](https://docs.python.org/3/tutorial/classes.html)
    - 关键练习：实现一个GameState类，包含玩家列表、回合状态

2. **状态机设计**（Day 2-3）
    - 重点：有限状态机（FSM）概念、状态转换表
    - 资源：[Python状态机实现指南](https://github.com/pytransitions/transitions)
    - 关键练习：实现游戏阶段转换（夜晚→白天→投票）

3. **软件架构基础**（Day 1-2）
    - 重点：模块化设计、接口定义、依赖管理
    - 资源：[Python项目结构指南](https://docs.python-guide.org/writing/structure/)
    - 关键练习：设计法官系统模块划分图

4. **Git团队协作**（Day 1）
    - 重点：分支管理、合并冲突解决、Pull Request流程
    - 资源：[Git团队协作指南](https://www.atlassian.com/git/tutorials/comparing-workflows)
    - 关键练习：创建项目仓库，设置分支保护规则

**每日学习计划**：

- Day 1：Python类设计 + Git协作
- Day 2：状态机实现 + 模块设计
- Day 3：接口定义 + 团队协调实践
- Day 4：调试技巧 + 代码审查方法
- Day 5-10：基于问题驱动学习

### 组员A（流程控制与主循环）

**核心任务**：游戏主循环、阶段管理、超时处理

**技术栈学习清单**：

1. **Python基础**（Day 1）
    - 重点：循环控制、条件判断、函数定义
    - 资源：[Python控制流教程](https://docs.python.org/3/tutorial/controlflow.html)
    - 关键练习：编写游戏回合循环伪代码

2. **异步编程基础**（Day 2-3）
    - 重点：asyncio基本概念、async/await语法
    - 资源：[Python asyncio快速入门](https://docs.python.org/3/library/asyncio-task.html)
    - 关键练习：实现简单的异步任务调度

3. **超时与异常处理**（Day 4）
    - 重点：try/except/finally、超时装饰器
    - 资源：[Python异常处理](https://docs.python.org/3/tutorial/errors.html)
    - 关键练习：编写带超时限制的函数调用

4. **日志系统**（Day 5）
    - 重点：Python logging模块、日志级别配置
    - 资源：[Python Logging教程](https://docs.python.org/3/howto/logging.html)
    - 关键练习：为游戏事件添加结构化日志

**每日学习计划**：

- Day 1：Python基础控制流
- Day 2：异步编程概念
- Day 3：实现游戏状态机
- Day 4：超时与错误处理
- Day 5：日志系统集成
- Day 6-10：调试和优化

### 组员B（权限控制与API开发）

**核心任务**：信息过滤、API开发、通信协议

**技术栈学习清单**：

1. **Flask框架**（Day 1-2）
    - 重点：路由定义、请求处理、响应返回
    - 资源：[Flask快速入门](https://flask.palletsprojects.com/en/2.3.x/quickstart/)
    - 关键练习：创建一个返回JSON的简单API

2. **REST API设计**（Day 2-3）
    - 重点：HTTP方法、状态码、请求/响应格式
    - 资源：[REST API设计最佳实践](https://restfulapi.net/)
    - 关键练习：设计法官API端点（POST /action, GET /state）

3. **数据验证**（Day 3）
    - 重点：Pydantic模型、输入验证
    - 资源：[Pydantic快速开始](https://docs.pydantic.dev/latest/)
    - 关键练习：定义AgentContext数据模型

4. **信息权限控制**（Day 4）
    - 重点：访问控制列表、角色权限管理
    - 资源：[Python访问控制模式](https://www.geeksforgeeks.org/access-control-models/)
    - 关键练习：实现信息过滤函数（根据角色返回不同数据）

5. **WebSocket基础**（可选，Day 5）
    - 重点：Socket.IO或WebSocket基本概念
    - 资源：[Flask-SocketIO快速开始](https://flask-socketio.readthedocs.io/en/latest/getting_started.html)
    - 关键练习：实现简单的实时消息推送

**每日学习计划**：

- Day 1：Flask基础
- Day 2：REST API设计
- Day 3：数据验证与模型
- Day 4：权限控制实现
- Day 5：通信协议优化
- Day 6-10：API测试与安全

### 组员C（数据持久化与复盘）

**核心任务**：状态存储、日志管理、复盘功能

**技术栈学习清单**：

1. **JSON文件操作**（Day 1-2）
    - 重点：json模块、序列化/反序列化
    - 资源：[Python JSON教程](https://docs.python.org/3/library/json.html)
    - 关键练习：将GameState保存到JSON文件

2. **Python文件操作**（Day 2）
    - 重点：文件读写、路径管理、异常处理
    - 资源：[Python文件操作指南](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
    - 关键练习：实现自动化的游戏状态备份

3. **结构化日志**（Day 3）
    - 重点：日志格式、日志轮转、日志分析
    - 资源：[Python Logging高级配置](https://docs.python.org/3/howto/logging-cookbook.html)
    - 关键练习：配置结构化JSON日志

4. **数据查询与分析**（Day 4-5）
    - 重点：Pandas基础、数据筛选、简单统计
    - 资源：[Pandas 10分钟教程](https://pandas.pydata.org/docs/user_guide/10min.html)
    - 关键练习：从日志中提取游戏统计信息

5. **简单前端展示**（Day 6-7）
    - 重点：HTML基础、模板渲染、数据可视化
    - 资源：[Flask模板渲染](https://flask.palletsprojects.com/en/2.3.x/templating/)
    - 关键练习：创建游戏复盘HTML页面

**每日学习计划**：

- Day 1：JSON操作
- Day 2：文件系统管理
- Day 3：日志系统
- Day 4：数据查询基础
- Day 5：复盘功能开发
- Day 6-7：Web界面开发
- Day 8-10：优化与测试

---

## 三、第二小组：AI Agent框架组

### 小组长（Agent框架设计）

**核心任务**：Agent基类设计、框架集成、团队协调

**技术栈学习清单**：

1. **Python面向对象**（Day 1）
    - 重点：继承、多态、抽象类、设计模式
    - 资源：[Python OOP深入](https://realpython.com/python3-object-oriented-programming/)
    - 关键练习：设计Agent基类架构

2. **设计模式**（Day 2）
    - 重点：模板方法模式、策略模式、工厂模式
    - 资源：[Python设计模式](https://refactoring.guru/design-patterns/python)
    - 关键练习：实现Agent工厂方法

3. **异步消息处理**（Day 3）
    - 重点：消息队列、事件循环、回调
    - 资源：[Python异步编程模式](https://docs.python.org/3/library/asyncio-queue.html)
    - 关键练习：设计Agent消息处理管道

4. **错误处理与恢复**（Day 4）
    - 重点：异常传播、重试机制、降级策略
    - 资源：[Python错误处理最佳实践](https://realpython.com/python-exceptions/)
    - 关键练习：实现Agent的自动恢复机制

5. **性能优化基础**（Day 5）
    - 重点：性能分析、内存管理、并发控制
    - 资源：[Python性能优化指南](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
    - 关键练习：使用cProfile分析Agent性能

**每日学习计划**：

- Day 1：OOP高级特性
- Day 2：设计模式应用
- Day 3：异步架构设计
- Day 4：容错机制
- Day 5：性能监控
- Day 6-10：框架集成与调试

### 组员D（记忆模块开发）

**核心任务**：记忆存储、上下文管理、检索优化

**技术栈学习清单**：

1. **Python数据结构**（Day 1）
    - 重点：列表、字典、集合、队列
    - 资源：[Python数据结构](https://docs.python.org/3/tutorial/datastructures.html)
    - 关键练习：设计记忆存储结构（列表+字典）

2. **上下文窗口管理**（Day 2）
    - 重点：滑动窗口、优先级队列、摘要生成
    - 资源：[上下文管理算法](https://en.wikipedia.org/wiki/Sliding_window_protocol)
    - 关键练习：实现固定大小的记忆窗口

3. **简单检索算法**（Day 3）
    - 重点：关键词匹配、相似度计算
    - 资源：[Python字符串处理](https://docs.python.org/3/library/stdtypes.html#string-methods)
    - 关键练习：实现基于关键词的记忆检索

4. **记忆摘要生成**（Day 4）
    - 重点：文本摘要、信息提取
    - 资源：[文本摘要技术简介](https://en.wikipedia.org/wiki/Automatic_summarization)
    - 关键练习：使用LLM生成记忆摘要

5. **缓存机制**（Day 5）
    - 重点：LRU缓存、缓存失效策略
    - 资源：[Python functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)
    - 关键练习：实现记忆查询缓存

**每日学习计划**：

- Day 1：数据结构设计
- Day 2：上下文管理
- Day 3：检索算法
- Day 4：摘要技术
- Day 5：缓存优化
- Day 6-10：集成测试

### 组员E（LLM调用与输出解析）

**核心任务**：OpenAI API调用、提示词渲染、输出结构化

**技术栈学习清单**：

1. **OpenAI API基础**（Day 1-2）
    - 重点：API认证、请求格式、响应处理
    - 资源：[OpenAI API快速开始](https://platform.openai.com/docs/quickstart)
    - 关键练习：发送简单的ChatCompletion请求

2. **Python环境变量管理**（Day 1）
    - 重点：dotenv库、敏感信息保护
    - 资源：[python-dotenv教程](https://github.com/theskumar/python-dotenv)
    - 关键练习：安全存储API密钥

3. **提示词工程基础**（Day 2-3）
    - 重点：System/User/Assistant角色、few-shot示例
    - 资源：[OpenAI提示词指南](https://platform.openai.com/docs/guides/prompt-engineering)
    - 关键练习：设计Agent系统提示词模板

4. **输出结构化解析**（Day 3-4）
    - 重点：JSON解析、错误处理、格式验证
    - 资源：[Python json模块](https://docs.python.org/3/library/json.html)
    - 关键练习：解析LLM返回的结构化数据

5. **错误处理与重试**（Day 4-5）
    - 重点：API限流处理、网络错误重试
    - 资源：[Python重试库tenacity](https://tenacity.readthedocs.io/en/latest/)
    - 关键练习：实现带指数退避的重试机制

6. **成本控制与监控**（Day 5）
    - 重点：Token计数、API成本估算
    - 资源：[OpenAI Token计算](https://platform.openai.com/tokenizer)
    - 关键练习：实现简单的API使用监控

**每日学习计划**：

- Day 1：OpenAI API基础
- Day 2：提示词设计
- Day 3：输出解析
- Day 4：错误处理
- Day 5：成本优化
- Day 6-10：性能调优

---

## 四、第三小组：策略与集成组

### 小组长（狼人策略与对抗）

**核心任务**：狼人策略设计、对抗策略、团队协调

**技术栈学习清单**：

1. **提示词工程进阶**（Day 1-2）
    - 重点：角色扮演提示、思维链（Chain of Thought）
    - 资源：[高级提示词技术](https://www.promptingguide.ai/)
    - 关键练习：设计狼人伪装策略提示词

2. **博弈论基础**（Day 2-3）
    - 重点：纳什均衡、零和博弈、策略选择
    - 资源：[博弈论入门](https://plato.stanford.edu/entries/game-theory/)
    - 关键练习：分析狼人杀中的博弈场景

3. **行为模式分析**（Day 3-4）
    - 重点：玩家行为分类、异常检测
    - 资源：[行为分析基础](https://en.wikipedia.org/wiki/Behavioral_analytics)
    - 关键练习：识别狼人发言特征

4. **A/B测试方法**（Day 4-5）
    - 重点：实验设计、效果评估、迭代优化
    - 资源：[A/B测试指南](https://www.optimizely.com/optimization-glossary/ab-testing/)
    - 关键练习：设计策略对比实验

5. **数据分析基础**（Day 5）
    - 重点：基本统计、数据可视化
    - 资源：[Python数据分析基础](https://realpython.com/python-data-analysis/)
    - 关键练习：分析游戏对局数据

**每日学习计划**：

- Day 1：高级提示词技术
- Day 2：博弈论应用
- Day 3：行为模式识别
- Day 4：策略测试方法
- Day 5：数据分析
- Day 6-10：策略优化实践

### 组员F（神职策略开发）

**核心任务**：预言家和女巫策略、身份验证、资源管理

**技术栈学习清单**：

1. **逻辑推理提示词**（Day 1-2）
    - 重点：推理链设计、条件判断、证据评估
    - 资源：[逻辑推理提示词设计](https://www.anthropic.com/index/teaching-ai-systematically-reason)
    - 关键练习：设计预言家验人逻辑

2. **风险评估与决策**（Day 2-3）
    - 重点：风险收益分析、不确定性处理
    - 资源：[决策理论基础](https://en.wikipedia.org/wiki/Decision_theory)
    - 关键练习：实现女巫用药决策算法

3. **贝叶斯推理基础**（Day 3-4）
    - 重点：条件概率、贝叶斯更新
    - 资源：[贝叶斯思维](https://www.greenteapress.com/thinkbayes/html/thinkbayes001.html)
    - 关键练习：实现简单的身份概率更新

4. **协作策略设计**（Day 4）
    - 重点：信息共享、信任建立、团队协调
    - 资源：[多智能体协作](https://en.wikipedia.org/wiki/Multi-agent_system)
    - 关键练习：设计预言家-女巫协作机制

5. **对抗性思维**（Day 5）
    - 重点：反欺骗策略、信息验证
    - 资源：[对抗性机器学习基础](https://adversarial-ml-tutorial.org/)
    - 关键练习：设计对抗悍跳狼的策略

**每日学习计划**：

- Day 1：推理链设计
- Day 2：决策理论
- Day 3：概率推理
- Day 4：协作策略
- Day 5：对抗策略
- Day 6-10：策略实施与测试

### 组员G（村民策略与集成测试）

**核心任务**：村民策略、系统集成、测试开发

**技术栈学习清单**：

1. **测试驱动开发**（Day 1-2）
    - 重点：pytest框架、测试用例设计
    - 资源：[pytest快速入门](https://docs.pytest.org/en/stable/getting-started.html)
    - 关键练习：为Agent编写单元测试

2. **集成测试方法**（Day 2-3）
    - 重点：端到端测试、Mock对象、测试覆盖率
    - 资源：[Python集成测试指南](https://realpython.com/python-integration-testing/)
    - 关键练习：编写法官-Agent集成测试

3. **性能测试基础**（Day 3-4）
    - 重点：响应时间测试、负载测试、压力测试
    - 资源：[Python性能测试工具](https://locust.io/)
    - 关键练习：测试系统并发处理能力

4. **调试与诊断**（Day 4-5）
    - 重点：日志分析、断点调试、问题定位
    - 资源：[Python调试技巧](https://realpython.com/python-debugging-pdb/)
    - 关键练习：使用pdb调试游戏流程

5. **持续集成基础**（Day 5）
    - 重点：GitHub Actions、自动化测试
    - 资源：[GitHub Actions入门](https://docs.github.com/en/actions/quickstart)
    - 关键练习：配置简单的CI流水线

**每日学习计划**：

- Day 1：pytest基础
- Day 2：集成测试
- Day 3：性能测试
- Day 4：调试技术
- Day 5：CI/CD基础
- Day 6-10：测试实施与优化

---

## 五、跨组共享学习资源

### 所有人都需要的基础知识：

1. **Python基础语法**（Day 1）
    - 变量、数据类型、控制流
    - 函数定义与调用
    - 基础数据结构

2. **Git基础操作**（Day 1）
    - clone, add, commit, push
    - 分支创建与切换
    - 合并与冲突解决

3. **虚拟环境管理**（Day 1）
    - venv创建与激活
    - requirements.txt管理
    - 依赖安装

### 共享学习资料库：

1. **代码示例库**：
   ```
   /examples/
   ├── flask_api_example.py      # Flask API示例
   ├── openai_chat_example.py    # OpenAI调用示例
   ├── state_machine_example.py  # 状态机示例
   └── pytest_example.py         # 测试示例
   ```

2. **速查手册**：
   ```
   /cheatsheets/
   ├── python_cheatsheet.md      # Python语法速查
   ├── flask_cheatsheet.md       # Flask速查
   ├── openai_cheatsheet.md      # OpenAI API速查
   └── git_cheatsheet.md         # Git命令速查
   ```

### 每日学习检查点：

| Day | 所有成员必须完成            | 小组特定检查                      |
|-----|---------------------|-----------------------------|
| 1   | Python基础、Git配置、环境搭建 | 各组完成架构设计                    |
| 2   | 完成第一个Hello World程序  | 法官：基础API；Agent：基础类；策略：基础提示词 |
| 3   | 代码提交到Git，通过PR       | 完成模块接口定义                    |
| 4   | 完成第一次集成测试           | 系统能运行一个简单回合                 |
| 5   | 修复第一天发现的主要Bug       | 各组完成核心功能                    |
| 6   | 完成性能基准测试            | 优化响应时间                      |
| 7   | 完成用户文档初稿            | 准备演示用例                      |
| 8   | 完成最终集成测试            | 系统稳定运行3回合                   |
| 9   | 代码审查完成              | 准备最终演示                      |
| 10  | 项目演示成功              | 总结报告完成                      |

---

## 六、学习支持与求助渠道

### 1. 内部支持机制

- **结对编程**：每天安排1小时结对时间
- **技术分享会**：每天下午最后30分钟分享
- **代码评审**：每天合并前进行简单评审
- **问题白板**：共享问题列表，集体解决

### 2. 外部学习资源

**Python学习**：

- [Python官方教程](https://docs.python.org/3/tutorial/) - 最权威
- [Real Python](https://realpython.com/) - 实践导向
- [Python Crash Course](https://ehmatthes.github.io/pcc_2e/) - 快速上手

**Flask学习**：

- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world) - 完整教程
- [Flask官方文档](https://flask.palletsprojects.com/) - 权威参考

**OpenAI学习**：

- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) - 实用示例
- [OpenAI API文档](https://platform.openai.com/docs/api-reference) - API参考

### 3. 紧急求助策略

1. **遇到问题**：先尝试自己搜索15分钟
2. **小组内求助**：在小组成员群提问
3. **跨组求助**：在项目总群提问，标注[紧急]
4. **导师求助**：向有经验的组员请教
5. **外部求助**：Stack Overflow、GitHub Issues

### 4. 学习效果评估

- **每日小测验**：5个关键概念选择题
- **代码审查**：检查代码质量和理解程度
- **实践演示**：展示学习成果的实际应用
- **结对反馈**：同伴评估学习进展

---

**最后提醒**：10天学习要**聚焦核心**，不要追求完美。重点学习能**立即应用**的知识，通过**实践**巩固理解。每天都要有**代码产出
**，通过**解决问题**来学习。

**学习口号**：写代码是最好的学习方式！