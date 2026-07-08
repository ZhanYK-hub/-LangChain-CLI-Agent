# -LangChain-CLI-Agent
本项目是一个基于 **LangChain** 框架构建的 CLI Agent，具备以下核心能力：

- 使用大语言模型（LLM）理解用户自然语言问题
- 自主决策是否调用工具（Tool Calling / ReAct 模式）
- 集成计算器、联网搜索、时间查询三类工具
- 支持单次提问与交互式对话两种运行模式
- 兼容 LangChain v1 新 API 与旧版回退方案

### 1.1 项目结构

```
agent4/
├── main.py              # CLI 入口：参数解析、交互循环
├── agent.py             # Agent 构建：模型初始化、三层 API 回退
├── tools.py             # 工具定义：@tool 装饰器注册
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
└── README.md            # 本文档
```

### 1.2 技术栈

| 组件 | 用途 |
|------|------|
| `langchain` | Agent 框架核心 |
| `langchain-openai` | OpenAI 兼容 LLM 接入 |
| `langgraph` | Agent 运行时（LangChain v1 底层） |
| `langchain-core` | `@tool` 装饰器、Message 抽象 |
| `python-dotenv` | `.env` 环境变量加载 |
| `duckduckgo-search` | 联网搜索工具后端 |

---

## 二、LangChain 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户 (CLI)                            │
│                   main.py  入口层                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ question
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent 编排层  agent.py                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ ChatOpenAI  │  │ SYSTEM_PROMPT│  │  create_agent()     │ │
│  │  (LLM 模型) │  │  (系统指令)  │  │  (LangGraph 运行时)  │ │
│  └──────┬──────┘  └──────────────┘  └──────────┬──────────┘ │
└─────────┼───────────────────────────────────────┼───────────┘
          │                                       │
          │         Agent Loop (ReAct)            │
          │  ┌──────────────────────────────────┐ │
          │  │  1. LLM 推理 → 决定是否调工具    │ │
          │  │  2. 解析 Tool Call               │ │
          │  │  3. 执行工具 → 获取 Observation  │ │
          │  │  4. 将结果反馈给 LLM             │ │
          │  │  5. 重复或输出最终答案            │ │
          │  └──────────────────────────────────┘ │
          │                                       │
          ▼                                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     工具层  tools.py                         │
│   calculator  │  web_search  │  get_current_time             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 LangChain Agent 核心概念

| 概念 | 说明 | 本项目对应 |
|------|------|-----------|
| **Model（模型）** | 负责推理与决策的 LLM | `ChatOpenAI`（gpt-4o-mini） |
| **Tools（工具）** | Agent 可调用的外部能力 | `calculator` / `web_search` / `get_current_time` |
| **Prompt（提示词）** | 约束 Agent 行为的系统指令 | `SYSTEM_PROMPT` |
| **Agent（智能体）** | 模型 + 工具 + 提示词 的运行时组合 | `create_agent()` 返回值 |
| **Messages（消息）** | Agent 状态的载体 | `HumanMessage` / `AIMessage` |
| **Graph（图）** | LangGraph 底层状态机，驱动 Agent 循环 | LangChain v1 内置 |

### 2.3 Agent 执行流程（ReAct 模式）

LangChain v1 的 `create_agent` 底层基于 **LangGraph** 实现，遵循 ReAct（Reasoning + Acting）范式：

```
用户输入
   │
   ▼
┌──────────┐    不需要工具     ┌──────────┐
│  LLM     │ ────────────────► │ 输出答案  │
│  推理    │                   └──────────┘
└────┬─────┘
     │ 需要工具
     ▼
┌──────────┐    工具结果       ┌──────────┐
│ Tool Call│ ────────────────► │  LLM     │ ──► 输出答案
│ 执行工具  │   (Observation)  │  再推理  │
└──────────┘                   └──────────┘
```

**示例：用户问「计算 sqrt(144) + 2**10」**

```
Step 1  LLM 推理  →  这是数学问题，需要 calculator
Step 2  Tool Call →  calculator("sqrt(144)+2**10")
Step 3  Observation → "1036"
Step 4  LLM 推理  →  组织自然语言回答
Step 5  输出      →  "sqrt(144) + 2^10 = 1036"
```

### 2.4 三层 API 兼容策略

`agent.py` 中的 `create_langchain_agent()` 按优先级自动选择 API，保证不同 LangChain 版本均可运行：

```
优先级 1（推荐）          优先级 2（回退）              优先级 3（经典）
langchain.agents          langgraph.prebuilt            langchain.agents
  create_agent()     →      create_react_agent()    →   create_tool_calling_agent()
  + LangGraph 运行时        + LangGraph 运行时            + AgentExecutor
  LangChain v1 新 API       旧版 ReAct API                最广泛兼容
```

---

## 三、模块功能详解

### 3.1 tools.py — 工具层

工具是 Agent 的「手脚」，通过 LangChain 的 `@tool` 装饰器注册。LLM 根据工具的 **名称** 和 **docstring** 决定何时调用。

#### calculator — 安全计算器

```python
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持 + - * / ** sqrt()，例如: sqrt(144)+2**10"""
```

| 属性 | 说明 |
|------|------|
| 输入 | 数学表达式字符串 |
| 输出 | 计算结果或错误信息 |
| 安全机制 | AST 白名单解析，禁止 `eval()` 执行任意代码 |
| 支持运算 | `+` `-` `*` `/` `**` `sqrt()` `abs()` `round()` |

#### web_search — 联网搜索

```python
@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。输入搜索关键词。"""
```

| 属性 | 说明 |
|------|------|
| 输入 | 搜索关键词 |
| 输出 | Top 3 搜索结果的标题 + 摘要 |
| 后端 | DuckDuckGo（无需 API Key） |
| 容错 | 网络失败时返回错误提示，不中断 Agent |

#### get_current_time — 时间查询

```python
@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
```

| 属性 | 说明 |
|------|------|
| 输入 | 无 |
| 输出 | `YYYY-MM-DD HH:MM:SS` 格式时间字符串 |

#### 工具注册

```python
ALL_TOOLS = [calculator, get_current_time, web_search]
```

所有工具汇总为列表，传入 `create_agent(tools=ALL_TOOLS)` 完成注册。

---

### 3.2 agent.py — Agent 构建层

#### 系统提示词（SYSTEM_PROMPT）

约束 Agent 的行为边界：

```
1. 需要精确计算时使用 calculator
2. 需要最新信息时使用 web_search
3. 需要知道时间时使用 get_current_time
4. 尽量用中文回答，简洁准确
5. 如果工具失败，诚实告知用户
```

#### 模型初始化（_build_model）

```python
ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.2,   # 低温度 → 更确定性的工具选择
)
```

支持任何 OpenAI 兼容接口（DeepSeek、Moonshot、通义千问等），只需修改 `.env` 中的 `OPENAI_BASE_URL`。

#### 统一调用接口（invoke_agent）

兼容不同 Agent 版本的返回格式：

```python
def invoke_agent(agent, question: str) -> str:
    # v1: {"messages": [HumanMessage, AIMessage, ...]}
    # 经典: {"output": "..."}
```

---

### 3.3 main.py — CLI 入口层

| 模式 | 命令 | 说明 |
|------|------|------|
| 单次提问 | `python main.py "你的问题"` | 问一次，输出答案 |
| 交互模式 | `python main.py -i` | 循环对话，输入 quit 退出 |
| 默认 | `python main.py` | 无参数时进入交互模式 |

**启动流程：**

```
解析命令行参数
    → create_langchain_agent()  构建 Agent
    → 打印可用工具列表
    → run_once() 或 run_interactive()
        → invoke_agent(agent, question)
        → 打印回答
```

---

## 四、配置说明

### 4.1 环境变量（.env）

```env
OPENAI_API_KEY=sk-your-key-here          # 必填
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，兼容接口地址
OPENAI_MODEL=gpt-4o-mini                 # 可选，默认 gpt-4o-mini
```

### 4.2 快速开始

```bash
cd c:\Users\张煜坤\Desktop\agent4
pip install -r requirements.txt
copy .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

python main.py "计算 sqrt(144) + 2**10 等于多少"
python main.py -i
```

---

## 五、LangChain 与纯 Python ReAct 的对比

本项目（agent4）使用 LangChain 框架，与 agent2 的纯 Python 实现形成对比：

| 维度 | agent2（纯 Python） | agent4（LangChain） |
|------|---------------------|---------------------|
| 依赖 | 仅标准库 | langchain + langgraph |
| Agent 循环 | 手写 while 循环 | LangGraph 状态机自动驱动 |
| 工具注册 | 手动 dispatch 字典 | `@tool` 装饰器 + 自动 schema |
| 工具选择 | 正则解析 Thought/Action | LLM 原生 Tool Calling |
| Prompt 管理 | 字符串拼接 | `system_prompt` 参数 |
| 流式输出 | 不支持 | 可扩展 `agent.stream()` |
| 可观测性 | 需手动实现 | 原生支持 LangSmith |
| 代码量 | ~150 行 | ~120 行（框架承担循环逻辑） |
| 适用场景 | 学习原理、零依赖 | 快速开发、生产部署 |

---

## 六、扩展方向

基于当前架构，以下改进可按优先级逐步实施：

### 6.1 短期（体验提升）

| 改进 | 说明 |
|------|------|
| 多轮记忆 | 接入 `ConversationBufferMemory`，交互模式记住上下文 |
| 流式输出 | 实现 `--stream`，逐 token 打印回答 |
| Verbose 模式 | `--verbose` 打印 Tool Call 链，便于调试 |

### 6.2 中期（生产就绪）

| 改进 | 说明 |
|------|------|
| LangSmith 追踪 | 环境变量接入，追踪每次 tool call 和 token 用量 |
| 安全护栏 | 移植 agent3 的敏感词过滤与拒答策略 |
| 语义缓存 | 相似问题跳过 LLM 调用，降低成本 |
| 模型降级 | 主模型失败自动切换备用模型 |

### 6.3 长期（工程化）

| 改进 | 说明 |
|------|------|
| pytest 测试集 | 工具层单测 + Agent 行为评估 |
| Docker 部署 | Dockerfile + docker-compose |
| Web API | FastAPI 封装，提供 HTTP 接口 |
| 多 Agent 协作 | LangGraph 自定义 Workflow |

---

## 七、关键代码索引

| 文件 | 函数/类 | 职责 |
|------|---------|------|
| `tools.py` | `calculator` | AST 安全数学计算 |
| `tools.py` | `web_search` | DuckDuckGo 联网搜索 |
| `tools.py` | `get_current_time` | 返回当前时间 |
| `tools.py` | `ALL_TOOLS` | 工具列表，供 Agent 注册 |
| `agent.py` | `_build_model()` | 初始化 ChatOpenAI |
| `agent.py` | `create_langchain_agent()` | 三层 API 回退构建 Agent |
| `agent.py` | `invoke_agent()` | 统一调用，兼容多版本返回格式 |
| `main.py` | `run_once()` | 单次问答 |
| `main.py` | `run_interactive()` | 交互式对话循环 |
| `main.py` | `main()` | CLI 入口 |

---

## 八、依赖关系图

```
main.py
  └── agent.py
        ├── langchain_openai.ChatOpenAI     (LLM)
        ├── langchain.agents.create_agent   (Agent 构建, v1)
        ├── langgraph.prebuilt              (Agent 构建, 回退)
        ├── langchain_core.messages         (消息格式)
        ├── dotenv                          (配置加载)
        └── tools.py
              ├── langchain_core.tools.@tool (工具注册)
              ├── ast / math                 (安全计算)
              └── duckduckgo_search          (联网搜索)
```
