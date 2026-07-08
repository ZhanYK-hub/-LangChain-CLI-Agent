"""
LangChain Agent 构建模块

优先使用 LangChain v1 的 create_agent；
若不可用则回退到 LangGraph 的 create_react_agent。
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """你是一个有用的 AI 助手，可以使用工具来回答问题。

规则：
1. 需要精确计算时使用 calculator
2. 需要最新信息时使用 web_search
3. 需要知道时间时使用 get_current_time
4. 尽量用中文回答，简洁准确
5. 如果工具失败，诚实告知用户
"""


def _build_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 OPENAI_API_KEY。请复制 .env.example 为 .env 并填入 Key。"
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.2,
    )


def create_langchain_agent():
    """创建 LangChain Agent 实例。"""
    model = _build_model()

    # LangChain v1 新 API
    try:
        from langchain.agents import create_agent
        return create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
    except ImportError:
        pass

    # 回退：LangGraph prebuilt ReAct Agent
    try:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(model, ALL_TOOLS, prompt=SYSTEM_PROMPT)
    except ImportError:
        pass

    # 回退：经典 Tool Calling Agent
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(model, ALL_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=True)


def invoke_agent(agent, question: str) -> str:
    """统一调用接口，兼容不同 Agent 返回格式。"""
    from langchain_core.messages import HumanMessage

    # create_agent (v1) 使用 messages 列表
    try:
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
    except TypeError:
        result = agent.invoke({"input": question})

    if isinstance(result, dict):
        if "messages" in result:
            last = result["messages"][-1]
            return getattr(last, "content", str(last))
        if "output" in result:
            return result["output"]
    return str(result)