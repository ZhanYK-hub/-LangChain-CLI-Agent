#!/usr/bin/env python3
"""
LangChain Agent CLI 入口

用法:
    python main.py "计算 sqrt(144) + 2**10"
    python main.py                          # 交互模式
    python main.py --stream "今天天气如何"   # 流式输出（若支持）
"""
from __future__ import annotations

import argparse
import sys

from agent import create_langchain_agent, invoke_agent


def run_once(agent, question: str) -> str:
    print(f"\n问题: {question}\n")
    print("-" * 50)
    answer = invoke_agent(agent, question)
    print(f"回答: {answer}")
    print("-" * 50)
    return answer


def run_interactive(agent):
    print("LangChain Agent 交互模式（输入 quit 退出）\n")
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not q or q.lower() in {"quit", "exit", "q"}:
            print("再见！")
            break
        run_once(agent, q)


def main():
    parser = argparse.ArgumentParser(description="LangChain Agent")
    parser.add_argument("question", nargs="?", help="单次提问")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    try:
        agent = create_langchain_agent()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print("[Agent 已就绪，工具: calculator, web_search, get_current_time]")

    if args.interactive or not args.question:
        run_interactive(agent)
    else:
        run_once(agent, args.question)


if __name__ == "__main__":
    main()