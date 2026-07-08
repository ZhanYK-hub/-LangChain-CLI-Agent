"""
LangChain Agent 工具集

使用 @tool 装饰器定义，Agent 可自主决定何时调用。
"""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime

from langchain_core.tools import tool

# ---------- 安全计算器（AST 白名单）----------
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        op = _BIN.get(type(node.op))
        if not op:
            raise ValueError("unsupported")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        name = node.func.id
        return _FUNCS[name](*[_safe_eval(a) for a in node.args])
    raise ValueError("invalid")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持 + - * / ** sqrt()，例如: sqrt(144)+2**10"""
    try:
        result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
        return str(int(result) if isinstance(result, float) and result.is_integer() else result)
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。输入搜索关键词。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"未找到与「{query}」相关的结果"
        return "\n".join(f"- {r.get('title','')}: {r.get('body','')[:200]}" for r in results)
    except Exception as e:
        return f"搜索失败: {e}，请稍后重试或使用其他工具"


# 导出给 Agent 使用的工具列表
ALL_TOOLS = [calculator, get_current_time, web_search]