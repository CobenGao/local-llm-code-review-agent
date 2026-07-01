#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitLab AI-Powered Code Review Agent

一个基于 LangGraph 和本地 Ollama 大模型的自动 Python 代码评审系统。
集成 GitLab CI/CD 门禁，对每个 MR 的 Python 文件进行深度分析和规范检查。

核心流程：
  MR 变更 → Diff 解析 → AI 分析评审 → 格式校验/自愈重试 → 行间精准评论回传

技术栈：
  - LangGraph (StateGraph)：构建分析→评审→校验的闭环状态机
  - Ollama + qwen3-coder:30b：本地大模型推理
  - GitLab REST API v4：MR 信息获取和评论回传

使用场景：
  由 GitLab Runner (Shell Executor) 触发，在 MR 创建/更新时自动执行。

依赖安装：
  pip install langgraph requests
"""

# ────────────────────────── 标准库 ──────────────────────────
import json
import logging
import os
import re
import sys
from typing import List

# ────────────────────────── 第三方库 ──────────────────────────
import requests
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


# ════════════════════════════════════════════════════════════════
#  日志配置
# ════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai_agent_review")


# ════════════════════════════════════════════════════════════════
#  配置常量
# ════════════════════════════════════════════════════════════════

# Ollama 本地推理服务
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3-coder:30b")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "360"))

# LangGraph 重试机制
MAX_RETRY_COUNT: int = 3

# GitLab CI 环境变量（由 Runner 注入）
GITLAB_URL: str = (
    os.getenv("GITLAB_ACTUAL_URL") or os.getenv("CI_SERVER_URL", "http://localhost:9090")
).rstrip("/")
PROJECT_ID: str = os.getenv("CI_PROJECT_ID", "")
MR_IID: str = os.getenv("CI_MERGE_REQUEST_IID", "")
PROJECT_TOKEN: str = os.getenv("AI_REVIEW_TOKEN", "")
BASE_SHA: str = os.getenv("CI_MERGE_REQUEST_DIFF_BASE_SHA", "")
HEAD_SHA: str = (
    os.getenv("CI_MERGE_REQUEST_DIFF_HEAD_SHA") or os.getenv("CI_COMMIT_SHA", "")
)


# ════════════════════════════════════════════════════════════════
#  Agent 状态定义
# ════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    """LangGraph 状态机的数据结构"""

    diff_content: str  # 待评审的 Diff 全文
    file_path: str  # 文件相对路径
    context_code: str  # 辅助上下文（预留，未来可接向量检索）
    raw_response: str  # 模型原始输出
    structured_opinions: List[dict]  # 解析后的审查意见
    retry_count: int  # 格式错误重试次数
    last_error: str  # 上次格式错误信息


# ════════════════════════════════════════════════════════════════
#  AI 评审系统提示词
# ════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """\
你是一名资深 Python 代码评审专家，熟悉 PEP 8、安全漏洞与工程规范。
你的任务是对提供的 Python 代码 Diff 进行全面、严格的评审，识别下述三大类问题。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A. Python 代码审查准则（命名规范 / 排版格式 / 语法漏洞）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 命名规范
   • 函数、变量、模块属性名 → 下划线命名法（snake_case）
   • 类名                   → 大驼峰命名法（PascalCase）
   • 模块级常量             → 全大写（UPPER_CASE）
   • 严禁无意义单字母命名（除循环计数器 i / j / k 外）

2. 排版格式
   • 魔鬼数字：直写硬编码常数而未定义具名常量（如 `time.sleep(86400)`）
   • 缺失 Docstring 或核心注释的公共函数 / 类
   • 函数体超过 50 行建议拆分

3. 语法与安全漏洞
   • 【高频易错】可变默认参数：`def func(lst=[])` 必须改为 `def func(lst=None)`
     并在函数体内部执行 `if lst is None: lst = []`
   • 多线程竞争冒险：共享资源未使用 threading.Lock 保护
   • 危险调用：eval() / exec() 的使用
   • 宽泛异常吞噬：裸 `except: pass` 或 `except Exception: pass`（无任何处理）
   • SQL 注入风险：字符串拼接构造 SQL 而非使用参数化查询
   • 资源泄漏：文件/连接未使用 `with` 语句管理

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  B. GitLab Suggestion（一键 Apply）格式规约
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在 comment 字段中给出修复建议时，修复代码必须使用如下格式包裹，
这样 GitLab MR 页面会渲染出可一键点击的 "Apply suggestion" 蓝色按钮：

```suggestion
<修改后的完整正确代码行>
```

注意：suggestion 块内只放修改后的代码，不要添加解释文字。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  C. 输出格式约束（强制 JSON Schema）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

你必须且只能输出纯粹的 JSON 数组，不允许包含任何外围解释性文字或 Markdown 标记。
格式严格如下：

[
  {
    "line_number": <新文件中错误所在行号，整数>,
    "category": "<命名规范 | 排版格式 | 语法漏洞>",
    "level": "WARNING",
    "comment": "<深度剖析原因>。\\n\\n```suggestion\\n<修改后的完整代码行>\\n```"
  }
]

• 若代码 Diff 中未发现任何问题，输出空数组：[]
• line_number 对应新文件（new_path）中的实际行号，需从 Diff hunk 头 @@ 信息推算
• 只分析以 "+" 开头的新增行（忽略以 "-" 开头的删除行）
• 绝对禁止在 JSON 数组之外输出任何字符
"""


# ════════════════════════════════════════════════════════════════
#  Ollama 接口封装
# ════════════════════════════════════════════════════════════════
def call_ollama(messages: list) -> str:
    """
    调用本地 Ollama 推理服务。

    Args:
        messages: 对话消息列表，每条消息包含 role 和 content。

    Returns:
        模型返回的文本内容。

    Raises:
        RuntimeError: 网络连接、超时或响应解析异常。
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": 0.1,
            "num_ctx": 65536,
            "top_p": 0.9,
        },
    }
    logger.info(
        f"→ 调用 Ollama  model={OLLAMA_MODEL}  "
        f"turns={len(messages)}  ctx_window=65536"
    )

    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama 请求超时（>{OLLAMA_TIMEOUT}s），"
            "模型可能仍在加载或负载过高。"
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"无法连接 Ollama 服务 [{url}]：{exc}")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama HTTP {resp.status_code}：{exc}  "
            f"响应：{resp.text[:400]}"
        )

    try:
        data = resp.json()
        content: str = data["message"]["content"]
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"Ollama 响应解析失败：{exc}  原始：{resp.text[:400]}"
        )

    logger.info(f"← Ollama 返回 {len(content)} 字符")
    return content


# ════════════════════════════════════════════════════════════════
#  JSON 处理工具函数
# ════════════════════════════════════════════════════════════════
def _sanitize_json_control_chars(text: str) -> str:
    """
    将 JSON 字符串值内的裸控制字符转义。

    大模型有时会在 JSON 字符串中直接包含换行符而非 \\n，
    导致 json.loads() 抛出 "Invalid control character" 错误。
    此函数对文本中的字符串值内部的 \\n / \\r / \\t 进行转义。

    Args:
        text: 待处理的 JSON 文本。

    Returns:
        已转义的 JSON 文本。
    """
    result: list = []
    in_string: bool = False
    escape_next: bool = False

    for char in text:
        if escape_next:
            # 当前字符是转义序列的第二个字符，原样保留
            result.append(char)
            escape_next = False
        elif char == "\\" and in_string:
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif in_string and char == "\n":
            result.append("\\n")
        elif in_string and char == "\r":
            result.append("\\r")
        elif in_string and char == "\t":
            result.append("\\t")
        else:
            result.append(char)

    return "".join(result)


def _extract_json_array(text: str) -> str:
    """
    从模型输出中提取有效的 JSON 数组文本。

    处理常见的格式问题：
      - 被 ```json ``` 代码块包裹
      - 前后包含解释性文字
      - 字符串值内含未转义的控制字符

    Args:
        text: 模型的原始输出文本。

    Returns:
        可直接传入 json.loads() 的 JSON 数组字符串。
    """
    # 优先尝试提取代码块内容
    fence_re = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    for block in fence_re.findall(text):
        stripped = block.strip()
        if stripped.startswith("["):
            return _sanitize_json_control_chars(stripped)

    # 直接查找最外层的 [ ... ] 区间
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        return _sanitize_json_control_chars(candidate)

    return _sanitize_json_control_chars(text.strip())


# ════════════════════════════════════════════════════════════════
#  LangGraph 节点实现
# ════════════════════════════════════════════════════════════════
def analyze_and_retrieve(state: AgentState) -> AgentState:
    """
    节点1：初始化状态并准备背景信息。

    当前版本注入默认的项目背景；未来可集成向量检索来获取
    相关的历史代码片段和规范指南。
    """
    logger.info(f"[Node-1] 初始化分析  file={state['file_path']}")

    default_context = (
        "# 项目背景：Python 3.10+ 代码，遵循 PEP 8 规范。\n"
        "# 重点关注：可变默认参数、裸异常、资源管理、命名规范等。"
    )

    return {
        **state,
        "context_code": state.get("context_code") or default_context,
        "structured_opinions": [],
        "retry_count": state.get("retry_count", 0),
        "last_error": state.get("last_error", ""),
    }


def execute_review(state: AgentState) -> AgentState:
    """
    节点2：调用 AI 模型执行代码评审。

    拼接系统提示词、代码 Diff 和背景信息，
    请求 Ollama 模型并保存输出。
    """
    retry_count = state.get("retry_count", 0)
    last_error = state.get("last_error", "")

    logger.info(
        f"[Node-2] 执行评审  attempt={retry_count + 1}  "
        f"file={state['file_path']}"
    )

    # 构建用户请求
    sections = [
        f"## 待评审文件\n`{state['file_path']}`\n",
        f"## 项目背景\n{state['context_code']}\n",
        (
            f"## 代码 Diff\n"
            f"```diff\n{state['diff_content']}\n```\n"
        ),
    ]

    # 重试时附加错误信息，引导模型修正输出格式
    if retry_count > 0 and last_error:
        sections.append(
            f"\n## ⚠️ 格式修正提醒（第 {retry_count} 次重试）\n"
            f"上次输出因格式错误被拒绝：\n"
            f"```\n{last_error}\n```\n"
            f"**请输出纯 JSON 数组，不包含任何解释文字。**"
        )

    user_content = "\n".join(sections)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw_response = call_ollama(messages)
    except RuntimeError as exc:
        logger.error(f"[Node-2] 评审异常：{exc}")
        raw_response = ""

    return {**state, "raw_response": raw_response}


def node_validate_format(state: AgentState) -> AgentState:
    """
    节点3：校验 AI 输出的 JSON 格式。

    流程：清洗输出 → JSON 解析 → Schema 校验 →
      成功：存储审查意见
      失败：记录错误，触发重试
    """
    logger.info(
        f"[Node-3] 格式校验  attempt={state.get('retry_count', 0) + 1}"
    )

    raw = state.get("raw_response", "").strip()

    if not raw:
        err = "模型返回空响应"
        logger.warning(f"[Node-3] {err}")
        return {
            **state,
            "structured_opinions": [],
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": err,
        }

    cleaned = _extract_json_array(raw)
    logger.debug(f"[Node-3] 清洗后（前 300 字）：{cleaned[:300]}")

    try:
        parsed = json.loads(cleaned)

        # 校验顶层结构
        if not isinstance(parsed, list):
            raise ValueError(
                f"顶层必须是数组，实际：{type(parsed).__name__}"
            )

        # 校验每个审查意见
        required_keys = {"line_number", "category", "level", "comment"}
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"元素 {idx} 不是字典：{item!r}")
            missing = required_keys - item.keys()
            if missing:
                raise ValueError(
                    f"元素 {idx} 缺少字段：{missing}"
                )
            try:
                item["line_number"] = int(item["line_number"])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"元素 {idx} 的 line_number 无法转为整数：{exc}"
                )

        logger.info(f"[Node-3] ✓ 校验成功，共 {len(parsed)} 条意见")
        return {
            **state,
            "structured_opinions": parsed,
            "last_error": "",
        }

    except (json.JSONDecodeError, ValueError) as exc:
        err_msg = str(exc)
        new_retry = state.get("retry_count", 0) + 1
        context = cleaned[:400] if cleaned else "(空)"
        full_error = f"{err_msg}\n\n输出片段：\n{context}"

        logger.warning(f"[Node-3] ✗ 校验失败（第 {new_retry} 次）：{err_msg}")
        logger.debug(f"[Node-3] 原始输出：{raw[:600]}")

        return {
            **state,
            "structured_opinions": [],
            "retry_count": new_retry,
            "last_error": full_error,
        }


# ════════════════════════════════════════════════════════════════
#  条件路由
# ════════════════════════════════════════════════════════════════
def route_after_validation(state: AgentState) -> str:
    """
    校验后的流程分支：

    • 校验成功 → 结束
    • 校验失败且未超重试次数 → 重新评审
    • 超过重试上限 → 结束（跳过该文件）
    """
    opinions = state.get("structured_opinions", [])
    retry_count = state.get("retry_count", 0)

    if opinions:
        logger.info("[Router] 校验成功 → 结束")
        return END

    if retry_count >= MAX_RETRY_COUNT:
        logger.warning(
            f"[Router] 达到重试上限 ({MAX_RETRY_COUNT}) → 结束"
        )
        return END

    logger.info(f"[Router] 格式错误，触发第 {retry_count} 次重试")
    return "execute_review"


# ════════════════════════════════════════════════════════════════
#  状态机构建
# ════════════════════════════════════════════════════════════════
def build_agent_graph():
    """
    构建 LangGraph 状态机。

    拓扑：

        [START]
           ↓
    analyze_and_retrieve
           ↓
      execute_review  ←─────────────────┐
           ↓                            │ (重试循环)
    node_validate_format                │
           ↓                            │
        [END] ──────────────────────────┘
    """
    graph = StateGraph(AgentState)

    graph.add_node("analyze_and_retrieve", analyze_and_retrieve)
    graph.add_node("execute_review", execute_review)
    graph.add_node("node_validate_format", node_validate_format)

    graph.set_entry_point("analyze_and_retrieve")
    graph.add_edge("analyze_and_retrieve", "execute_review")
    graph.add_edge("execute_review", "node_validate_format")

    graph.add_conditional_edges(
        "node_validate_format",
        route_after_validation,
        {
            "execute_review": "execute_review",
            END: END,
        },
    )

    return graph.compile()


# ════════════════════════════════════════════════════════════════
#  GitLab API 接口
# ════════════════════════════════════════════════════════════════
def get_mr_changes(
    gitlab_url: str,
    project_id: str,
    mr_iid: str,
    token: str,
) -> list:
    """
    获取 MR 中的 Python 文件变更列表。

    Returns:
        变更信息字典列表，包含 new_path / old_path / diff 等字段。

    Raises:
        requests.exceptions.RequestException: 网络或 API 错误。
    """
    url = (
        f"{gitlab_url}/api/v4/projects/{project_id}/"
        f"merge_requests/{mr_iid}/changes"
    )
    headers = {"PRIVATE-TOKEN": token}

    logger.info(f"获取 MR 变更：{url}")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error(f"获取 MR 变更失败：{exc}")
        raise

    all_changes = resp.json().get("changes", [])
    python_changes = [
        c
        for c in all_changes
        if c.get("new_path", "").endswith(".py")
        and not c.get("deleted_file", False)
    ]
    logger.info(
        f"总计 {len(all_changes)} 个文件变更，"
        f"其中 Python 文件 {len(python_changes)} 个"
    )
    return python_changes


def _post_inline_discussion(
    gitlab_url: str,
    project_id: str,
    mr_iid: str,
    token: str,
    note_body: str,
    change: dict,
    line_number: int,
    base_sha: str,
    head_sha: str,
) -> bool:
    """
    发表行间定位的 Discussion。

    Parameters:
        note_body: 评论内容（支持 Markdown）。
        change: GitLab 返回的变更信息。
        line_number: 新文件中的行号。
        base_sha, head_sha: Commit SHA 标识。

    Returns:
        成功返回 True，失败返回 False。
    """
    url = (
        f"{gitlab_url}/api/v4/projects/{project_id}/"
        f"merge_requests/{mr_iid}/discussions"
    )
    headers = {"PRIVATE-TOKEN": token, "Content-Type": "application/json"}

    # 若 change 中携带了真实 old_path（如重命名场景）则优先使用它。
    old_path = change.get("old_path") or change["new_path"]

    payload = {
        "body": note_body,
        "position": {
            "position_type": "text",
            "base_sha": base_sha,
            "start_sha": base_sha,
            "head_sha": head_sha,
            "old_path": old_path,
            "new_path": change["new_path"],
            "new_line": int(line_number),
        },
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code in (200, 201):
            return True
        logger.warning(
            f"行间评论失败 HTTP {resp.status_code}  "
            f"{change['new_path']}:{line_number}  "
            f"响应：{resp.text[:300]}"
        )
        return False
    except requests.exceptions.RequestException as exc:
        logger.error(f"行间评论异常：{exc}")
        return False


def _post_mr_note(
    gitlab_url: str,
    project_id: str,
    mr_iid: str,
    token: str,
    note_body: str,
) -> bool:
    """
    在 MR 顶层发表普通评论（降级方案）。

    Returns:
        成功返回 True，失败返回 False。
    """
    url = (
        f"{gitlab_url}/api/v4/projects/{project_id}/"
        f"merge_requests/{mr_iid}/notes"
    )
    headers = {"PRIVATE-TOKEN": token, "Content-Type": "application/json"}
    try:
        resp = requests.post(
            url, headers=headers, json={"body": note_body}, timeout=30
        )
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as exc:
        logger.error(f"顶层评论异常：{exc}")
        return False


def post_discussion(
    gitlab_url: str,
    project_id: str,
    mr_iid: str,
    token: str,
    opinion: dict,
    change: dict,
    base_sha: str,
    head_sha: str,
) -> bool:
    """
    发表审查意见。

    优先尝试行间精准定位；若失败则降级为 MR 顶层评论。

    Args:
        opinion: 审查意见（含 line_number / category / level / comment）。
        change: GitLab 变更信息。
        base_sha, head_sha: Diff 基础和头部 Commit SHA。

    Returns:
        至少一种方式成功返回 True。
    """
    line_number: int = int(opinion["line_number"])

    note_body = (
        "🤖 **AI Code Review**\n\n"
        f"⚠️ **类别**：`{opinion['category']}` | "
        f"**级别**：`{opinion['level']}`\n\n"
        f"💡 **评审意见**：{opinion['comment']}"
    )

    # 尝试行间评论
    success = _post_inline_discussion(
        gitlab_url,
        project_id,
        mr_iid,
        token,
        note_body,
        change,
        line_number,
        base_sha,
        head_sha,
    )
    if success:
        logger.info(
            f"✓ 行间评论已发表  {change['new_path']}:{line_number}  "
            f"({opinion['category']})"
        )
        return True

    # 行间定位失败 → 降级
    logger.warning(
        f"行间定位失败，降级为顶层评论  "
        f"{change['new_path']}:{line_number}"
    )
    fallback_body = (
        "🤖 **AI Code Review** *(行定位失败，已降级)*\n\n"
        f"📄 **文件**：`{change['new_path']}`  "
        f"**行号**：`{line_number}`\n\n"
        f"⚠️ **类别**：`{opinion['category']}` | "
        f"**级别**：`{opinion['level']}`\n\n"
        f"💡 **评审意见**：{opinion['comment']}"
    )
    success = _post_mr_note(gitlab_url, project_id, mr_iid, token, fallback_body)
    if success:
        logger.info(
            f"✓ 顶层评论已发表（降级）  {change['new_path']}:{line_number}"
        )
    return success


# ════════════════════════════════════════════════════════════════
#  环境校验
# ════════════════════════════════════════════════════════════════
def validate_env() -> bool:
    """
    校验必填的 GitLab CI 环境变量。

    Returns:
        全部检验通过返回 True。
    """
    required = {
        "CI_PROJECT_ID": PROJECT_ID,
        "CI_MERGE_REQUEST_IID": MR_IID,
        "AI_REVIEW_TOKEN": PROJECT_TOKEN,
        "CI_MERGE_REQUEST_DIFF_BASE_SHA": BASE_SHA,
        "CI_MERGE_REQUEST_DIFF_HEAD_SHA / CI_COMMIT_SHA": HEAD_SHA,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        logger.error(f"✗ 缺少环境变量：{missing}")
        return False
    return True


# ════════════════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════════════════
def main() -> None:
    """
    主流程：

    1. 校验 CI 环境
    2. 拉取 MR 中的 Python 文件变更
    3. 逐文件运行 AI 评审状态机
    4. 将意见精准回传至 GitLab MR
    """
    DIVIDER = "═" * 64
    logger.info(DIVIDER)
    logger.info("  GitLab AI Code Review  启动")
    logger.info(DIVIDER)
    logger.info(f"  GitLab URL  : {GITLAB_URL}")
    logger.info(f"  Project ID  : {PROJECT_ID}")
    logger.info(f"  MR IID      : {MR_IID}")
    logger.info(
        f"  AI Model    : {OLLAMA_MODEL}  @{OLLAMA_BASE_URL}"
    )
    logger.info(
        f"  Base SHA    : {BASE_SHA[:14]}..."
        if len(BASE_SHA) > 14
        else f"  Base SHA    : {BASE_SHA}"
    )
    logger.info(
        f"  Head SHA    : {HEAD_SHA[:14]}..."
        if len(HEAD_SHA) > 14
        else f"  Head SHA    : {HEAD_SHA}"
    )
    logger.info(DIVIDER)

    # 环境校验
    if not validate_env():
        logger.error("✗ 环境校验失败，退出。")
        sys.exit(1)

    # 获取变更文件
    try:
        changes = get_mr_changes(GITLAB_URL, PROJECT_ID, MR_IID, PROJECT_TOKEN)
    except requests.exceptions.RequestException as exc:
        logger.error(f"✗ 获取 MR 变更失败，退出：{exc}")
        sys.exit(1)

    if not changes:
        logger.info("✓ 本次 MR 无 Python 文件变更，评审完成。")
        sys.exit(0)

    # 编译状态机
    agent_graph = build_agent_graph()
    logger.info("✓ 状态机编译完成\n")

    # 统计
    total_files = len(changes)
    total_opinions = 0
    total_posted = 0
    total_inline = 0
    total_fallback = 0

    # 逐文件评审
    for file_idx, change in enumerate(changes, start=1):
        file_path = change.get("new_path", "unknown")
        diff_content = change.get("diff", "")

        logger.info(f"\n{'─' * 64}")
        logger.info(
            f"[{file_idx}/{total_files}] 评审开始  "
            f"{file_path}  ({len(diff_content)} 字符)"
        )

        if not diff_content.strip():
            logger.info("  Diff 为空，跳过。")
            continue

        # 初始化状态
        initial_state: AgentState = {
            "diff_content": diff_content,
            "file_path": file_path,
            "context_code": "",
            "raw_response": "",
            "structured_opinions": [],
            "retry_count": 0,
            "last_error": "",
        }

        # 运行评审
        try:
            final_state = agent_graph.invoke(initial_state)
        except Exception as exc:
            logger.error(
                f"✗ 状态机异常 [{file_path}]：{exc}",
                exc_info=True,
            )
            continue

        opinions = final_state.get("structured_opinions", [])
        retry_used = final_state.get("retry_count", 0)
        logger.info(
            f"  ✓ 完成  意见数={len(opinions)}  重试次={retry_used}"
        )
        total_opinions += len(opinions)

        if not opinions:
            if retry_used >= MAX_RETRY_COUNT:
                logger.warning(
                    f"  ⚠️  达到重试上限，输出格式始终无效。"
                )
            else:
                logger.info("  ✓ 未发现问题。")
            continue

        # 回传评论
        for opinion in opinions:
            line_number = int(opinion["line_number"])
            inline_ok = _post_inline_discussion(
                GITLAB_URL,
                PROJECT_ID,
                MR_IID,
                PROJECT_TOKEN,
                (
                    "🤖 **AI Code Review**\n\n"
                    f"⚠️ **类别**：`{opinion['category']}` | "
                    f"**级别**：`{opinion['level']}`\n\n"
                    f"💡 **评审意见**：{opinion['comment']}"
                ),
                change,
                line_number,
                BASE_SHA,
                HEAD_SHA,
            )
            if inline_ok:
                total_inline += 1
                total_posted += 1
                logger.info(
                    f"✓ 行间评论已发表  {file_path}:{line_number}  "
                    f"({opinion['category']})"
                )
                continue

            # 降级处理
            logger.warning(
                f"行间定位失败，尝试降级  {file_path}:{line_number}"
            )
            fallback_body = (
                "🤖 **AI Code Review** *(行定位失败，已降级)*\n\n"
                f"📄 **文件**：`{file_path}`  **行号**：`{line_number}`\n\n"
                f"⚠️ **类别**：`{opinion['category']}` | "
                f"**级别**：`{opinion['level']}`\n\n"
                f"💡 **评审意见**：{opinion['comment']}"
            )
            posted = _post_mr_note(
                GITLAB_URL, PROJECT_ID, MR_IID, PROJECT_TOKEN, fallback_body
            )
            if posted:
                total_fallback += 1
                total_posted += 1
                logger.info(
                    f"✓ 顶层评论已发表（降级）  {file_path}:{line_number}"
                )

    # 完成汇总
    logger.info(f"\n{DIVIDER}")
    logger.info("  AI Code Review  完成")
    logger.info(f"  处理文件数：{total_files}")
    logger.info(f"  生成意见数：{total_opinions}")
    logger.info(
        f"  回传评论数：{total_posted}  "
        f"(行内: {total_inline}, 降级: {total_fallback})"
    )
    logger.info(DIVIDER)


if __name__ == "__main__":
    main()
