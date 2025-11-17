# llm_router.py
"""
Router between the chat UI, OpenAI tools, and the backend calculators.
"""

import json
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

from llm_tools import LLM_TOOLS
from calculators import CALC_REGISTRY

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM_PROMPT = """
You are a lab solution assistant for the 'Versatile Lab Solution Calculator'.

Your goals:
1. Understand the user's request.
2. If a calculator is appropriate, choose exactly ONE function from the available tools and provide all parameters.
3. Use realistic lab units (µL, mL, L, mg/mL, mM, µM, %, OD, etc.).
4. Always convert units correctly before calling a tool.
5. If the question is conceptual or about storage/stability, you may still call the closest helper tool.
6. When tool output is provided to you, explain it step-by-step and give a clear final answer.

Be precise and explicit about assumptions and units.
Only call ONE tool per user request.
"""

# ---------------------------------------------------------------------
# Helper to call OpenAI
# ---------------------------------------------------------------------
def _call_llm(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
):
    """Thin wrapper around client.chat.completions.create."""
    kwargs: Dict[str, Any] = {
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "messages": messages,
    }
    if tools is not None:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    resp = client.chat.completions.create(**kwargs)
    return resp


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def handle_user_query(
    user_text: str,
    history_messages: List[Dict[str, str]],
) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_text})

    # 1) First pass: let model decide whether to call a tool
    first = _call_llm(messages, tools=LLM_TOOLS, tool_choice="auto")
    msg = first.choices[0].message

    # No tool call → just text reply
    if not msg.tool_calls:
        assistant_text = msg.content or ""
        return assistant_text, None, None

    # 2) Tool call
    tool_call = msg.tool_calls[0]
    tool_name: str = tool_call.function.name
    raw_args: str = tool_call.function.arguments or "{}"
    tool_call_id: str = tool_call.id  # needed for the tool role message

    try:
        args: Dict[str, Any] = json.loads(raw_args)
    except Exception:
        args = {}

    tool_fn = CALC_REGISTRY.get(tool_name)
    if tool_fn is None:
        tool_output: Dict[str, Any] = {"error": f"Unknown tool: {tool_name}", "args": args}
    else:
        try:
            tool_output = tool_fn(**args)
        except Exception as e:
            tool_output = {"error": str(e), "args": args}

    # 3) Second pass: ask LLM to explain the result (NO tools here)
    # We must echo the assistant's tool_call and then give a tool message
    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": raw_args,
                    },
                }
            ],
        }
    )

    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(tool_output),
    }
    messages.append(tool_message)

    # Important: second call has no tools / tool_choice
    second = _call_llm(messages)
    assistant_text = second.choices[0].message.content or ""

    return assistant_text, tool_name, tool_output
