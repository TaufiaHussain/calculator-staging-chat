# llm_router.py
"""
Router between the chat UI, OpenAI tools, and the backend calculators.

- Uses OpenAI function-calling (tools) with LLM_TOOLS (29 tools).
- Picks EXACTLY one tool for each user question.
- Calls the matching function from CALC_REGISTRY.
- Then asks the LLM to explain the tool output in plain lab language.

Used by app.py in the Tier-5 chat tab.
"""

import json
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

from llm_tools import LLM_TOOLS
from calculators import CALC_REGISTRY

# ---------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a lab solution assistant for the 'Versatile Lab Solution Calculator'.

Your goals:
1. Understand the user's request.
2. If a calculator is appropriate, choose exactly ONE function from the available tools and provide all parameters.
3. Use realistic lab units (µL, mL, L, mg/mL, mM, µM, %, OD, etc.).
4. Always convert units correctly before calling a tool.
   Example:
   - 10 mM stock and 10 µM target → convert 10 mM = 10_000 µM if the tool expects µM.
5. If the question is conceptual or about storage/stability, you may still call the closest helper tool.
6. When tool output is provided to you, explain it step-by-step and give a clear final answer,
   including:
   - what to pipette
   - which units
   - any safety or stability notes if relevant.

Be precise and explicit about assumptions and units.
Only call ONE tool per user request.
"""

# ---------------------------------------------------------------------
# Internal helper to call the OpenAI chat API
# ---------------------------------------------------------------------
def _call_llm(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
):
    """Thin wrapper around client.chat.completions.create."""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",   # adjust if you want a different model
        temperature=0.0,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )
    return resp


# ---------------------------------------------------------------------
# Public function used by app.py
# ---------------------------------------------------------------------
def handle_user_query(
    user_text: str,
    history_messages: List[Dict[str, str]],
) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """
    Main entrypoint for the chat tab.

    Parameters
    ----------
    user_text : str
        The latest user message from the chat input.
    history_messages : list of {role, content}
        Previous turns in the conversation, *excluding* the system prompt.
        Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Returns
    -------
    assistant_text : str
        Final assistant text to show in the chat.
    tool_name : str or None
        Name of the tool that was called (e.g. 'single_dilution'), if any.
    tool_output : dict or None
        Raw JSON output returned by the calculator function, if any.
    """
    # 1) Build full message history including system prompt
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_text})

    # 2) First pass: let the model decide whether to call a tool
    first = _call_llm(messages, tools=LLM_TOOLS, tool_choice="auto")
    msg = first.choices[0].message

    # -----------------------------------------------------------------
    # CASE 1: Model chooses NOT to call a tool
    # -----------------------------------------------------------------
    if not msg.tool_calls:
        assistant_text = msg.content or ""
        return assistant_text, None, None

    # -----------------------------------------------------------------
    # CASE 2: Model calls exactly one tool (we enforce this in the prompt)
    # -----------------------------------------------------------------
    tool_call = msg.tool_calls[0]
    tool_name: str = tool_call.function.name
    raw_args: str = tool_call.function.arguments or "{}"

    # Parse arguments JSON-safely
    try:
        args: Dict[str, Any] = json.loads(raw_args)
    except Exception:
        args = {}

    # Look up calculator function
    tool_fn = CALC_REGISTRY.get(tool_name)
    if tool_fn is None:
        tool_output: Dict[str, Any] = {"error": f"Unknown tool: {tool_name}", "args": args}
    else:
        try:
            tool_output = tool_fn(**args)
        except Exception as e:
            tool_output = {"error": str(e), "args": args}

    # 3) Second pass: ask the LLM to explain the tool result to the user
    tool_message = {
        "role": "tool",
        "name": tool_name,
        "content": json.dumps(tool_output),
    }

    # Append the tool_call message and the tool result
    messages.append(
        {
            "role": "assistant",
            "tool_calls": msg.tool_calls,
            "content": msg.content or "",
        }
    )
    messages.append(tool_message)

    second = _call_llm(messages, tools=None, tool_choice="none")
    assistant_text = second.choices[0].message.content or ""

    return assistant_text, tool_name, tool_output
