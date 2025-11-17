# llm_router.py
import json
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
3. Use realistic lab assumptions (e.g. mL/µL, mg/mL, mM) and explain units in your reasoning.
4. If the question is purely conceptual or about storage/stability, you may still call a tool that summarizes guidance.
5. If tool output is provided to you, explain it step-by-step and give a clear final answer.

Be precise and explicit about assumptions.
"""

def call_llm(messages, tools=None, tool_choice="auto"):
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # or gpt-4.1, gpt-4o, etc.
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )
    return resp

def handle_user_query(user_text: str, history_messages: list[dict]):
    """
    history_messages: list of dicts [{role, content}, ...] *excluding* system.
    Returns: assistant_text, optional_tool_name, optional_tool_output
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history_messages
    messages.append({"role": "user", "content": user_text})

    first = call_llm(messages, tools=LLM_TOOLS, tool_choice="auto")
    msg = first.choices[0].message

    # CASE 1: normal answer (no tools)
    if not msg.tool_calls:
        assistant_text = msg.content
        return assistant_text, None, None

    # CASE 2: tool call
    tool_call = msg.tool_calls[0]
    tool_name = tool_call.function.name
    raw_args = tool_call.function.arguments

    try:
        args = json.loads(raw_args)
    except Exception:
        args = {}

    # run backend calculator
    tool_fn = CALC_REGISTRY.get(tool_name)
    if tool_fn is None:
        tool_output = {"error": f"Unknown tool: {tool_name}"}
    else:
        try:
            tool_output = tool_fn(**args)
        except Exception as e:
            tool_output = {"error": str(e), "args": args}

    # now get explanation
    tool_message = {
        "role": "tool",
        "name": tool_name,
        "content": json.dumps(tool_output),
    }

    messages.append(msg)          # assistant's tool_call message
    messages.append(tool_message) # tool result

    second = call_llm(
        messages,
        tools=None,  # no more tools; just explain
        tool_choice="none",
    )
    assistant_text = second.choices[0].message.content

    return assistant_text, tool_name, tool_output
