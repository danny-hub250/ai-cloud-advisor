from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from utils.model import get_client, get_model_name
from agents.tools import research_tool, cost_estimate_tool
from agents.prompts import SYSTEM_ROLE, COT_HINT, PLANNER_FEWSHOTS

def llm_chat(messages):
    client = get_client()
    model = get_model_name()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": ("system" if isinstance(m, SystemMessage) else "user" if isinstance(m, HumanMessage) else "assistant"), "content": m.content} for m in messages],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def node_planner(state: Dict[str, Any]):
    user_input = state.get("user_input","")
    msgs = [SystemMessage(content=SYSTEM_ROLE)]
    for ex in PLANNER_FEWSHOTS:
        if ex["role"] == "user":
            msgs.append(HumanMessage(content=ex["content"]))
        else:
            msgs.append(SystemMessage(content=ex["content"]))
    msgs.append(HumanMessage(content=f"{user_input}\n{COT_HINT}"))
    state["plan"] = llm_chat(msgs)
    return state

def node_research(state: Dict[str, Any]):
    query = f"{state.get('user_input','')} {state.get('plan','')[:200]}"
    state["evidence"] = research_tool(query, k=4)
    return state

def node_cost(state: Dict[str, Any]):
    requirements = {
        "monthly_users": state.get("monthly_users", 100000),
        "region": state.get("region","apac"),
        "storage_gb": state.get("storage_gb", 200),
        "compute_tier": state.get("compute_tier","medium"),
    }
    state["cost"] = cost_estimate_tool(requirements)
    return state

def node_final(state: Dict[str, Any]):
    ev = state.get("evidence", [])
    citations = "\n".join([f"[{i+1}] ({h['source']}) score={h['score']:.3f} -> {h['text'][:180]}..." for i, h in enumerate(ev)])
    plan = state.get("plan","")
    cost = state.get("cost",{})
    prompt = f"""You are CloudAdvisor. Given:
- User input: {state.get('user_input','')}
- Industry: {state.get('industry', 'general')}
- Compliance requirements: {state.get('compliance') or 'none specified'}
- Existing cloud: {state.get('existing_cloud') or 'none'}
- Plan: {plan}
- Evidence: {citations}
- Cost estimates: {cost}

Recommend Azure or AWS with rationale (performance, fit, security/compliance, ops), propose a minimal target architecture, and provide next steps.
Return strict JSON with keys: provider, rationale, architecture, monthly_cost_estimate, next_steps.
"""
    out = llm_chat([SystemMessage(content=prompt)])
    state["final"] = out
    return state

def build_graph():
    sg = StateGraph(dict)
    sg.add_node("plan", node_planner)
    sg.add_node("research", node_research)
    sg.add_node("cost", node_cost)
    sg.add_node("final", node_final)
    sg.set_entry_point("plan")
    sg.add_edge("plan","research")
    sg.add_edge("research","cost")
    sg.add_edge("cost","final")
    sg.add_edge("final", END)
    return sg.compile()
