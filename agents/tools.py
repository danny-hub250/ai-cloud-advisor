from typing import List, Dict, Any
from rag.retriever import rag_search

def cost_estimate_tool(requirements: Dict[str, Any]) -> Dict[str, Any]:
    tier_map = {"small": 500, "medium": 2000, "large": 5000}
    base = tier_map.get(str(requirements.get("compute_tier", "small")).lower(), 500)
    storage_cost = float(requirements.get("storage_gb", 100)) * 0.023
    egress_cost = 150.0 if str(requirements.get("region","")).lower() not in ("us","eu") else 100.0
    monthly_users = int(requirements.get("monthly_users", 100000))
    cdn = 0.05 * (monthly_users / 1000)
    return {
        "azure_estimate_usd": round(base * 1.05 + storage_cost + egress_cost + cdn, 2),
        "aws_estimate_usd": round(base * 1.0 + storage_cost + egress_cost + cdn, 2),
        "notes": "Heuristic demo — replace with real calculators."
    }

def research_tool(query: str, k: int = 4) -> List[Dict[str, Any]]:
    results = rag_search(query, k=k)
    return [{"source": r.metadata.get("source","local"), "score": r.score, "text": r.page_content} for r in results]
