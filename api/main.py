from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from agents.graph import build_graph

app = FastAPI(title="AI Cloud Advisor API")
graph = build_graph()

class AdviseRequest(BaseModel):
    prompt: str = Field(..., description="User requirements and constraints")
    monthly_users: Optional[int] = 100000
    region: Optional[str] = "apac"
    storage_gb: Optional[int] = 200
    compute_tier: Optional[str] = "medium"
    industry: Optional[str] = "general"
    compliance: Optional[str] = None
    existing_cloud: Optional[str] = None

@app.post("/advise")
def advise(req: AdviseRequest) -> Dict[str, Any]:
    state = {
        "user_input": req.prompt,
        "monthly_users": req.monthly_users,
        "region": req.region,
        "storage_gb": req.storage_gb,
        "compute_tier": req.compute_tier,
        "industry": req.industry,
        "compliance": req.compliance,
        "existing_cloud": req.existing_cloud,
    }
    result = graph.invoke(state)
    return {"result": result.get("final")}
