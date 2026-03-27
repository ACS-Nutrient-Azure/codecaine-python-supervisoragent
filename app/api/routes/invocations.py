from fastapi import APIRouter, HTTPException
from app.schemas.agent import SupervisorRequest, SupervisorResponse
from app.services.supervisor_agent import SupervisorAgent

router = APIRouter()


@router.post("/invocations", response_model=SupervisorResponse)
async def invocations(req: SupervisorRequest):
    try:
        agent = SupervisorAgent()
        return await agent.run(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
