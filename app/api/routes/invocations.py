import logging
import traceback
import boto3

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.schemas.agent import SupervisorRequest
from app.services.supervisor_agent import SupervisorAgent
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/invocations")
async def invocations(request: Request):
    import json
    raw = await request.body()
    try:
        data = json.loads(raw)
        req = SupervisorRequest(**data)
    except Exception as e:
        logger.error(f"[INVOCATIONS] Request parse failed: {e} | raw={raw[:500]}")
        return JSONResponse(status_code=422, content={"error": f"Request parse error: {e}"})

    try:
        agent = SupervisorAgent()
        result = await agent.run(req)
        return JSONResponse(content={
            "cognito_id": result.cognito_id,
            "response": result.response,
        })
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"[{req.cognito_id}] error: {error_detail}")
        return JSONResponse(content={
            "cognito_id": req.cognito_id,
            "response": f"[DEBUG ERROR] boto3={boto3.__version__} | {type(e).__name__}: {str(e)} | QUESTION_ARN={settings.QUESTION_AGENT_ARN[:20] if settings.QUESTION_AGENT_ARN else 'EMPTY'}",
        })
