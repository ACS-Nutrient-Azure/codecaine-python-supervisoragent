import logging
import traceback
import boto3

from fastapi import APIRouter
from app.schemas.agent import SupervisorRequest, SupervisorResponse
from app.services.supervisor_agent import SupervisorAgent
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/invocations", response_model=SupervisorResponse)
async def invocations(req: SupervisorRequest):
    try:
        agent = SupervisorAgent()
        return await agent.run(req)
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"[{req.cognito_id}] error: {error_detail}")
        return SupervisorResponse(
            cognito_id=req.cognito_id,
            response=f"[DEBUG ERROR] boto3={boto3.__version__} | {type(e).__name__}: {str(e)} | QUESTION_ARN={settings.QUESTION_AGENT_ARN[:20] if settings.QUESTION_AGENT_ARN else 'EMPTY'}"
        )
