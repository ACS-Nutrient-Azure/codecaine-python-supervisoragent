from pydantic import BaseModel


class SupervisorRequest(BaseModel):
    cognito_id: str
    chat_result_id: int
    codef_health_data: dict | None = None
    codef_medication_info: list[dict] | None = None
    chat_history: str


class SupervisorResponse(BaseModel):
    cognito_id: str
    response: str
